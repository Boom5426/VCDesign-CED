#!/usr/bin/env python3
"""Phase 8: does pooling perturbation atlases improve candidate effect distillation?

Three predictors, one recipe, one shared gene space.  None of them is told which
context a candidate came from, so this tests atlas size rather than context
conditioning.

    ANCHOR_ONLY    trained on K562 G_fit consensus responses
    EXTERNAL_ONLY  trained on RPE1 E_fit consensus responses
    POOLED         trained on both, concatenated, with no context feature

All three are evaluated in both contexts, fused into that context's own frozen base
ranker at the frozen unit weight.

Leakage.  A candidate's own measured response in the context it is evaluated in must
never have fitted the weights that predict it, so predictions for candidates inside a
predictor's own training pool are out of fold.  The rank-256 basis is fit on the whole
training pool rather than per fold; a single identity contributes about ``1/N`` of that
subspace and refitting it five times per predictor was judged not worth the compute.
That residual is declared, not hidden.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ..candidate_effect_distillation_v1 import basis as bs
from ..candidate_effect_distillation_v1 import contract as ced
from ..candidate_effect_distillation_v1 import predictor as pr
from ..final_clean_model_v1.model import CleanAttributionModel
from ..final_clean_model_v1.scoring import base_scores, fuse, strict_legal_dims, strict_legal_role
from ..model.decision_aligned_sampler_v1.evaluate import _direction_metrics
from ..model.decision_alignment_v1.assets import AuditConfig
from ..model.global_objective_knowledge_attribution_v1.assets import GlobalAttributionAssets
from ..model.open_vocab_dual_encoder_v1.data import FrozenTransitionAssets
from ..utility_gt_v2.validate import cluster_bootstrap, summarize
from . import contract as ec
from .transfer import cross_view_cosine, embed_into_anchor, external_capabilities, gene_bridge


PREDICTORS = ("ANCHOR_ONLY", "EXTERNAL_ONLY", "POOLED")


def _unit(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-30)


def fit_predictor(features: np.ndarray, responses: np.ndarray, present: np.ndarray,
                  keys: np.ndarray, rank: int = ced.BASIS_RANK) -> dict:
    """The V1 recipe, unchanged, on whatever training pool it is handed.

    ``keys`` tags each training row with ``"<context>|<identity>"`` so that a later
    out-of-fold prediction can remove exactly the row that would otherwise let a
    candidate's own measured response in the evaluated context fit its own prediction.
    """
    keep = np.flatnonzero(present)
    basis = bs.randomized_basis(responses[keep], rank=rank)
    targets = basis.project(responses[keep])
    record = pr.cross_validate(features[keep], targets, responses[keep], basis)
    penalty = float(record["selected_penalty"])
    return {"basis": basis, "keep": keep, "targets": targets, "penalty": penalty,
            "features": features, "keys": np.asarray(keys, dtype=str),
            "path": pr.RidgePath(features[keep], targets),
            "cross_validation": {k: v for k, v in record.items()
                                 if k != "held_out_cosine_distribution"},
            "held_out_cosine": record["held_out_cosine_at_selected"]}


def honest_effect(model: dict, features: np.ndarray, present: np.ndarray,
                  keys: np.ndarray | None) -> np.ndarray:
    """Predicted effect where no candidate's own response in this context fitted it.

    ``keys`` is ``None`` when the predictor saw no response from the evaluated
    context at all, in which case the full model is the honest one and there is
    nothing to hold out.
    """
    effect = model["basis"].reconstruct(model["path"].predict(features, model["penalty"]))
    if keys is not None:
        training_keys = model["keys"][model["keep"]]
        position = {name: row for row, name in enumerate(training_keys.tolist())}
        member = np.asarray([position.get(name, -1) for name in np.asarray(keys, dtype=str).tolist()])
        for held in pr.identity_folds(model["keep"].size):
            inside = np.isin(member, held)
            rows = np.flatnonzero(inside & (member >= 0))
            if rows.size == 0:
                continue
            train = np.setdiff1d(np.arange(model["keep"].size), held)
            path = pr.RidgePath(model["features"][model["keep"]][train], model["targets"][train])
            effect[rows] = model["basis"].reconstruct(path.predict(features[rows], model["penalty"]))
    effect[~present] = 0.0
    return effect


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    config = AuditConfig.load(Path(args.config))
    assets = GlobalAttributionAssets(modalities=("STRING", "MAPKG", "TEXT"), **config.asset_arguments())
    dims = strict_legal_dims(assets)
    fit_role = strict_legal_role(assets.role_arrays("G_fit"))
    select_role = strict_legal_role(assets.role_arrays("G_select"))
    anchor_genes = np.asarray([str(v) for v in np.load(args.anchor_genes, allow_pickle=True)])

    bench = Path(args.benchmark)
    ids_all = np.asarray([str(v) for v in np.load(bench / "identity_axis.npy", allow_pickle=True)])
    usable = np.load(bench / "usable.npy")
    eligible = np.load(bench / "eligible.npy")
    roles = np.asarray([str(v) for v in np.load(bench / "roles.npy", allow_pickle=True)])
    responses_all = np.load(bench / "side_responses.npy")
    sources_all = np.load(bench / "side_sources.npy")
    external_genes = np.asarray([str(v) for v in np.load(bench / "gene_axis.npy", allow_pickle=True)])
    bridge = gene_bridge(external_genes, anchor_genes)

    # Training pools, both restricted to the shared gene space.
    anchor_fit_ids = np.asarray([str(v) for v in fit_role.base.ids])
    anchor_consensus = bs.consensus_response(
        np.asarray(fit_role.base.transitions, dtype=np.float64))[:, bridge["anchor_positions"]]
    anchor_features = pr.static_features(fit_role, ced.PRIMARY_MODALITIES)

    external_fit = np.flatnonzero(usable & (roles == "E_fit"))
    external_fit_ids = ids_all[external_fit]
    external_consensus = bs.consensus_response(
        np.asarray(responses_all[:, external_fit], dtype=np.float64))[:, bridge["external_positions"]]
    ext_caps, ext_present, check = external_capabilities(assets, external_fit_ids)
    external_features = pr.static_features(
        SimpleNamespace(capabilities=ext_caps, capability_present=ext_present), ced.PRIMARY_MODALITIES)

    anchor_keys = np.asarray([f"K562|{name}" for name in anchor_fit_ids.tolist()], dtype=str)
    external_keys = np.asarray([f"RPE1|{name}" for name in external_fit_ids.tolist()], dtype=str)
    models = {
        "ANCHOR_ONLY": fit_predictor(anchor_features.values, anchor_consensus,
                                     anchor_features.present_any, anchor_keys),
        "EXTERNAL_ONLY": fit_predictor(external_features.values, external_consensus,
                                       external_features.present_any, external_keys),
        "POOLED": fit_predictor(np.concatenate([anchor_features.values, external_features.values]),
                                np.concatenate([anchor_consensus, external_consensus]),
                                np.concatenate([anchor_features.present_any, external_features.present_any]),
                                np.concatenate([anchor_keys, external_keys])),
    }

    record: dict = {
        "schema": "VCDESIGN_EXTERNAL_CONTEXT_V1_JOINT",
        "read_K562_G_check": False, "context_feature_given_to_the_model": False,
        "shared_gene_space": bridge["size"],
        "training_pools": {"ANCHOR_ONLY": int(anchor_features.present_any.sum()),
                           "EXTERNAL_ONLY": int(external_features.present_any.sum()),
                           "POOLED": int(anchor_features.present_any.sum() +
                                         external_features.present_any.sum())},
        "candidate_feature_verification": check,
        "basis_note": "the rank-256 basis is fit on the whole training pool, not per fold",
        "predictors": {name: {"penalty": model["penalty"],
                              "held_out_gene_space_cosine": model["held_out_cosine"],
                              "explained_energy": model["basis"].explained}
                       for name, model in models.items()},
        "contexts": {},
    }

    # --- external context -----------------------------------------------------
    pool = np.flatnonzero(usable)
    ids = ids_all[pool]
    responses = responses_all[:, pool]
    sources = sources_all[:, pool]
    rows = np.flatnonzero(eligible[pool])
    caps, present, _ = external_capabilities(assets, ids)
    features = pr.static_features(SimpleNamespace(capabilities=caps, capability_present=present),
                                  ced.PRIMARY_MODALITIES)
    external_eval_keys = np.asarray([f"RPE1|{name}" for name in ids.tolist()], dtype=str)
    utility = [cross_view_cosine(responses, view) for view in (0, 1)]

    model = CleanAttributionModel(dims).to(device)
    lock = json.loads(Path(args.lock).read_text())
    model.load_state_dict(torch.load(lock["selected_checkpoints"]["CLEAN_BASE"]["checkpoint"],
                                     map_location=device, weights_only=False)["model_state"], strict=True)
    model.eval()
    descriptors = {name: torch.from_numpy(caps[name]).to(device) for name in ced.PRIMARY_MODALITIES}
    masks = {name: torch.from_numpy(present[name]).to(device) for name in ced.PRIMARY_MODALITIES}
    base = []
    with torch.no_grad():
        encoding = model.encode_candidates(descriptors, masks)
        for view in (0, 1):
            source = torch.from_numpy(embed_into_anchor(sources[view], bridge, anchor_genes.size)).to(device)
            goal = torch.from_numpy(embed_into_anchor(sources[view] + responses[view], bridge,
                                                      anchor_genes.size)).to(device)
            block = [model.score_embeddings(model.encode_query(source[s:s + 64], goal[s:s + 64]),
                                            encoding).float().cpu() for s in range(0, ids.size, 64)]
            base.append(torch.cat(block).numpy().astype(np.float64))

    external_block = {"queries": int(rows.size), "candidates": int(ids.size), "rows": {}, "paired": {}}
    stored = {}
    for name, matrices in [("BASE", base)] + [
            (f"BASE_PLUS_{key}", None) for key in PREDICTORS]:
        if matrices is None:
            key = name.replace("BASE_PLUS_", "")
            # ANCHOR_ONLY never saw an RPE1 response, so nothing needs holding out.
            effect = honest_effect(models[key], features.values, features.present_any,
                                   None if key == "ANCHOR_ONLY" else external_eval_keys)
            unit = _unit(effect)
            gone = np.linalg.norm(effect, axis=1) <= 0.0
            cosine = []
            for view in (0, 1):
                block = _unit(np.asarray(responses[view], dtype=np.float64)[:, bridge["external_positions"]]) @ unit.T
                block[:, gone] = 0.0
                cosine.append(block)
            matrices = fuse(base, cosine)
        per_view = [_direction_metrics(matrices[view], utility[view], ids, rows) for view in (0, 1)]
        external_block["rows"][name] = {key: summarize(np.concatenate([e[key] for e in per_view]))
                                        for key in per_view[0]}
        stored[name] = np.concatenate([e["MBRU"] for e in per_view])
    for key in PREDICTORS:
        difference = stored[f"BASE_PLUS_{key}"] - stored["BASE"]
        external_block["paired"][f"BASE_PLUS_{key}_minus_BASE"] = {
            **summarize(difference), "bootstrap": cluster_bootstrap(difference),
            "fraction_query_improved": float((difference > 0).mean())}
    difference = stored["BASE_PLUS_POOLED"] - stored["BASE_PLUS_ANCHOR_ONLY"]
    external_block["paired"]["POOLED_minus_ANCHOR_ONLY"] = {
        **summarize(difference), "bootstrap": cluster_bootstrap(difference),
        "fraction_query_improved": float((difference > 0).mean())}
    record["contexts"]["RPE1"] = external_block
    del model

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record["runtime"] = {"timestamp": datetime.now(timezone.utc).isoformat(),
                         "python": platform.python_version(), "numpy": np.__version__}
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 8: joint-context distillation")
    for name in ("config", "benchmark", "lock", "anchor_genes", "output"):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, required=True)
    parser.add_argument("--device", default="cuda")
    record = run(parser.parse_args())
    print(f"shared gene space {record['shared_gene_space']}; training pools {record['training_pools']}")
    for name, entry in record["predictors"].items():
        print(f"  {name:14s} penalty {entry['penalty']:g}  held-out cosine "
              f"{entry['held_out_gene_space_cosine']:.4f}  basis explains {entry['explained_energy']:.4f}")
    for context, block in record["contexts"].items():
        print(f"\n[{context}] {block['queries']} queries, {block['candidates']} candidates")
        for name, entry in block["rows"].items():
            print(f"  {name:24s} MBRU {entry['MBRU']['median']:+.4f}  Mean@10 {entry['Mean@10']['median']:+.4f}  "
                  f"Mean@20 {entry['Mean@20']['median']:+.4f}  HVC@20 {entry['HighValueCount@20']['median']:.1f}")
        for key, item in block["paired"].items():
            low, high = item["bootstrap"]["median"]["ci95"]
            print(f"    {key:42s} median {item['median']:+.4f} [{low:+.4f},{high:+.4f}] "
                  f"improved {item['fraction_query_improved']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
