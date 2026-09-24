#!/usr/bin/env python3
"""Phase 6: zero-shot transfer of the frozen anchor model into RPE1.

Nothing is trained.  No RPE1 response is read by any model input.  The anchor's
CLEAN_BASE checkpoint and the anchor's effect predictor are both frozen, and the two
arms share everything except the effect branch, so their difference is that branch
and nothing else.

Context generalization and candidate generalization are reported separately, never
pooled: a candidate that appeared in the anchor's ``G_fit`` had its anchor response
seen by the effect predictor, so it tests context transfer; a candidate absent from
``G_fit`` tests candidate transfer as well.
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

from ..candidate_effect_distillation_v1 import contract as ced
from ..final_clean_model_v1.model import CleanAttributionModel
from ..final_clean_model_v1.scoring import FrozenEffectPredictor, fuse, strict_legal_dims, strict_legal_role
from ..model.decision_aligned_sampler_v1.evaluate import HIGH_VALUE_QUANTILE, _direction_metrics
from ..model.decision_alignment_v1.assets import AuditConfig
from ..model.decision_alignment_v1.contract import BUDGETS
from ..model.global_objective_knowledge_attribution_v1.assets import GlobalAttributionAssets
from ..utility_gt_v2.validate import cluster_bootstrap, summarize
from . import contract as ec
from .transfer import cross_view_cosine, embed_into_anchor, external_capabilities, gene_bridge


LEGALITY = {
    "RANDOM": "no information",
    "ANCHOR_EFFECT_MAGNITUDE": "strict legal candidate knowledge, no goal",
    "EFFECT_BRANCH_ONLY": "strict legal, zero shot",
    "ZS_BASE": "frozen anchor base ranker, zero shot",
    "ZS_EFFECT": "frozen anchor base plus frozen anchor effect branch, zero shot",
    "ZS_BASE_PLUS_MAGNITUDE": "ablation: the same fusion with the goal-blind magnitude prior",
    "ZS_BASE_PLUS_RANDOM": "ablation: the same fusion with a random score",
    "MEASURED_RESPONSE_ORACLE": "DIAGNOSTIC_ORACLE_NOT_DEPLOYABLE, privileged upper bound",
}


def _unit(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-30)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    config = AuditConfig.load(Path(args.config))
    assets = GlobalAttributionAssets(modalities=("STRING", "MAPKG", "TEXT"), **config.asset_arguments())
    dims = strict_legal_dims(assets)
    fit_role = strict_legal_role(assets.role_arrays("G_fit"))
    anchor_fit_ids = set(str(v) for v in fit_role.base.ids)

    bench = Path(args.benchmark)
    ids_all = np.asarray([str(v) for v in np.load(bench / "identity_axis.npy", allow_pickle=True)])
    usable = np.load(bench / "usable.npy")
    eligible = np.load(bench / "eligible.npy")
    responses_all = np.load(bench / "side_responses.npy")
    sources_all = np.load(bench / "side_sources.npy")
    external_genes = np.asarray([str(v) for v in np.load(bench / "gene_axis.npy", allow_pickle=True)])

    pool = np.flatnonzero(usable)
    ids = ids_all[pool]
    responses = responses_all[:, pool]
    sources = sources_all[:, pool]
    rows = np.flatnonzero(eligible[pool])

    anchor_genes = np.asarray([str(v) for v in np.load(args.anchor_genes, allow_pickle=True)])
    bridge = gene_bridge(external_genes, anchor_genes)

    capabilities, present, feature_check = external_capabilities(assets, ids)
    if feature_check.get("STRING_max_abs_error", 0.0) > 1e-4 or \
            feature_check.get("MAPKG_max_abs_error", 0.0) > 1e-4:
        raise RuntimeError(f"external candidate features do not reproduce the anchor's: {feature_check}")

    lock = json.loads(Path(args.lock).read_text())
    predictor = FrozenEffectPredictor(fit_role, float(lock["effect_predictor"]["ridge_penalty"]))
    saved = np.load(Path(args.distillation) / "predicted_effect_G_select.npy")
    reproduction = predictor.verify_against(saved, strict_legal_role(assets.role_arrays("G_select")))
    if reproduction > 1e-3:
        raise RuntimeError(f"the effect predictor is not the locked one ({reproduction:.3g})")

    external_role = SimpleNamespace(capabilities=capabilities, capability_present=present)
    effect_anchor = predictor.effect(external_role)          # 8248-dim, anchor axis
    effect_common = effect_anchor[:, bridge["anchor_positions"]]
    magnitude = np.linalg.norm(effect_common, axis=1)
    missing = ~(present["STRING"] | present["MAPKG"])
    effect_common[missing] = 0.0
    magnitude[missing] = 0.0

    utility = [cross_view_cosine(responses, view) for view in (0, 1)]

    model = CleanAttributionModel(dims).to(device)
    model.load_state_dict(torch.load(lock["selected_checkpoints"]["CLEAN_BASE"]["checkpoint"],
                                     map_location=device, weights_only=False)["model_state"], strict=True)
    model.eval()
    descriptors = {name: torch.from_numpy(capabilities[name]).to(device) for name in ced.PRIMARY_MODALITIES}
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
    del model

    effect_matrices = []
    for view in (0, 1):
        query = _unit(np.asarray(responses[view], dtype=np.float64)[:, bridge["external_positions"]])
        block = query @ _unit(effect_common).T
        block[:, missing] = 0.0
        effect_matrices.append(block)

    generator = np.random.default_rng(ec.SPLIT_SEED)
    scores = {
        "RANDOM": [generator.standard_normal((ids.size, ids.size)) for _ in (0, 1)],
        "ANCHOR_EFFECT_MAGNITUDE": [np.broadcast_to(magnitude, (ids.size, ids.size)).copy() for _ in (0, 1)],
        "EFFECT_BRANCH_ONLY": effect_matrices,
        "ZS_BASE": base,
        "ZS_EFFECT": fuse(base, effect_matrices),
        "MEASURED_RESPONSE_ORACLE": [
            _unit(np.asarray(responses[view], dtype=np.float64)) @
            _unit(np.asarray(responses[view], dtype=np.float64)).T for view in (0, 1)],
    }
    # The two ablations that could explain the fusion gain more cheaply than the
    # goal-conditioned effect geometry does.
    scores["ZS_BASE_PLUS_MAGNITUDE"] = fuse(base, scores["ANCHOR_EFFECT_MAGNITUDE"])
    scores["ZS_BASE_PLUS_RANDOM"] = fuse(base, [generator.standard_normal((ids.size, ids.size))
                                                for _ in (0, 1)])

    seen = np.asarray([name in anchor_fit_ids for name in ids.tolist()])
    strata = {"seen_in_K562_distillation": np.flatnonzero(seen[rows]),
              "unseen_in_K562_distillation": np.flatnonzero(~seen[rows]),
              "all_candidates": np.arange(rows.size)}

    record: dict = {
        "schema": "VCDESIGN_EXTERNAL_CONTEXT_V1_ZERO_SHOT",
        "context": ec.PRIMARY_CONTEXT, "anchor": ec.ANCHOR_CONTEXT,
        "trained_anything": False, "read_external_candidate_response_as_input": False,
        "read_K562_G_check": False,
        "transfer_convention": ec.TRANSFER_CONVENTION,
        "gene_bridge": {k: v for k, v in bridge.items() if k != "common" and not k.endswith("positions")},
        "candidate_feature_verification": feature_check,
        "effect_predictor_reproduction_max_abs_error": reproduction,
        "pool": {"identities_total": int(ids_all.size), "usable_pool": int(ids.size),
                 "eligible_queries": int(rows.size),
                 "queries_seen_in_anchor_distillation": int(seen[rows].sum()),
                 "queries_unseen": int((~seen[rows]).sum()),
                 "candidates_seen_in_anchor_distillation": int(seen.sum()),
                 "candidates_with_no_legal_modality": int(missing.sum())},
        "high_value_quantile": HIGH_VALUE_QUANTILE, "budgets": list(BUDGETS),
        "strata": {},
    }

    for stratum, subset in strata.items():
        if subset.size == 0:
            continue
        query_rows = rows[subset]
        block = {"queries": int(query_rows.size), "rows": {}}
        stored = {}
        for name, matrices in scores.items():
            per_view = [_direction_metrics(matrices[view], utility[view], ids, query_rows) for view in (0, 1)]
            block["rows"][name] = {
                "legality": LEGALITY[name],
                "direction": {key: summarize(np.concatenate([e[key] for e in per_view]))
                              for key in per_view[0]}}
            stored[name] = np.concatenate([e["MBRU"] for e in per_view])
        block["paired"] = {}
        for better, worse in (("ZS_EFFECT", "ZS_BASE"), ("ZS_EFFECT", "ANCHOR_EFFECT_MAGNITUDE"),
                              ("ZS_BASE", "ANCHOR_EFFECT_MAGNITUDE"),
                              ("EFFECT_BRANCH_ONLY", "ANCHOR_EFFECT_MAGNITUDE"),
                              ("ZS_BASE", "RANDOM"),
                              ("ZS_BASE_PLUS_MAGNITUDE", "ZS_BASE"),
                              ("ZS_BASE_PLUS_RANDOM", "ZS_BASE")):
            difference = stored[better] - stored[worse]
            block["paired"][f"{better}_minus_{worse}"] = {
                **summarize(difference), "bootstrap": cluster_bootstrap(difference),
                "fraction_query_improved": float((difference > 0).mean())}
        record["strata"][stratum] = block

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record["runtime"] = {"timestamp": datetime.now(timezone.utc).isoformat(),
                         "python": platform.python_version(), "numpy": np.__version__}
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 6: zero-shot context transfer")
    for name in ("config", "benchmark", "lock", "distillation", "anchor_genes", "output"):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, required=True)
    parser.add_argument("--device", default="cuda")
    record = run(parser.parse_args())
    g, p = record["gene_bridge"], record["pool"]
    print(f"G_common {g['size']} ({g['fraction_of_anchor_axis']:.4f} of the anchor axis, "
          f"{g['fraction_of_external_axis']:.4f} of the external axis)")
    print(f"candidate features reproduce the anchor: {record['candidate_feature_verification']}")
    print(f"pool {p['usable_pool']} candidates, {p['eligible_queries']} eligible queries "
          f"({p['queries_seen_in_anchor_distillation']} seen, {p['queries_unseen']} unseen)")
    for stratum, block in record["strata"].items():
        print(f"\n[{stratum}]  {block['queries']} queries")
        print(f"  {'row':28s} {'MBRU':>8s} {'Mean@10':>9s} {'Mean@20':>9s} {'HVC@10':>7s} {'HVC@20':>7s} {'HVC@50':>7s}")
        for name, entry in block["rows"].items():
            d = entry["direction"]
            print(f"  {name:28s} {d['MBRU']['median']:+8.4f} {d['Mean@10']['median']:+9.4f} "
                  f"{d['Mean@20']['median']:+9.4f} {d['HighValueCount@10']['median']:7.2f} "
                  f"{d['HighValueCount@20']['median']:7.2f} {d['HighValueCount@50']['median']:7.2f}")
        for key, item in block["paired"].items():
            low, high = item["bootstrap"]["median"]["ci95"]
            print(f"    {key:46s} median {item['median']:+.4f} [{low:+.4f},{high:+.4f}] "
                  f"improved {item['fraction_query_improved']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
