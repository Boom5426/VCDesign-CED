#!/usr/bin/env python3
"""Phase G: the one locked final evaluation.

This module is the only thing in the repository that reaches ``G_check``, and it
does so through ``scoring.locked_check_arrays``, which refuses to run without an
explicit acknowledgement string.  It refuses to overwrite its own output, so it
cannot be run twice into the same place by accident.

The split is called a **locked final evaluation split**, never an untouched blind
test: earlier phases of this programme read ``G_check``.  What is asserted, and what
matters, is that this phase's method, effect predictor, fusion and checkpoints were
selected without it.

The query gate is the frozen ``QUERY_ELIGIBILITY_V1`` rule applied unchanged to this
role, at production A/B depth, exactly as it was applied to the other two roles.
Nothing here is tuned on the result.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ..model.decision_aligned_sampler_v1.evaluate import HIGH_VALUE_QUANTILE, _direction_metrics
from ..model.decision_alignment_v1.assets import AuditConfig
from ..model.decision_alignment_v1.contract import BUDGETS
from ..model.global_objective_knowledge_attribution_v1.assets import GlobalAttributionAssets
from ..model.global_objective_knowledge_attribution_v1.model import GlobalObjectiveKnowledgeAttributionModel
from ..model.open_vocab_dual_encoder_v1.data import FrozenTransitionAssets
from ..model.open_vocab_dual_encoder_v1.metrics import stable_order
from ..model.query_conditional_capacity_v1.run import _guardrail
from ..utility_gt_v2 import contract as gt
from ..utility_gt_v2.validate import cluster_bootstrap, summarize
from . import contract as fc
from .model import CleanAttributionModel
from .scoring import (
    ACKNOWLEDGEMENT, FrozenEffectPredictor, base_scores, effect_scores, fuse, locked_check_arrays,
    strict_legal_dims, strict_legal_role, unit_rows,
)


LEGALITY = {
    "RANDOM": "no information",
    "STRICT_LEGAL_CANDIDATE_PRIOR": "strict legal candidate knowledge, no goal",
    "EFFECT_BRANCH_ONLY": "strict legal, deployment legal",
    "LEGACY_M_SET": "legacy contract, carries TEXT, context only, not the paired comparator",
    "CLEAN_BASE": "final, deployment legal, strict legal features",
    "CLEAN_EFFECT": "final, deployment legal, strict legal features",
    "MEASURED_RESPONSE_ORACLE": "DIAGNOSTIC_ORACLE_NOT_DEPLOYABLE, privileged upper bound",
}


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    if output.exists():
        raise RuntimeError("the locked evaluation refuses to overwrite an existing record")
    device = torch.device(args.device)
    config = AuditConfig.load(Path(args.config))
    assets = GlobalAttributionAssets(modalities=("STRING", "MAPKG", "TEXT"), **config.asset_arguments())
    dims = strict_legal_dims(assets)
    fit_role = strict_legal_role(assets.role_arrays("G_fit"))

    legacy_check = locked_check_arrays(assets, ACKNOWLEDGEMENT)
    check_role = strict_legal_role(legacy_check)
    ids = np.asarray([str(v) for v in check_role.base.ids])

    # The frozen query gate, applied unchanged to this role at production A/B depth.
    transitions = np.asarray(check_role.base.transitions, dtype=np.float64)
    gate = gt.eligibility(gt.cross_view_gram(transitions[0], transitions[1]), gt.PRIMARY_FDR)
    rows = np.flatnonzero(gate.eligible)
    utility = [np.asarray(FrozenTransitionAssets.cross_view_utility(check_role.base, view),
                          dtype=np.float64) for view in (0, 1)]

    lock = json.loads(Path(args.lock).read_text())
    penalty = float(lock["effect_predictor"]["ridge_penalty"])
    predictor = FrozenEffectPredictor(fit_role, penalty)
    saved = np.load(Path(args.distillation) / "predicted_effect_G_select.npy")
    select_role = strict_legal_role(assets.role_arrays("G_select"))
    reproduction_error = predictor.verify_against(saved, select_role)
    if reproduction_error > 1e-3:
        raise RuntimeError(f"the effect predictor is not the locked one ({reproduction_error:.3g})")
    effect = predictor.effect(check_role)
    effect_matrices = effect_scores(check_role, effect)
    magnitude = np.linalg.norm(effect, axis=1)
    missing = magnitude <= 0.0

    asset = Path(args.asset)
    quarter_all = np.load(asset / "quarter_responses.npy", mmap_mode="r")
    identity_axis = np.asarray([str(v) for v in np.load(asset / "identity_axis.npy", allow_pickle=True)])
    four_way = np.load(asset / "four_way_eligible.npy")
    axis_index = {name: index for index, name in enumerate(identity_axis.tolist())}
    position = np.asarray([axis_index[name] for name in ids.tolist()], dtype=np.int64)
    candidate_rows = np.flatnonzero(four_way[position])
    guard_rows = np.flatnonzero(gate.eligible[candidate_rows])
    quarters = [np.asarray(quarter_all[index][position][candidate_rows], dtype=np.float64)
                for index in range(4)]

    scores: dict[str, list[np.ndarray]] = {}
    generator = np.random.default_rng(20260918)
    scores["RANDOM"] = [generator.standard_normal((ids.size, ids.size)) for _ in (0, 1)]
    scores["STRICT_LEGAL_CANDIDATE_PRIOR"] = [
        np.broadcast_to(magnitude, (ids.size, ids.size)).copy() for _ in (0, 1)]
    scores["EFFECT_BRANCH_ONLY"] = effect_matrices

    for arm in fc.ARMS:
        entry = lock["selected_checkpoints"][arm]
        model = CleanAttributionModel(dims).to(device)
        payload = torch.load(entry["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"], strict=True)
        base = base_scores(model, check_role, device)
        scores[arm] = base if arm == "CLEAN_BASE" else fuse(base, effect_matrices)
        del model

    legacy_model = GlobalObjectiveKnowledgeAttributionModel(assets.capability_dims).to(device)
    legacy_model.load_state_dict(
        torch.load(config.checkpoints["M_SET"]["path"], map_location=device,
                   weights_only=False)["model_state"], strict=True)
    scores["LEGACY_M_SET"] = base_scores(legacy_model, legacy_check, device)
    del legacy_model

    oracle = []
    for view in (0, 1):
        query = unit_rows(np.asarray(check_role.base.transitions[view], dtype=np.float64))
        oracle.append(query @ query.T)
    scores["MEASURED_RESPONSE_ORACLE"] = oracle

    record: dict = {
        "schema": "VCDESIGN_FINAL_CLEAN_MODEL_V1_LOCKED_EVALUATION",
        "evaluation_split": fc.EVALUATION_SPLIT_NAME,
        "must_not_be_called": fc.EVALUATION_SPLIT_MUST_NOT_BE_CALLED,
        "method_of_record": fc.FINAL_METHOD, "fusion": fc.FINAL_FUSION, "beta": fc.FINAL_BETA,
        "fusion_selected_before_G_check": fc.FINAL_FUSION_SELECTED_BEFORE_G_CHECK,
        "lock_sha256": lock.get("lock_sha256"), "lock_path": str(Path(args.lock).resolve()),
        "effect_predictor_reproduction_max_abs_error": reproduction_error,
        "query_gate": {
            "rule": "QUERY_ELIGIBILITY_V1 applied unchanged at production A/B depth",
            "fdr": gt.PRIMARY_FDR, "nominal": int(ids.size), "eligible": int(rows.size),
            "eligibility_fraction": float(rows.size / ids.size),
            "four_way_candidates": int(candidate_rows.size),
            "eligible_and_four_way": int(guard_rows.size)},
        "all_missing_candidates": int(missing.sum()),
        "high_value_quantile": HIGH_VALUE_QUANTILE, "budgets": list(BUDGETS),
        "rows": {},
    }

    stored = {}
    for name in LEGALITY:
        matrices = scores[name]
        per_view = [_direction_metrics(matrices[view], utility[view], ids, rows) for view in (0, 1)]
        direction = {key: summarize(np.concatenate([e[key] for e in per_view])) for key in per_view[0]}
        stored[name] = np.concatenate([e["MBRU"] for e in per_view])
        share = {str(budget): [] for budget in BUDGETS}
        for view in (0, 1):
            for query in rows.tolist():
                order = stable_order(matrices[view][query], ids, excluded_index=query)
                order = order[order != query]
                for budget in BUDGETS:
                    share[str(budget)].append(float(missing[order[:budget]].mean()))
        record["rows"][name] = {
            "legality": LEGALITY[name], "direction": direction,
            "guardrail": {k: v for k, v in _guardrail(matrices, quarters, candidate_rows, guard_rows,
                                                      ids[candidate_rows]).items()
                          if not k.startswith("_")},
            "all_missing_fraction_of_prefix": {k: float(np.mean(v)) for k, v in share.items()},
        }

    record["paired"] = {}
    for better, worse in (("CLEAN_EFFECT", "CLEAN_BASE"), ("CLEAN_EFFECT", "STRICT_LEGAL_CANDIDATE_PRIOR"),
                          ("CLEAN_BASE", "STRICT_LEGAL_CANDIDATE_PRIOR"), ("CLEAN_EFFECT", "LEGACY_M_SET"),
                          ("CLEAN_BASE", "LEGACY_M_SET"), ("EFFECT_BRANCH_ONLY", "STRICT_LEGAL_CANDIDATE_PRIOR")):
        difference = stored[better] - stored[worse]
        record["paired"][f"{better}_minus_{worse}"] = {
            **summarize(difference), "bootstrap": cluster_bootstrap(difference),
            "fraction_query_improved": float((difference > 0).mean())}
    record["primary_comparison"] = "CLEAN_EFFECT_minus_CLEAN_BASE"

    output.parent.mkdir(parents=True, exist_ok=True)
    record["runtime"] = {"timestamp": datetime.now(timezone.utc).isoformat(),
                         "python": platform.python_version(), "numpy": np.__version__,
                         "torch": torch.__version__}
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase G: the one locked final evaluation")
    for name in ("config", "asset", "lock", "distillation", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--device", default="cuda")
    record = run(parser.parse_args())
    gate = record["query_gate"]
    print(f"{record['evaluation_split']}: {gate['eligible']} eligible of {gate['nominal']} nominal "
          f"({gate['eligibility_fraction']:.4f}); four-way candidates {gate['four_way_candidates']}")
    print(f"effect predictor reproduced to {record['effect_predictor_reproduction_max_abs_error']:.3g}\n")
    print(f"{'row':32s} {'MBRU':>8s} {'Mean@10':>9s} {'Mean@20':>9s} {'HVC@10':>7s} {'HVC@20':>7s} "
          f"{'HVC@50':>7s} {'J RU@10':>9s} {'J>0@10':>7s} {'miss@10':>8s}")
    for name, entry in record["rows"].items():
        d, g = entry["direction"], entry["guardrail"]
        print(f"{name:32s} {d['MBRU']['median']:+8.4f} {d['Mean@10']['median']:+9.4f} "
              f"{d['Mean@20']['median']:+9.4f} {d['HighValueCount@10']['median']:7.2f} "
              f"{d['HighValueCount@20']['median']:7.2f} {d['HighValueCount@50']['median']:7.2f} "
              f"{g['both_views']['RU@10']['median']:+9.5f} {g['view0']['fraction_J_positive@10']:7.4f} "
              f"{entry['all_missing_fraction_of_prefix']['10']:8.4f}")
    print("\npaired identity-cluster bootstrap, 10000 replicates")
    for key, item in record["paired"].items():
        low, high = item["bootstrap"]["median"]["ci95"]
        mean_low, mean_high = item["bootstrap"]["mean"]["ci95"]
        mark = "  <== PRIMARY" if key == record["primary_comparison"] else ""
        print(f"  {key:52s} median {item['median']:+.4f} [{low:+.4f},{high:+.4f}] "
              f"mean {item['mean']:+.4f} [{mean_low:+.4f},{mean_high:+.4f}] "
              f"improved {item['fraction_query_improved']:.3f}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
