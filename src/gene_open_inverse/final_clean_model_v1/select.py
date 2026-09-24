#!/usr/bin/env python3
"""Phase E: choose one checkpoint per arm, on G_select, changing nothing else.

Architecture, fusion, predictor, feature set, sampler, loss and query gate are all
fixed before this runs.  The only decision taken here is which epoch of the single
clean training curve each arm uses, and each arm is allowed its own because each is
scored by its own rule.  ``G_check`` is not read.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ..candidate_effect_distillation_v1 import contract as ced
from ..model.decision_aligned_sampler_v1.evaluate import _direction_metrics
from ..model.decision_alignment_v1.assets import AuditConfig
from ..model.decision_alignment_v1.contract import BUDGETS
from ..model.global_objective_knowledge_attribution_v1.assets import GlobalAttributionAssets
from ..model.open_vocab_dual_encoder_v1.data import FrozenTransitionAssets, sha256
from ..model.open_vocab_dual_encoder_v1.metrics import stable_order
from ..model.query_conditional_capacity_v1.run import _guardrail
from ..utility_gt_v2.validate import summarize
from . import contract as fc
from .model import CleanAttributionModel
from .scoring import FrozenEffectPredictor, base_scores, effect_scores, fuse, strict_legal_dims, strict_legal_role


def prefix_all_missing(matrices, rows, ids, missing) -> dict:
    share = {str(budget): [] for budget in BUDGETS}
    for view in (0, 1):
        for query in rows.tolist():
            order = stable_order(matrices[view][query], ids, excluded_index=query)
            order = order[order != query]
            for budget in BUDGETS:
                share[str(budget)].append(float(missing[order[:budget]].mean()))
    return {key: float(np.mean(value)) for key, value in share.items()}


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    config = AuditConfig.load(Path(args.config))
    assets = GlobalAttributionAssets(modalities=("STRING", "MAPKG", "TEXT"), **config.asset_arguments())
    dims = strict_legal_dims(assets)
    fit_role = strict_legal_role(assets.role_arrays("G_fit"))
    select_role = strict_legal_role(assets.role_arrays("G_select"))
    ids = np.asarray([str(v) for v in select_role.base.ids])
    rows = np.flatnonzero(np.load(Path(args.eligibility) / "G_select_eligible.npy"))
    utility = [np.asarray(FrozenTransitionAssets.cross_view_utility(select_role.base, view),
                          dtype=np.float64) for view in (0, 1)]

    manifest = json.loads((Path(args.checkpoints) / "train_manifest.json").read_text())
    penalty = float(json.loads(
        (Path(args.distillation) / "candidate_effect_distillation.json").read_text()
    )["predictor"]["cross_validation"]["selected_penalty"])
    predictor = FrozenEffectPredictor(fit_role, penalty)
    saved = np.load(Path(args.distillation) / "predicted_effect_G_select.npy")
    reproduction_error = predictor.verify_against(saved, select_role)
    if reproduction_error > 1e-3:
        raise RuntimeError(
            f"the rebuilt effect predictor does not reproduce the frozen artifact "
            f"(max |difference| {reproduction_error:.3g})")
    effect = predictor.effect(select_role)
    effect_matrices = effect_scores(select_role, effect)
    missing = np.linalg.norm(effect, axis=1) <= 0.0

    asset = Path(args.asset)
    quarter_all = np.load(asset / "quarter_responses.npy", mmap_mode="r")
    identity_axis = np.asarray([str(v) for v in np.load(asset / "identity_axis.npy", allow_pickle=True)])
    four_way = np.load(asset / "four_way_eligible.npy")
    axis_index = {name: index for index, name in enumerate(identity_axis.tolist())}
    position = np.asarray([axis_index[name] for name in ids.tolist()], dtype=np.int64)
    candidate_rows = np.flatnonzero(four_way[position])
    guard_rows = np.flatnonzero(np.load(Path(args.eligibility) / "G_select_eligible.npy")[candidate_rows])
    quarters = [np.asarray(quarter_all[index][position][candidate_rows], dtype=np.float64)
                for index in range(4)]

    record: dict = {
        "schema": "VCDESIGN_FINAL_CLEAN_MODEL_V1_SELECTION",
        "read_G_check": False,
        "what_may_change_here": "the checkpoint epoch of each arm, and nothing else",
        "effect_predictor_reproduction_max_abs_error": reproduction_error,
        "ridge_penalty": penalty,
        "all_missing_candidates": int(missing.sum()),
        "epochs": {},
    }
    for epoch in sorted(manifest["checkpoints"], key=int):
        path = Path(manifest["checkpoints"][epoch]["path"])
        model = CleanAttributionModel(dims).to(device)
        payload = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"], strict=True)
        base = base_scores(model, select_role, device)
        fused = fuse(base, effect_matrices)
        entry = {"checkpoint": str(path), "sha256": sha256(path), "arms": {}}
        for arm, matrices in (("CLEAN_BASE", base), ("CLEAN_EFFECT", fused)):
            per_view = [_direction_metrics(matrices[view], utility[view], ids, rows) for view in (0, 1)]
            direction = {key: summarize(np.concatenate([e[key] for e in per_view])) for key in per_view[0]}
            entry["arms"][arm] = {
                "direction": direction,
                "guardrail": {k: v for k, v in _guardrail(matrices, quarters, candidate_rows, guard_rows,
                                                          ids[candidate_rows]).items()
                              if not k.startswith("_")},
                "all_missing_fraction_of_prefix": prefix_all_missing(matrices, rows, ids, missing),
            }
        record["epochs"][epoch] = entry
        print(f"epoch {epoch}: CLEAN_BASE MBRU {entry['arms']['CLEAN_BASE']['direction']['MBRU']['median']:+.4f}  "
              f"CLEAN_EFFECT MBRU {entry['arms']['CLEAN_EFFECT']['direction']['MBRU']['median']:+.4f}", flush=True)
        del model

    record["selected"] = {}
    for arm in fc.ARMS:
        best = max(record["epochs"], key=lambda e: record["epochs"][e]["arms"][arm]["direction"]["MBRU"]["median"])
        record["selected"][arm] = {
            "epoch": int(best), "checkpoint": record["epochs"][best]["checkpoint"],
            "sha256": record["epochs"][best]["sha256"],
            "G_select_MBRU_median": record["epochs"][best]["arms"][arm]["direction"]["MBRU"]["median"]}

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record["runtime"] = {"timestamp": datetime.now(timezone.utc).isoformat(),
                         "python": platform.python_version(), "numpy": np.__version__}
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase E: final checkpoint selection on G_select")
    for name in ("config", "eligibility", "asset", "checkpoints", "distillation", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--device", default="cuda")
    record = run(parser.parse_args())
    print(f"\neffect predictor reproduced to {record['effect_predictor_reproduction_max_abs_error']:.3g}")
    for arm, entry in record["selected"].items():
        print(f"FINAL_{arm}_CHECKPOINT = epoch {entry['epoch']}  MBRU {entry['G_select_MBRU_median']:+.4f}  "
              f"sha256 {entry['sha256'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
