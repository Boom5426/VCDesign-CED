#!/usr/bin/env python3
"""Extract the real row-level K562 values needed by Figure 2.

The frozen locked-evaluation JSON stores only summaries.  This script replays the
locked scoring path without training or changing any model, validates every derived
summary against that record, and writes a compact machine-readable source-data file.

Run it on the compute host that holds the frozen assets and checkpoints::

    python3 - --config CONFIG.json --lock VCDesign_FINAL_LOCK.json \
        --distillation CED_PROBE_DIR --reference-record locked_evaluation.json \
        --output - < figures/scripts/extract_fig2_rows.py

Use ``--output -`` to stream the JSON without creating a file on the compute host.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

from gene_open_inverse.model.decision_aligned_sampler_v1.evaluate import _direction_metrics
from gene_open_inverse.model.decision_alignment_v1.assets import AuditConfig
from gene_open_inverse.model.global_objective_knowledge_attribution_v1.assets import (
    GlobalAttributionAssets,
)
from gene_open_inverse.model.global_objective_knowledge_attribution_v1.model import (
    GlobalObjectiveKnowledgeAttributionModel,
)
from gene_open_inverse.model.open_vocab_dual_encoder_v1.data import FrozenTransitionAssets
from gene_open_inverse.utility_gt_v2 import contract as gt
from gene_open_inverse.final_clean_model_v1 import contract as fc
from gene_open_inverse.final_clean_model_v1.model import CleanAttributionModel
from gene_open_inverse.final_clean_model_v1.scoring import (
    ACKNOWLEDGEMENT,
    FrozenEffectPredictor,
    base_scores,
    effect_scores,
    fuse,
    locked_check_arrays,
    strict_legal_dims,
    strict_legal_role,
)


ARMS = {
    "Goal-blind prior": "STRICT_LEGAL_CANDIDATE_PRIOR",
    "Forward-then-match": "EFFECT_BRANCH_ONLY",
    "Base scorer": "CLEAN_BASE",
    "Text-augmented ranker": "LEGACY_M_SET",
    "VCDesign-CED": "CLEAN_EFFECT",
}
BUDGETS = (10, 20, 50)
ATOL = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_close(label: str, observed: float, expected: float) -> None:
    if not np.isclose(observed, expected, rtol=0.0, atol=ATOL):
        raise RuntimeError(f"{label}: reproduced {observed:.17g}, expected {expected:.17g}")


def score_and_measure(args: argparse.Namespace) -> dict:
    config_path = Path(args.config)
    lock_path = Path(args.lock)
    distillation = Path(args.distillation)
    reference_path = Path(args.reference_record)
    reference = json.loads(reference_path.read_text())
    lock = json.loads(lock_path.read_text())

    device = torch.device(args.device)
    config = AuditConfig.load(config_path)
    assets = GlobalAttributionAssets(
        modalities=("STRING", "MAPKG", "TEXT"), **config.asset_arguments()
    )
    dims = strict_legal_dims(assets)
    fit_role = strict_legal_role(assets.role_arrays("G_fit"))
    legacy_check = locked_check_arrays(assets, ACKNOWLEDGEMENT)
    check_role = strict_legal_role(legacy_check)
    ids = np.asarray([str(value) for value in check_role.base.ids])

    transitions = np.asarray(check_role.base.transitions, dtype=np.float64)
    gate = gt.eligibility(gt.cross_view_gram(transitions[0], transitions[1]), gt.PRIMARY_FDR)
    rows = np.flatnonzero(gate.eligible)
    utility = [
        np.asarray(FrozenTransitionAssets.cross_view_utility(check_role.base, view), dtype=np.float64)
        for view in (0, 1)
    ]

    predictor = FrozenEffectPredictor(fit_role, float(lock["effect_predictor"]["ridge_penalty"]))
    saved = np.load(distillation / "predicted_effect_G_select.npy")
    select_role = strict_legal_role(assets.role_arrays("G_select"))
    reproduction_error = predictor.verify_against(saved, select_role)
    if reproduction_error > 1e-3:
        raise RuntimeError(f"locked effect predictor mismatch: {reproduction_error:.3g}")
    effect = predictor.effect(check_role)
    effect_matrices = effect_scores(check_role, effect)
    magnitude = np.linalg.norm(effect, axis=1)

    scores: dict[str, list[np.ndarray]] = {
        "STRICT_LEGAL_CANDIDATE_PRIOR": [
            np.broadcast_to(magnitude, (ids.size, ids.size)).copy() for _ in (0, 1)
        ],
        "EFFECT_BRANCH_ONLY": effect_matrices,
    }

    checkpoint = lock["selected_checkpoints"]["CLEAN_BASE"]["checkpoint"]
    model = CleanAttributionModel(dims).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    base = base_scores(model, check_role, device)
    scores["CLEAN_BASE"] = base
    scores["CLEAN_EFFECT"] = fuse(base, effect_matrices)
    del model

    legacy_model = GlobalObjectiveKnowledgeAttributionModel(assets.capability_dims).to(device)
    legacy_payload = torch.load(
        config.checkpoints["M_SET"]["path"], map_location=device, weights_only=False
    )
    legacy_model.load_state_dict(legacy_payload["model_state"], strict=True)
    scores["LEGACY_M_SET"] = base_scores(legacy_model, legacy_check, device)
    del legacy_model

    metrics: dict[str, dict[str, np.ndarray]] = {}
    for name, arm in ARMS.items():
        per_view = [
            _direction_metrics(scores[arm][view], utility[view], ids, rows) for view in (0, 1)
        ]
        metrics[name] = {
            metric: np.concatenate([entry[metric] for entry in per_view])
            for metric in per_view[0]
        }

    row_ids = [
        {"query": str(ids[index]), "view": view}
        for view in (0, 1)
        for index in rows.tolist()
    ]
    if len(row_ids) != int(reference["paired"]["CLEAN_EFFECT_minus_CLEAN_BASE"]["bootstrap"]["clusters"]):
        raise RuntimeError("row count does not match the frozen paired-analysis population")

    for name, arm in ARMS.items():
        check_close(
            f"{name} median MBRU",
            float(np.median(metrics[name]["MBRU"])),
            float(reference["rows"][arm]["direction"]["MBRU"]["median"]),
        )
        if name in ("Base scorer", "Text-augmented ranker", "VCDesign-CED"):
            for budget in BUDGETS:
                metric = f"HighValueCount@{budget}"
                check_close(
                    f"{name} median {metric}",
                    float(np.median(metrics[name][metric])),
                    float(reference["rows"][arm]["direction"][metric]["median"]),
                )

    paired = {}
    for comparator, record_key in (
        ("Base scorer", "CLEAN_EFFECT_minus_CLEAN_BASE"),
        ("Text-augmented ranker", "CLEAN_EFFECT_minus_LEGACY_M_SET"),
    ):
        values = metrics["VCDesign-CED"]["MBRU"] - metrics[comparator]["MBRU"]
        frozen = reference["paired"][record_key]
        check_close(f"VCDesign-CED - {comparator} median", float(np.median(values)), frozen["median"])
        check_close(
            f"VCDesign-CED - {comparator} win rate",
            float(np.mean(values > 0)),
            frozen["fraction_query_improved"],
        )
        paired[comparator] = [float(value) for value in values]

    return {
        "schema": "VCDESIGN_FIG2_K562_ROWS_V1",
        "generated_by": "figures/scripts/extract_fig2_rows.py",
        "provenance": {
            "reference_record": {
                "kind": "locked_evaluation.json",
                "sha256": sha256(reference_path),
            },
            "lock_sha256": lock.get("lock_sha256"),
            "config_sha256": sha256(config_path),
            "effect_predictor_reproduction_max_abs_error": float(reproduction_error),
            "checkpoint_sha256": lock["selected_checkpoints"]["CLEAN_BASE"]["sha256"],
            "legacy_checkpoint_sha256": config.provenance()["checkpoints"]["M_SET"]["sha256"],
        },
        "population": {
            "queries": int(rows.size),
            "query_view_rows": len(row_ids),
            "candidates_per_row": int(ids.size - 1),
            "row_order": "view 0 eligible queries, then view 1 eligible queries",
        },
        "rows": row_ids,
        "mbu": {
            name: [float(value) for value in metrics[name]["MBRU"]] for name in ARMS
        },
        "paired_delta_mbu": paired,
        "high_value_hits": {
            name: {
                str(budget): [
                    float(value) for value in metrics[name][f"HighValueCount@{budget}"]
                ]
                for budget in BUDGETS
            }
            for name in ("Base scorer", "Text-augmented ranker", "VCDesign-CED")
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--distillation", required=True)
    parser.add_argument("--reference-record", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True, help="explicit output path, or '-' for stdout")
    args = parser.parse_args()
    data = score_and_measure(args)
    payload = json.dumps(data, indent=2) + "\n"
    if args.output == "-":
        sys.stdout.write(payload)
    else:
        output = Path(args.output)
        if output.exists():
            raise RuntimeError(f"refusing to overwrite existing output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
