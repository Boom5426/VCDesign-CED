#!/usr/bin/env python3
"""Assemble the canonical row source for Figure 4 without refitting models.

The extractor joins three frozen, already-computed sources: the former Figure 4
diagnostic artifact (factorial probe, atlas-context companion, and capacity
interventions), the same-identity measurement control, and the prediction-design
closure block. Submission-facing output retains source roles and hashes, never
machine-specific paths.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


HELD = ("RPE1", "HepG2", "Jurkat")
SCHEMA = "VCDESIGN_FIG4_ROWS_V2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    out = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        average = (start + 1 + stop) / 2.0
        for index in order[start:stop]:
            out[index] = average
        start = stop
    return out


def pearson(x: list[float], y: list[float]) -> float:
    mx, my = sum(x) / len(x), sum(y) / len(y)
    dx, dy = [v - mx for v in x], [v - my for v in y]
    return sum(a * b for a, b in zip(dx, dy)) / math.sqrt(
        sum(a * a for a in dx) * sum(b * b for b in dy)
    )


def cleaned_sources(legacy: dict) -> dict:
    out = {}
    for name, source in legacy["sources"].items():
        if name == "evidence_fits":
            out[name] = {
                "kind": "evidence-v1 fit records",
                "record_sha256": source["record_sha256"],
            }
        else:
            out[name] = {"kind": f"{name} canonical record", "sha256": source["sha256"]}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--diagnosis-v1", required=True, type=Path)
    parser.add_argument("--same-identity", required=True, type=Path)
    parser.add_argument("--prediction-design", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise RuntimeError(f"refusing to overwrite without --overwrite: {args.output}")

    legacy = json.loads(args.diagnosis_v1.read_text())
    prediction = json.loads(args.prediction_design.read_text())
    if legacy.get("schema") != "VCDESIGN_FIG4_DATA_V1":
        raise RuntimeError("the diagnostic source is not Figure 4 V1")

    with args.same_identity.open(newline="") as handle:
        matched_rows = list(csv.DictReader(handle))
    matched = {row["held"]: row for row in matched_rows}
    if not all(name in matched for name in (*HELD, "K562")):
        raise RuntimeError("same-identity control is missing a held context")
    panel_b = {
        "statistic": (
            "median paired MBRU gain over the frozen K562 base scorer for raw-target CED; "
            "same perturbation identities in both arms"
        ),
        "conditions": {
            "M_spread": "M rows spread across the three training contexts",
            "three_M": "the same identities measured in all three contexts (3M rows)",
        },
        "contexts": {
            held: {
                "M_spread": float(matched[held]["spread_3ctx"]),
                "three_M": float(matched[held]["same_ids_3x"]),
                "increment": float(matched[held]["same_ids_3x"])
                - float(matched[held]["spread_3ctx"]),
            }
            for held in HELD
        },
        "posthoc_best_single_companion": {
            held: {
                "value": float(matched[held]["best_single"]),
                "context": matched[held]["best_single_ctx"],
            }
            for held in HELD
        },
        "main_figure_scope": (
            "the retrospectively selected best-single arm is omitted from the main panel and "
            "retained in the supplementary same-identity figure"
        ),
        "fit_scope": "3M is one fit; M-spread is the mean of five row-sampling-seed medians",
    }

    points = [
        item for item in prediction["points"]
        if item["regime"] == "MASKED" and item["family"] == "ridge"
    ]
    within = [
        item for item in prediction["within"]
        if item["regime"] == "MASKED" and item["kind"] == "same atlas, one input changed"
    ]
    across = prediction["across|MASKED"]
    agreement = prediction["within_same_sign|same atlas, one input changed|MASKED"]
    observed_rho = pearson(
        rank([item["prediction"] for item in points]),
        rank([item["design_effect_only_BU@20"] for item in points]),
    )
    if len(points) != 57 or len(within) != 15:
        raise RuntimeError(f"prediction-design closure changed: {len(points)} / {len(within)}")
    if not math.isclose(observed_rho, across["spearman"], abs_tol=1e-12, rel_tol=0.0):
        raise RuntimeError("stored and recomputed MASKED Spearman correlations disagree")
    observed_agree = sum(bool(item["same_sign"]) for item in within)
    if observed_agree != agreement["agree"] or agreement["pairs"] != len(within):
        raise RuntimeError("stored and recomputed sign-agreement counts disagree")

    panel_a = legacy["panel_a"]
    cells = panel_a["cells"]
    if not math.isclose(
        cells["Q_CURRENT__C_ORACLE"] - cells["Q_CURRENT__C_LEGAL"],
        panel_a["candidate_swap"]["Q_CURRENT"], abs_tol=1e-12, rel_tol=0.0,
    ):
        raise RuntimeError("factorial probe edge does not reproduce its cells")

    variants = legacy["panel_c"]["variants"]
    common_reference = all(
        "RIDGE_UNIT" in item["record_key"] or "minus_A1" in item["record_key"]
        for item in variants
    )
    assertions = {
        "panel_a_one_factorial_artifact": True,
        "panel_a_candidate_edges_recomputed_from_cells": True,
        "panel_b_same_identity_universe": True,
        "panel_b_posthoc_reference_excluded_from_main": True,
        "panel_c_common_unit_direction_ridge_reference": common_reference,
        "panel_d_masked_grid_count_57": len(points) == 57,
        "panel_d_masked_controlled_count_15": len(within) == 15,
        "panel_d_spearman_recomputed": True,
        "panel_d_sign_agreement_recomputed": observed_agree == 3,
        "no_cross_panel_effect_size_ratio": True,
    }
    if not all(assertions.values()):
        raise RuntimeError(f"a Figure 4 assertion failed: {assertions}")

    output = {
        "schema": SCHEMA,
        "generated_by": "figures/scripts/extract_fig4_rows.py",
        "sources": {
            **cleaned_sources(legacy),
            "same_identity_control": {
                "kind": "same-identity atlas intervention CSV",
                "sha256": sha256(args.same_identity),
            },
            "prediction_design_closure": {
                "kind": "prediction-versus-design closure block",
                "sha256": sha256(args.prediction_design),
            },
        },
        "panel_a": panel_a,
        "panel_b": panel_b,
        "panel_c": {
            "statistic": legacy["panel_c"]["statistic"],
            "ci_source": legacy["panel_c"]["ci_source"],
            "variants": variants,
        },
        "panel_d": {
            "prediction_metric": (
                "held-context candidate-specific cosine after removing the A1 common response axis"
            ),
            "design_metric": "effect-only BU@20",
            "regime": "MASKED",
            "grid_points": points,
            "controlled_changes": within,
            "spearman": float(across["spearman"]),
            "p_value": float(across["p_value"]),
            "n_grid": int(across["n"]),
            "same_sign": {"agree": int(agreement["agree"]), "pairs": int(agreement["pairs"])},
        },
        "supplementary": {
            "atlas_context_curve": legacy["panel_b"],
            "context_specific_oracle_headroom": legacy["panel_c"]["headroom"],
        },
        "mandatory_assertions": assertions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
