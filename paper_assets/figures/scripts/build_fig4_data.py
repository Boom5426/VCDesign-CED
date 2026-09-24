#!/usr/bin/env python3
"""Build Figure 4 data, row-oriented source data, provenance, and its SI companion."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


SCHEMA_IN = "VCDESIGN_FIG4_ROWS_V2"
SCHEMA_OUT = "VCDESIGN_FIG4_DATA_V2"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: dict, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise RuntimeError(f"refusing to overwrite without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def validate(rows: dict) -> None:
    if rows.get("schema") != SCHEMA_IN:
        raise RuntimeError(f"unexpected Figure 4 row schema: {rows.get('schema')}")
    if not all(rows["mandatory_assertions"].values()):
        raise RuntimeError(f"mandatory assertions failed: {rows['mandatory_assertions']}")
    if rows["panel_d"]["n_grid"] != 57:
        raise RuntimeError("prediction-design grid count changed")
    if rows["panel_d"]["same_sign"] != {"agree": 3, "pairs": 15}:
        raise RuntimeError("prediction-design sign agreement changed")


def main_data(rows: dict, rows_path: Path) -> dict:
    return {
        "schema": SCHEMA_OUT,
        "generated_by": "figures/scripts/build_fig4_data.py",
        "source": {"kind": "canonical Figure 4 row extraction", "sha256": sha256(rows_path)},
        "contract": {
            "core_conclusion": (
                "controlled interventions localize the information, measurement, capacity, and "
                "model-selection limits of transferable intervention design"
            ),
            "archetype": "asymmetric four-panel insight figure",
            "final_size_inches": [5.5, 4.05],
            "panel_map": {
                "a": "candidate-information factorial diagnostic",
                "b": "same-identity measurement intervention",
                "c": "capacity and candidate-information interventions",
                "d": "global prediction-design association with within-atlas changes",
            },
            "cross_panel_warning": (
                "panels use different populations, references, and metrics; effect sizes are not "
                "ranked or ratioed across panels"
            ),
        },
        "panel_a": rows["panel_a"],
        "panel_b": rows["panel_b"],
        "panel_c": rows["panel_c"],
        "panel_d": rows["panel_d"],
        "mandatory_assertions": rows["mandatory_assertions"],
    }


def write_source_csv(path: Path, rows: dict, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise RuntimeError(f"refusing to overwrite without --overwrite: {path}")
    fields = [
        "panel", "record_type", "context", "condition_or_variant", "x_value", "y_value",
        "value", "ci95_low", "ci95_high", "metric", "reference", "owning_section", "note",
    ]
    output = []
    a = rows["panel_a"]
    for cell, value in a["cells"].items():
        query, candidate = cell.split("__")
        output.append({
            "panel": "a", "record_type": "probe_cell", "context": "development split",
            "condition_or_variant": cell, "x_value": value, "y_value": query, "value": value,
            "metric": "MBRU", "reference": candidate, "owning_section": "sec:analysis",
            "note": "one bilinear probe scorer per factorial cell",
        })
    for query, value in a["candidate_swap"].items():
        output.append({
            "panel": "a", "record_type": "candidate_swap", "context": "development split",
            "condition_or_variant": query, "value": value, "metric": "delta MBRU",
            "reference": "static knowledge to measured response", "owning_section": "sec:analysis",
            "note": "difference of recorded cell medians",
        })
    for held, entry in rows["panel_b"]["contexts"].items():
        for condition in ("M_spread", "three_M"):
            output.append({
                "panel": "b", "record_type": "plotted_level", "context": held,
                "condition_or_variant": condition, "value": entry[condition],
                "metric": "paired delta MBRU", "reference": "frozen K562 base scorer",
                "owning_section": "sec:analysis", "note": "same perturbation identities",
            })
    for item in rows["panel_c"]["variants"]:
        output.append({
            "panel": "c", "record_type": "plotted_contrast", "context": "pooled",
            "condition_or_variant": item["label"], "value": item["median"],
            "x_value": item["median"], "ci95_low": item["ci95"][0],
            "ci95_high": item["ci95"][1], "metric": "paired delta BU@20",
            "reference": "unit-direction ridge", "owning_section": "sec:analysis",
            "note": item["group"],
        })
    for item in rows["panel_d"]["grid_points"]:
        output.append({
            "panel": "d", "record_type": "grid_point", "context": item["held"],
            "condition_or_variant": item["estimator"], "x_value": item["prediction"],
            "y_value": item["design_effect_only_BU@20"],
            "value": item["design_effect_only_BU@20"], "metric": "effect-only BU@20",
            "reference": "MASKED evidence grid", "owning_section": "sec:analysis",
            "note": item["atlas"],
        })
    for item in rows["panel_d"]["controlled_changes"]:
        output.append({
            "panel": "d", "record_type": "controlled_change", "context": item["held"],
            "condition_or_variant": item["pair"], "x_value": item["d_prediction"],
            "y_value": item["d_design"], "value": item["d_design"],
            "metric": "delta effect-only BU@20", "reference": "within-atlas full specification",
            "owning_section": "sec:analysis", "note": f"same_sign={item['same_sign']}",
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in output:
            writer.writerow({field: item.get(field, "") for field in fields})


def audit_data(rows: dict, rows_path: Path, plot_script: Path, plotting_commit: str) -> dict:
    return {
        "schema": "VCDESIGN_FIGURE_4_SOURCE_AUDIT_V1",
        "panel_contracts": {
            "a": {"artifact": "factorial probe", "statistic": rows["panel_a"]["statistic"], "metric": "MBRU"},
            "b": {"artifact": "same-identity measurement control", "statistic": rows["panel_b"]["statistic"], "metric": "MBRU"},
            "c": {"artifact": "capacity and knowledge interventions", "statistic": rows["panel_c"]["statistic"], "metric": "BU@20", "reference": "RIDGE_UNIT"},
            "d": {"artifact": "prediction-design closure", "prediction_metric": rows["panel_d"]["prediction_metric"], "design_metric": rows["panel_d"]["design_metric"], "regime": "MASKED"},
        },
        "source_artifact": {"sha256": sha256(rows_path)},
        "source_records": rows["sources"],
        "plotting_commit": plotting_commit,
        "plot_script_sha256": sha256(plot_script),
        "build_script_sha256": sha256(Path(__file__)),
        "assertions": rows["mandatory_assertions"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-csv", required=True, type=Path)
    parser.add_argument("--audit-json", required=True, type=Path)
    parser.add_argument("--supp-output", required=True, type=Path)
    parser.add_argument("--plot-script", required=True, type=Path)
    parser.add_argument("--plotting-commit", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    rows = json.loads(args.rows.read_text())
    validate(rows)
    write_json(args.output, main_data(rows, args.rows), args.overwrite)
    write_source_csv(args.source_csv, rows, args.overwrite)
    write_json(args.audit_json, audit_data(rows, args.rows, args.plot_script, args.plotting_commit), args.overwrite)
    write_json(
        args.supp_output,
        {
            "schema": "VCDESIGN_SUPP_ATLAS_CONTEXT_DATA_V1",
            "generated_by": "figures/scripts/build_fig4_data.py",
            "source": {"sha256": sha256(args.rows)},
            "panel": rows["supplementary"]["atlas_context_curve"],
        },
        args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
