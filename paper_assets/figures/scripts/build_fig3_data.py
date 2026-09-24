#!/usr/bin/env python3
"""Build Figure 3, its source-data CSV, audit JSON, and SI scaling data.

The main figure is built only from the canonical RIDGE_UNIT / BU@20 extraction.
The former main-text vocabulary-scaling panel is preserved as a separately labelled
raw-target, MBRU supplementary stress test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


SCHEMA_IN = "VCDESIGN_FIG3_ROWS_V2"
SCHEMA_OUT = "VCDESIGN_FIG3_DATA_V2"
LEGACY_SCHEMA = "VCDESIGN_FIG3_DATA_V1"
HELD = ("RPE1", "HepG2", "Jurkat")
STATES = ("full_measured", "size_controlled", "response_unseen")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: dict, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise RuntimeError(f"refusing to overwrite existing output without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def validate(rows: dict) -> None:
    if rows.get("schema") != SCHEMA_IN:
        raise RuntimeError(f"unexpected row source schema: {rows.get('schema')}")
    if rows["estimator"]["id"] != "RIDGE_UNIT" or rows["estimator"]["metric"] != "BU@20":
        raise RuntimeError("Figure 3 must use RIDGE_UNIT and BU@20 throughout")
    assertions = rows["mandatory_assertions"]
    if not all(assertions.values()):
        raise RuntimeError(f"a mandatory Figure 3 assertion failed: {assertions}")
    masked = rows["panel_a"]["headline_contrasts"][
        "MASKED|VCDesign-CED_minus_Base"
    ]["median"]
    retrieval = rows["panel_a"]["headline_contrasts"][
        "SEEN|Profile_retrieval_minus_VCDesign-CED"
    ]["median"]
    if masked <= 0 or retrieval <= 0:
        raise RuntimeError("the canonical solver-regime relationships are not satisfied")
    for held in HELD:
        points = rows["panel_c"]["contexts"][held]["points"]
        if [point["share"] for point in points] != [0.0, 0.25, 0.5, 0.75, 1.0]:
            raise RuntimeError(f"{held}: mixed deployment does not contain the five frozen shares")


def main_data(rows: dict, rows_path: Path) -> dict:
    panel_c = {
        **rows["panel_c"],
        "range_by_context": {
            held: max(point["median"] for point in rows["panel_c"]["contexts"][held]["points"])
            - min(point["median"] for point in rows["panel_c"]["contexts"][held]["points"])
            for held in HELD
        },
    }
    return {
        "schema": SCHEMA_OUT,
        "generated_by": "figures/scripts/build_fig3_data.py",
        "source": {
            "kind": "canonical Figure 3 row extraction",
            "sha256": sha256(rows_path),
        },
        "contract": {
            "core_conclusion": (
                "candidate-response availability changes the appropriate solver, while "
                "VCDesign-CED retains positive design gain for response-unseen candidates"
            ),
            "estimator": rows["estimator"],
            "final_size_inches": [5.5, 3.55],
            "archetype": "asymmetric quantitative figure",
            "panel_map": {
                "a": "solver landscape across MASKED and SEEN regimes",
                "b": "direct three-state controlled masking trajectories",
                "c": "nested mixed-information deployment test",
            },
        },
        "panel_a": rows["panel_a"],
        "panel_b": rows["panel_b"],
        "panel_c": panel_c,
        "mandatory_assertions": rows["mandatory_assertions"],
    }


def supplementary_scaling(legacy: dict, legacy_path: Path) -> dict:
    if legacy.get("schema") != LEGACY_SCHEMA:
        raise RuntimeError(f"unexpected legacy Figure 3 schema: {legacy.get('schema')}")
    old_sources = {
        key: {"kind": key, "sha256": value["sha256"]}
        for key, value in legacy["sources"].items()
        if key in ("poolsize", "masking", "loco")
    }
    return {
        "schema": "VCDESIGN_SUPP_VOCAB_SCALING_DATA_V1",
        "generated_by": "figures/scripts/build_fig3_data.py",
        "source": {
            "kind": "former Figure 3 vocabulary-scaling artifact",
            "sha256": sha256(legacy_path),
        },
        "source_artifacts": old_sources,
        "scope": {
            "estimator": "raw-target CED",
            "metric": "paired MBRU difference over the frozen K562 base scorer",
            "population": "K562 held out; development-split queries",
            "reason_in_supplement": (
                "different estimator, metric, and population from the canonical main Figure 3"
            ),
        },
        "panel": legacy["panel_c"],
    }


def write_source_csv(path: Path, rows: dict, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise RuntimeError(f"refusing to overwrite existing output without --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "panel", "record_type", "context", "regime_or_state", "method", "share",
        "query", "view", "value", "ci95_low", "ci95_high", "metric",
        "owning_section", "note",
    ]
    output = []
    for method, entry in rows["panel_a"]["methods"].items():
        for regime, value in entry["levels"].items():
            output.append({
                "panel": "a", "record_type": "plotted_level", "context": "pooled",
                "regime_or_state": regime, "method": entry["label"], "value": value,
                "metric": "BU@20", "owning_section": "sec:context",
                "note": "median over pooled query-view rows",
            })
    for contrast, entry in rows["panel_a"]["headline_contrasts"].items():
        output.append({
            "panel": "a", "record_type": "headline_contrast", "context": "pooled",
            "regime_or_state": contrast.split("|", 1)[0], "method": contrast.split("|", 1)[1],
            "value": entry["median"], "ci95_low": entry["ci95"][0],
            "ci95_high": entry["ci95"][1], "metric": "paired delta BU@20",
            "owning_section": "sec:context", "note": "query-gene cluster bootstrap",
        })
    for held in HELD:
        for state in STATES:
            summary = rows["panel_b"]["contexts"][held]["states"][state]
            output.append({
                "panel": "b", "record_type": "plotted_summary", "context": held,
                "regime_or_state": state, "method": "VCDesign-CED minus Base",
                "value": summary["median"], "ci95_low": summary["ci95"][0],
                "ci95_high": summary["ci95"][1], "metric": "paired delta BU@20",
                "owning_section": "sec:context", "note": "direct per-row scores",
            })
        source = rows["row_source"][held]
        for index, (query, view) in enumerate(zip(source["query"], source["view"])):
            for state in STATES:
                output.append({
                    "panel": "b", "record_type": "source_row", "context": held,
                    "regime_or_state": state, "method": "VCDesign-CED minus Base",
                    "query": query, "view": view,
                    "value": source["gain_BU20_over_BASE"][state][index],
                    "metric": "paired delta BU@20", "owning_section": "sec:context",
                    "note": "query-view row",
                })
    for state in STATES:
        summary = rows["panel_b"]["pooled"][state]
        output.append({
            "panel": "b", "record_type": "plotted_summary", "context": "pooled",
            "regime_or_state": state, "method": "VCDesign-CED minus Base",
            "value": summary["median"], "ci95_low": summary["ci95"][0],
            "ci95_high": summary["ci95"][1], "metric": "paired delta BU@20",
            "owning_section": "sec:context", "note": "query-gene cluster bootstrap",
        })
    for held in HELD:
        for point in rows["panel_c"]["contexts"][held]["points"]:
            output.append({
                "panel": "c", "record_type": "plotted_summary", "context": held,
                "regime_or_state": "mixed", "method": "VCDesign-CED minus Base",
                "share": point["share"], "value": point["median"],
                "metric": "paired delta BU@20", "owning_section": "sec:context",
                "note": f"realised response-unseen share={point['realised_share']:.12g}",
            })
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in output:
            writer.writerow({field: record.get(field, "") for field in fields})


def audit_data(rows: dict, rows_path: Path, plot_script: Path, plotting_commit: str) -> dict:
    return {
        "schema": "VCDESIGN_FIGURE_3_SOURCE_AUDIT_V1",
        "estimator_version": rows["estimator"],
        "metric": "BU@20",
        "pool_definition": {
            "panel_a": "pooled C-016 eligible strata across RPE1, HepG2, and Jurkat",
            "panel_b": "same queries, eligible pools, static knowledge, BASE, and evaluation",
            "panel_c": rows["panel_c"]["pool_definition"],
        },
        "arm_definition": rows["panel_b"]["arm_definition"],
        "source_artifact": {"sha256": sha256(rows_path)},
        "source_records": rows["sources"],
        "plotting_commit": plotting_commit,
        "plot_script_sha256": sha256(plot_script),
        "build_script_sha256": sha256(Path(__file__)),
        "fit_reproduction_max_abs_error": {
            held: max(
                [entry["crossfit_max_abs_error"]]
                + [fold["reference_max_abs_error"] for fold in entry["folds"].values()]
            )
            for held, entry in rows["fit_reproduction_audit"].items()
        },
        "assertions": rows["mandatory_assertions"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument(
        "--legacy",
        type=Path,
        help="optional former V1 artifact; required only when --supp-output is requested",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-csv", required=True, type=Path)
    parser.add_argument("--audit-json", required=True, type=Path)
    parser.add_argument(
        "--supp-output",
        type=Path,
        help="optional sanitized SI scaling output; requires --legacy",
    )
    parser.add_argument("--plot-script", required=True, type=Path)
    parser.add_argument("--plotting-commit", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    rows = json.loads(args.rows.read_text())
    if (args.legacy is None) != (args.supp_output is None):
        parser.error("--legacy and --supp-output must be supplied together")
    validate(rows)
    write_json(args.output, main_data(rows, args.rows), args.overwrite)
    write_source_csv(args.source_csv, rows, args.overwrite)
    write_json(
        args.audit_json,
        audit_data(rows, args.rows, args.plot_script, args.plotting_commit),
        args.overwrite,
    )
    if args.legacy is not None:
        legacy = json.loads(args.legacy.read_text())
        write_json(
            args.supp_output,
            supplementary_scaling(legacy, args.legacy),
            args.overwrite,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
