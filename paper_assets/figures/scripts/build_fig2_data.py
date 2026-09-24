#!/usr/bin/env python3
"""Build Figure 2 source data from the frozen K562 record and real row values.

Nothing is trained, refit, or simulated here. ``--locked`` supplies the canonical
summary statistics and bootstrap intervals. ``--rows`` supplies the 684 row-level
values reproduced by ``extract_fig2_rows.py`` from the same frozen checkpoints.

    python3 figures/scripts/build_fig2_data.py \
        --locked locked_evaluation.json \
        --rows figures/data/fig2_k562_rows.json \
        --output figures/data/fig2_hero_data.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


BUDGETS = (10, 20, 50)
METHODS = (
    ("Goal-blind prior", "STRICT_LEGAL_CANDIDATE_PRIOR", "baseline"),
    ("Forward-then-match", "EFFECT_BRANCH_ONLY", "baseline"),
    ("Base scorer", "CLEAN_BASE", "baseline"),
    ("Text-augmented ranker", "LEGACY_M_SET", "baseline"),
    ("VCDesign-CED", "CLEAN_EFFECT", "method"),
)
PAIRED = (
    ("Base scorer", "CLEAN_EFFECT_minus_CLEAN_BASE"),
    ("Text-augmented ranker", "CLEAN_EFFECT_minus_LEGACY_M_SET"),
)
TOLERANCE = 1e-12


def _source(path: Path, kind: str) -> dict:
    return {
        "kind": kind,
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _close(label: str, observed: float, expected: float) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=TOLERANCE):
        raise SystemExit(f"{label}: observed {observed:.17g}, expected {expected:.17g}")


def _summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    p05, q25, median, q75, p95 = np.quantile(array, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "minimum": float(array.min()),
        "p05": float(p05),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "p95": float(p95),
        "maximum": float(array.max()),
    }


def build(locked: dict, rows: dict, locked_path: Path, rows_path: Path) -> dict:
    if rows.get("schema") != "VCDESIGN_FIG2_K562_ROWS_V1":
        raise SystemExit(f"unexpected row-data schema: {rows.get('schema')}")
    locked_sha = hashlib.sha256(locked_path.read_bytes()).hexdigest()
    if rows["provenance"]["reference_record"]["sha256"] != locked_sha:
        raise SystemExit("row data was not reproduced from the supplied locked record")

    gate = locked["query_gate"]
    population = rows["population"]
    expected_rows = int(locked["paired"][PAIRED[0][1]]["bootstrap"]["clusters"])
    checks = {
        "queries": (int(population["queries"]), int(gate["eligible"])),
        "query-view rows": (int(population["query_view_rows"]), expected_rows),
        "candidates per row": (int(population["candidates_per_row"]), int(gate["nominal"]) - 1),
    }
    for label, (observed, expected) in checks.items():
        if observed != expected:
            raise SystemExit(f"{label}: row data has {observed}, locked record has {expected}")

    panel_a = {"methods": {}}
    for name, arm, role in METHODS:
        values = [float(value) for value in rows["mbu"][name]]
        if len(values) != expected_rows:
            raise SystemExit(f"{name}: expected {expected_rows} rows, found {len(values)}")
        summary = _summary(values)
        frozen = locked["rows"][arm]["direction"]["MBRU"]
        for key in ("median", "p05", "p95"):
            _close(f"{name} {key}", summary[key], float(frozen[key]))
        panel_a["methods"][name] = {
            "record_arm": arm,
            "role": role,
            "values": values,
            "summary": summary,
        }
    panel_a["statistic"] = (
        "per-query-view mean budgeted utility, the average of BU@10, BU@20 and BU@50; "
        "half-density from the 684 observed rows, with median, IQR and 5th-95th percentiles"
    )

    panel_b = {"comparisons": {}}
    for comparator, record_key in PAIRED:
        values = [float(value) for value in rows["paired_delta_mbu"][comparator]]
        summary = _summary(values)
        frozen = locked["paired"][record_key]
        win_rate = float(np.mean(np.asarray(values) > 0))
        _close(f"VCDesign-CED - {comparator} median", summary["median"], float(frozen["median"]))
        _close(
            f"VCDesign-CED - {comparator} win rate",
            win_rate,
            float(frozen["fraction_query_improved"]),
        )
        panel_b["comparisons"][comparator] = {
            "record_key": f"paired.{record_key}",
            "values": values,
            "summary": summary,
            "ci95": [float(value) for value in frozen["bootstrap"]["median"]["ci95"]],
            "win_rate": win_rate,
        }
    panel_b["statistic"] = (
        "empirical cumulative distribution of paired row-level mean-BU differences, "
        "VCDesign-CED minus comparator"
    )

    panel_c = {"budgets": list(BUDGETS), "methods": {}}
    for name, arm in (
        ("Base scorer", "CLEAN_BASE"),
        ("Text-augmented ranker", "LEGACY_M_SET"),
        ("VCDesign-CED", "CLEAN_EFFECT"),
    ):
        medians = []
        for budget in BUDGETS:
            values = np.asarray(rows["high_value_hits"][name][str(budget)], dtype=float)
            median = float(np.median(values))
            frozen = float(
                locked["rows"][arm]["direction"][f"HighValueCount@{budget}"]["median"]
            )
            _close(f"{name} HvHit@{budget}", median, frozen)
            medians.append(median)
        panel_c["methods"][name] = medians
    panel_c["statistic"] = (
        "median over query-view rows of the number of top-B candidates at or above the "
        "query-specific 95th utility percentile"
    )

    return {
        "schema": "VCDESIGN_FIG2_DATA_V3",
        "generated_by": "figures/scripts/build_fig2_data.py",
        "sources": {
            "locked": _source(locked_path, "frozen K562 locked-evaluation record"),
            "rows": _source(rows_path, "reproduced row-level Figure 2 source data"),
        },
        "population": population,
        "panel_a": panel_a,
        "panel_b": panel_b,
        "panel_c": panel_c,
    }


def write_source_csv(path: Path, rows: dict) -> None:
    """Write one auditable row per query-view observation used in Figure 2."""
    method_keys = {
        "Goal-blind prior": "goal_blind_mbu",
        "Forward-then-match": "forward_match_mbu",
        "Base scorer": "base_mbu",
        "Text-augmented ranker": "text_mbu",
        "VCDesign-CED": "vcdesign_ced_mbu",
    }
    hit_keys = {
        "Base scorer": "base",
        "Text-augmented ranker": "text",
        "VCDesign-CED": "vcdesign_ced",
    }
    fields = ["query", "view", *method_keys.values(), "ced_minus_base", "ced_minus_text"]
    fields += [f"{prefix}_hvhit_{budget}" for prefix in hit_keys.values() for budget in BUDGETS]
    fields += ["owning_section"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, identity in enumerate(rows["rows"]):
            record = {"query": identity["query"], "view": identity["view"]}
            for method, field in method_keys.items():
                record[field] = rows["mbu"][method][index]
            record["ced_minus_base"] = rows["paired_delta_mbu"]["Base scorer"][index]
            record["ced_minus_text"] = rows["paired_delta_mbu"]["Text-augmented ranker"][index]
            for method, prefix in hit_keys.items():
                for budget in BUDGETS:
                    record[f"{prefix}_hvhit_{budget}"] = rows["high_value_hits"][method][str(budget)][index]
            record["owning_section"] = "sec:heldout"
            writer.writerow(record)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--locked", required=True, type=Path)
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-csv", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {args.output}; pass --overwrite explicitly")
    if args.source_csv is not None and args.source_csv.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {args.source_csv}; pass --overwrite explicitly")
    rows = json.loads(args.rows.read_text())
    data = build(
        json.loads(args.locked.read_text()),
        rows,
        args.locked,
        args.rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")
    if args.source_csv is not None:
        write_source_csv(args.source_csv, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
