#!/usr/bin/env python3
"""Export deterministic CSV views of the external-baseline closure source data."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "VCDESIGN_EXTERNAL_BASELINE_CLOSURE_SOURCE_DATA_V1"


def scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".12g")
    return value


def csv_text(fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: scalar(row.get(key)) for key in fieldnames})
    return stream.getvalue()


def exports(data: dict[str, Any]) -> dict[str, str]:
    if data.get("schema") != SCHEMA:
        raise ValueError(f"expected schema {SCHEMA!r}, found {data.get('schema')!r}")

    out: dict[str, str] = {}
    out["external_baseline_levels.csv"] = csv_text(
        [
            "regime", "method_id", "method", "role", "rpe1", "hepg2", "jurkat",
            "pooled", "qwr20_vs_base", "mu20_pooled", "hvhit20_pooled", "note",
        ],
        data["levels"],
    )
    out["external_baseline_contrasts.csv"] = csv_text(
        [
            "regime", "contrast_id", "contrast", "pooled", "ci95_low", "ci95_high",
            "rpe1", "hepg2", "jurkat", "qwr", "headline",
        ],
        data["contrasts"],
    )
    out["phr20_coverage.csv"] = csv_text(
        [
            "context", "candidates", "full_stratum_candidates", "queries",
            "full_stratum_queries", "positive_fraction", "powered",
        ],
        data["phr20"]["coverage"],
    )
    out["phr20_levels.csv"] = csv_text(
        ["method", "method_regime", "rpe1", "hepg2", "jurkat"],
        data["phr20"]["levels"],
    )
    out["phr20_contrasts.csv"] = csv_text(
        [
            "contrast", "contexts", "mean", "mean_ci95_low", "mean_ci95_high", "median",
            "preregistered_median_state", "rpe1_mean", "hepg2_mean_underpowered", "jurkat_mean",
        ],
        data["phr20"]["contrasts"],
    )
    out["open_vocabulary_retention.csv"] = csv_text(
        ["method", "context", "seen", "masked", "retention"],
        data["open_vocabulary_retention"],
    )
    out["compatibility_reproduction.csv"] = csv_text(
        [
            "method", "candidate_response_required", "unseen_candidate_supported",
            "masked_status", "seen_status", "core_change_required", "reproduction_status",
            "paper_treatment", "note",
        ],
        data["compatibility_reproduction"],
    )
    out["cellnavi_identification_companion.csv"] = csv_text(
        ["context", "classes", "query_class_median_rank"],
        data["cellnavi_identification_companion"],
    )
    out["prediction_design_summary.csv"] = csv_text(
        ["scope", "regime", "comparison_count", "statistic", "value", "p_value", "note"],
        data["prediction_design"],
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()

    data = json.loads(args.source.read_text(encoding="utf-8"))
    rendered = exports(data)
    failures: list[str] = []

    if args.write:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for name, content in rendered.items():
            (args.out_dir / name).write_text(content, encoding="utf-8")
    else:
        for name, expected in rendered.items():
            path = args.out_dir / name
            if not path.exists():
                failures.append(f"missing: {path}")
            elif path.read_text(encoding="utf-8") != expected:
                failures.append(f"stale: {path}")
        if failures:
            raise SystemExit("\n".join(failures))

    print(f"{len(rendered)} CSV files {'written' if args.write else 'verified'}")


if __name__ == "__main__":
    main()
