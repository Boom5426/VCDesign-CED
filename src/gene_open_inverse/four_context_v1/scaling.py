#!/usr/bin/env python3
"""Phase 9: summarise the scaling, and say which of size and diversity is doing the work.

    Gain = alpha + beta_N log N_rows + beta_C N_contexts

This is a description of the points the earlier phases produced, not a model that
anything is trained with and not a claim about a functional form.  Its only job is to
make the two coefficients separately readable, because the raw tables confound them:
in the natural scaling arms a bigger atlas is always also a more diverse one.

Three fits are reported and they answer different questions.

  full        the natural arms only.  ``N`` and ``C`` move together, so ``beta_C``
              here is not identified and is printed for completeness only.
  matched     the row-count matched arms only.  ``N`` is constant by construction, so
              ``beta_N`` is not identified and ``beta_C`` is the whole answer.
  pooled      both sets together with a per-held-context intercept.

The standard errors are ordinary least squares errors.  Points inside one fold share
a held context and a candidate pool, so they are not independent and these errors are
optimistic.  They are reported as a descriptive spread and no significance claim is
made from them; the claims in this mission rest on the paired bootstraps in the
phases that produced the points.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import contract as fc


def _ols(design: np.ndarray, target: np.ndarray) -> dict:
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    degrees = max(design.shape[0] - design.shape[1], 1)
    sigma = float(residual @ residual) / degrees
    covariance = sigma * np.linalg.pinv(design.T @ design)
    return {"coefficients": coefficients, "standard_errors": np.sqrt(np.maximum(np.diag(covariance), 0.0)),
            "residual_standard_deviation": float(np.sqrt(sigma)), "points": int(design.shape[0])}


def collect(loco: dict, matched: dict, stratum: str = "ALL") -> list[dict]:
    """Every ``(held, rows, contexts, gain)`` point the earlier phases produced."""
    points: list[dict] = []
    for held, fold in loco.get("folds", {}).items():
        block = fold["strata"].get(stratum, {})
        if "paired_vs_base" not in block:
            continue
        for name, entry in fold["predictors"].items():
            item = block["paired_vs_base"].get(name)
            if item is None:
                continue
            points.append({"held": held, "source": "full", "arm": name,
                           "rows": int(entry["rows"]), "contexts": int(entry["n_contexts"]),
                           "gain": float(item["median"])})
    for held, fold in matched.get("folds", {}).items():
        row_matched = fold.get("row_matched", {})
        if "skipped" in row_matched:
            # A fold whose matched budget could not support the frozen rank contributes
            # no points.  It is skipped here rather than crashing on the string that
            # records why, and Phase 8 already reports the reason.
            continue
        for name, entry in row_matched.items():
            if not isinstance(entry, dict):
                continue
            for seed_entry in entry.get("per_seed", []):
                block = seed_entry.get("strata", {}).get(stratum)
                if block is None:
                    continue
                points.append({"held": held, "source": "matched", "arm": name,
                               "rows": int(seed_entry["rows"]), "contexts": int(entry["n_contexts"]),
                               "gain": float(block["delta_MBRU"]["median"]), "seed": seed_entry["seed"]})
    return points


def fit(points: list[dict], with_fixed_effects: bool) -> dict:
    if len(points) < 3:
        return {"skipped": f"only {len(points)} points"}
    rows = np.log(np.asarray([p["rows"] for p in points], dtype=np.float64))
    contexts = np.asarray([p["contexts"] for p in points], dtype=np.float64)
    gain = np.asarray([p["gain"] for p in points], dtype=np.float64)
    columns, names = [], []
    if with_fixed_effects:
        held = np.asarray([p["held"] for p in points])
        for name in fc.CONTEXTS:
            if (held == name).any():
                columns.append((held == name).astype(np.float64))
                names.append(f"intercept[{name}]")
    else:
        columns.append(np.ones(len(points)))
        names.append("intercept")
    identified = {"beta_N_log_rows": float(np.std(rows)) > 1e-9,
                  "beta_C_contexts": float(np.std(contexts)) > 1e-9}
    if identified["beta_N_log_rows"]:
        columns.append(rows)
        names.append("beta_N_log_rows")
    if identified["beta_C_contexts"]:
        columns.append(contexts)
        names.append("beta_C_contexts")
    record = _ols(np.stack(columns, axis=1), gain)
    return {"terms": {name: {"estimate": float(record["coefficients"][index]),
                             "standard_error": float(record["standard_errors"][index])}
                      for index, name in enumerate(names)},
            "identified": identified, "points": record["points"],
            "residual_standard_deviation": record["residual_standard_deviation"]}


def verdict(matched_fit: dict, pooled_fit: dict) -> dict:
    """The Phase 9 classification, read off the two identified coefficients."""
    def positive(block: dict, term: str) -> bool:
        entry = block.get("terms", {}).get(term)
        if entry is None:
            return False
        return entry["estimate"] > 2.0 * max(entry["standard_error"], 1e-12)

    quantity = positive(pooled_fit, "beta_N_log_rows")
    diversity = positive(matched_fit, "beta_C_contexts")
    if quantity and diversity:
        label = "BOTH_SCALE_AND_DIVERSITY_MATTER"
    elif quantity:
        label = "DATA_QUANTITY_DOMINATES"
    elif diversity:
        label = "CONTEXT_DIVERSITY_ADDS_VALUE"
    else:
        label = "NEITHER_COEFFICIENT_IS_SEPARATED_FROM_ITS_SPREAD"
    return {"verdict": label,
            "beta_N_positive_in_pooled_fit": quantity,
            "beta_C_positive_at_matched_rows": diversity,
            "rule": "an estimate counts as positive when it exceeds twice its ordinary "
                    "least squares standard error; those errors are optimistic and the "
                    "claim of record remains the paired bootstrap in Phases 7 and 8"}


def run(args: argparse.Namespace) -> dict:
    loco = json.loads(Path(args.loco).read_text())
    matched = json.loads(Path(args.matched).read_text())
    record: dict = {"schema": "VCDESIGN_FOUR_CONTEXT_V1_SCALING", "strata": {}}
    for stratum in fc.STRATA:
        points = collect(loco, matched, stratum)
        if not points:
            continue
        full = [p for p in points if p["source"] == "full"]
        matched_points = [p for p in points if p["source"] == "matched"]
        entry = {"points": len(points),
                 "full": fit(full, with_fixed_effects=True),
                 "matched": fit(matched_points, with_fixed_effects=True),
                 "pooled": fit(points, with_fixed_effects=True),
                 "per_held_context": {}}
        for held in fc.CONTEXTS:
            subset = [p for p in points if p["held"] == held]
            if subset:
                entry["per_held_context"][held] = fit(subset, with_fixed_effects=False)
        entry["verdict"] = verdict(entry["matched"], entry["pooled"])
        entry["mean_gain_by_context_count"] = {
            str(count): float(np.mean([p["gain"] for p in points if p["contexts"] == count]))
            for count in (1, 2, 3) if any(p["contexts"] == count for p in points)}
        entry["mean_gain_by_context_count_matched_rows_only"] = {
            str(count): float(np.mean([p["gain"] for p in matched_points if p["contexts"] == count]))
            for count in (1, 2, 3) if any(p["contexts"] == count for p in matched_points)}
        record["strata"][stratum] = entry
    record["headline"] = record["strata"].get("ALL", {}).get("verdict")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record["runtime"] = {"timestamp": datetime.now(timezone.utc).isoformat(),
                         "python": platform.python_version(), "numpy": np.__version__}
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 9: scaling verdict")
    parser.add_argument("--loco", required=True)
    parser.add_argument("--matched", required=True)
    parser.add_argument("--output", required=True)
    record = run(parser.parse_args())
    for stratum, entry in record["strata"].items():
        print(f"\n[{stratum}] {entry['points']} points")
        for label in ("full", "matched", "pooled"):
            block = entry[label]
            terms = ", ".join(f"{name} {item['estimate']:+.5f}+-{item['standard_error']:.5f}"
                              for name, item in block.get("terms", {}).items()
                              if name.startswith("beta"))
            print(f"  {label:8s} {terms or 'no identified slope'}")
        print(f"  mean gain by context count {entry['mean_gain_by_context_count']}")
        print(f"  at matched rows            {entry['mean_gain_by_context_count_matched_rows_only']}")
        print(f"  verdict {entry['verdict']['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
