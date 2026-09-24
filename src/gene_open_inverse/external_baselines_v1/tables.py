#!/usr/bin/env python3
"""Paper-ready data tables (TSV) from ``external_baselines_v1.json``.  Numbers only; no prose."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import contract as cc

SCOPES = ("RPE1", "HepG2", "Jurkat", "pooled")
LABEL = {cc.RANDOM: "Random", cc.PRIOR: "Goal-blind prior", cc.BASE: "BASE",
         cc.CELLNAVI: "CellNavi (official recipe, all training identities as classes)",
         cc.CELLNAVI_NATIVE: "CellNavi (native class scale: the candidate library)",
         cc.GEARS: "GEARS (forward-enumerate)", cc.INTERNAL_FTM: "Internal forward-then-match (RIDGE_UNIT effect only)",
         cc.VCDESIGN: "VCDesign (BASE + RIDGE_UNIT)", cc.ORACLE: "Measured-response oracle (analysis only)",
         cc.RIDGE_K562: "RIDGE_UNIT, K562-only atlas (GEARS-matched)", cc.PROFILED: "Profiled signature retrieval (SEEN only)"}


def _fmt(x, digits=4):
    return "NA" if x is None else f"{x:.{digits}f}"


MAIN = tuple(a for a in cc.MAIN_ROWS if a != cc.GEARS)
COMPANION = (cc.GEARS, cc.RIDGE_K562, cc.PROFILED)


def main_table(result: dict, regime: str, metrics: tuple, arms: tuple = MAIN) -> list[str]:
    header = ["arm"] + [f"{s}|{m}" for s in SCOPES for m in metrics]
    lines = ["\t".join(header)]
    for arm in arms:
        entry = result["main_table"][regime].get(arm, "NOT_AVAILABLE")
        row = [LABEL[arm]]
        for scope in SCOPES:
            for metric in metrics:
                if entry == "NOT_AVAILABLE":
                    row.append("NA")
                elif metric == "QWR@20":
                    row.append(_fmt(entry[scope]["QWR@20_vs_BASE"]["paper"], 3))
                elif metric == "HvHit@20":
                    row.append(_fmt(entry[scope]["HvHit@20"]["mean"], 2))
                elif metric == "PHR@20":
                    block = entry[scope].get("PHR@20_four_way_subpool", {})
                    value = block.get("pooled_powered") if scope == "pooled" else block.get(scope)
                    row.append(_fmt(value["mean"], 3) if value else "NA")
                else:
                    row.append(_fmt(entry[scope][metric]["median"]))
        lines.append("\t".join(row))
    return lines


def contrasts(result: dict) -> list[str]:
    lines = ["\t".join(["regime", "contrast", "median", "ci_low", "ci_high", "mean", "state", "QWR_wins_over_all",
                        "RPE1", "HepG2", "Jurkat"])]
    for regime, block in result["contrasts"].items():
        for name, e in block.items():
            if e is None:
                lines.append("\t".join([regime, name] + ["NA"] * 9))
                continue
            lines.append("\t".join([regime, name, _fmt(e["median"]), _fmt(e["median_ci95"][0]), _fmt(e["median_ci95"][1]),
                                    _fmt(e["mean"]), e["state"], _fmt(e["qwr"]["paper"], 3)]
                                   + [f"{_fmt(e['by_context'][h]['median'])} ({e['by_context'][h]['state']})" for h in cc.PRIMARY_CONTEXTS]))
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="data tables")
    parser.add_argument("--run-dir", dest="run_dir", required=True)
    args = parser.parse_args()
    run = Path(args.run_dir)
    result = json.loads((run / "external_baselines_v1.json").read_text())
    out = run / "tables"
    out.mkdir(exist_ok=False)
    main_metrics = ("BU@20", "QWR@20", "MU@20", "HvHit@20", "PHR@20")
    (out / "main_table_MASKED.tsv").write_text("\n".join(main_table(result, "MASKED", main_metrics)) + "\n")
    (out / "main_table_SEEN.tsv").write_text("\n".join(main_table(result, "SEEN", main_metrics)) + "\n")
    for regime in cc.REGIMES:
        (out / f"companion_table_{regime}.tsv").write_text(
            f"# GEARS: {cc.GEARS_REPRO_VERDICT}; {cc.GEARS_PLACEMENT}\n"
            + "\n".join(main_table(result, regime, main_metrics, COMPANION)) + "\n")
    for regime in cc.REGIMES:
        (out / f"companion_BU10_BU50_{regime}.tsv").write_text(
            "\n".join(main_table(result, regime, ("BU@10", "BU@50"), MAIN + COMPANION)) + "\n")
    (out / "paired_contrasts.tsv").write_text("\n".join(contrasts(result)) + "\n")
    for name in sorted(p.name for p in out.iterdir()):
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
