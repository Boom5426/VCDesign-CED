#!/usr/bin/env python3
"""Is the seen-versus-unseen contrast a statement about candidates or about pool size?

Added on 2026-09-18, after the Phase A size-matched null came back.  That null was
meant to protect the leverage stratification and it exposed something larger: random
subsets of the K562 fold's 6255 unseen candidates give a positive delta MBRU at a
stratum size of 1251, while the whole 6255 pool gives a negative one.  The candidates
are the same, the predictor is the same, the queries are the same.  Only how many
candidates the query has to rank changed.

C-014 compared 1570 seen candidates against 6255 unseen ones.  If delta MBRU moves
with pool size at fixed candidate class, that comparison is confounded by a factor
nobody controlled, and this programme has already been burned once by ranking two
methods on candidate sets of different sizes.  So the curve is measured directly:
delta MBRU against pool size, separately inside the seen pool and inside the unseen
pool, at sizes both of them can reach.

Nothing here is a new model.  It reuses the same three-context pooled predictor, the
same base, the same utility and the same metric.

The mechanism this is testing for is not subtle.  The first B slots of a larger pool
are filled by the largest order statistics of the score, so a score with systematic
error puts more of its errors into the budget when there are more candidates to draw
them from.  A method can therefore help at one vocabulary size and hurt at another
without anything about the method or the candidates changing.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..four_context_v1 import contract as fc
from ..four_context_v1 import evaluate as ev
from ..four_context_v1 import harness as hn
from ..four_context_v1 import predictors as pd
from . import contract as ov

# Fixed here before the curve runs.  The sizes are the ones both pools can reach plus
# the two pools' own full sizes, and the seeds are the programme's usual three.
SIZES = (250, 500, 1000, 1570, 3000, 6255)
SEEDS = (20260918, 20260919, 20260920)

# A mechanism probe, not a comparison of record.  D-005 stands: random is never the
# comparison this programme judges on.  The claim being tested is that a score carrying
# error delivers more of that error into a fixed budget as the pool it ranks grows, so a
# deliberately degraded score should trace the same shape as the effect branch does.  The
# probe enters the fusion exactly where the effect branch enters it, as
# row_z(base) + lambda * row_z(noise), and the three weights are fixed here rather than
# chosen to match any observed curve.  Only the shape of the curve is read, never its level.
NOISE_WEIGHTS = (0.25, 0.5, 1.0)
NOISE_SEED = 20260921


def subsample(rows: np.ndarray, size: int, seed: int) -> np.ndarray:
    if size >= rows.size:
        return rows
    generator = np.random.default_rng(seed)
    return np.sort(rows[generator.choice(rows.size, size=size, replace=False)])


def run(args: argparse.Namespace) -> dict:
    packs = {name: hn.load_pack(name, args.packs) for name in fc.CONTEXTS}
    pack = packs[ov.NATURAL_FOLD]
    pool = hn.training_pool(packs, ov.NATURAL_TRAINING)
    fitted = pd.fit(pool["features"], pool["responses"], pool["present"], pool["keys"],
                    ov.NATURAL_TRAINING)
    effect = fitted.effect(pack.features, pack.present)
    arms = {"BASE": pack.base, "EFFECT": hn.effect_arms(pack, effect)}

    generator = np.random.default_rng(NOISE_SEED)
    noise = [generator.standard_normal(pack.base[view].shape) for view in (0, 1)]
    for weight in NOISE_WEIGHTS:
        arms[f"NOISE[{weight}]"] = [
            ev.row_z(pack.base[view]) + weight * ev.row_z(noise[view]) for view in (0, 1)]

    supervised = np.unique([key.split("|", 1)[1] for key in pool["keys"].tolist()])
    seen_mask = np.isin(pack.ids, supervised)
    classes = {"SEEN_CANDIDATE": np.flatnonzero(seen_mask),
               "UNSEEN_CANDIDATE": np.flatnonzero(~seen_mask),
               "ALL": np.arange(pack.ids.size)}

    record: dict = {
        "schema": "VCDESIGN_OPEN_VOCAB_V1_POOL_SIZE_CURVE",
        "read_anchor_G_check": False,
        "why": ("delta MBRU is a property of a candidate pool as well as of a method; "
                "a seen-versus-unseen contrast taken at 1570 against 6255 is not "
                "controlled unless the curve is flat"),
        "sizes": list(SIZES), "seeds": list(SEEDS),
        "predictor": {"rows": fitted.rows, "penalty": fitted.penalty},
        "class_sizes": {name: int(rows.size) for name, rows in classes.items()},
        "curve": {},
    }
    for name, rows in classes.items():
        record["curve"][name] = {}
        for size in SIZES:
            if size > rows.size:
                continue
            entries = []
            for seed in SEEDS:
                chosen = subsample(rows, size, seed)
                mask = np.zeros(pack.ids.size, dtype=bool)
                mask[chosen] = True
                block = ev.evaluate_arms(arms, pack.utility, pack.ids,
                                         pack.self_positions, mask)
                if "skipped" in block:
                    continue
                vectors = block.pop("_vectors")
                paired = ev.paired(vectors["EFFECT"]["MBRU"], vectors["BASE"]["MBRU"])
                entries.append({
                    "seed": seed, "count": int(chosen.size),
                    "delta_MBRU": paired["median"],
                    "ci95": paired["bootstrap"]["median"]["ci95"],
                    "mean_delta_MBRU": paired["bootstrap"]["mean"]["point"],
                    "improved": paired["fraction_query_improved"],
                    "BASE_MBRU": block["arms"]["BASE"]["MBRU"]["median"],
                    "EFFECT_MBRU": block["arms"]["EFFECT"]["MBRU"]["median"],
                    "delta_Mean@50": ev.paired(vectors["EFFECT"]["Mean@50"],
                                               vectors["BASE"]["Mean@50"])["median"],
                    "noise_probe": {
                        f"{weight}": ev.paired(vectors[f"NOISE[{weight}]"]["MBRU"],
                                               vectors["BASE"]["MBRU"])["median"]
                        for weight in NOISE_WEIGHTS}})
                if size >= rows.size:
                    break                       # the full pool has no randomness to average
            record["curve"][name][str(size)] = {
                "draws": entries,
                "median_delta_MBRU": float(np.median([e["delta_MBRU"] for e in entries])),
                "median_improved": float(np.median([e["improved"] for e in entries])),
                "median_BASE_MBRU": float(np.median([e["BASE_MBRU"] for e in entries])),
                "median_delta_Mean@50": float(np.median([e["delta_Mean@50"] for e in entries])),
                "noise_probe": {f"{weight}": float(np.median(
                    [e["noise_probe"][f"{weight}"] for e in entries]))
                    for weight in NOISE_WEIGHTS}}

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record["runtime"] = {"timestamp": datetime.now(timezone.utc).isoformat(),
                         "python": platform.python_version(), "numpy": np.__version__}
    output.write_text(json.dumps(record, indent=2, sort_keys=True, default=float) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="delta MBRU against candidate pool size")
    parser.add_argument("--packs", required=True)
    parser.add_argument("--output", required=True)
    record = run(parser.parse_args())
    print("delta MBRU against pool size, same predictor, same queries, same utility")
    header = "  ".join(f"{size:>8d}" for size in SIZES)
    print(f"{'class':<18}{header}")
    for name, curve in record["curve"].items():
        cells = []
        for size in SIZES:
            entry = curve.get(str(size))
            cells.append(f"{entry['median_delta_MBRU']:+8.4f}" if entry else f"{'.':>8s}")
        print(f"{name:<18}" + "  ".join(cells))
    print(f"\n{'class':<18}" + "  ".join(f"{size:>8d}" for size in SIZES) + "   (improved fraction)")
    for name, curve in record["curve"].items():
        cells = []
        for size in SIZES:
            entry = curve.get(str(size))
            cells.append(f"{entry['median_improved']:8.3f}" if entry else f"{'.':>8s}")
        print(f"{name:<18}" + "  ".join(cells))
    print("\nmechanism probe on the unseen pool: base plus weighted noise, same fusion slot")
    curve = record["curve"]["UNSEEN_CANDIDATE"]
    print(f"{'':<18}" + "  ".join(f"{size:>8d}" for size in SIZES))
    print(f"{'BASE MBRU':<18}" + "  ".join(
        f"{curve[str(s)]['median_BASE_MBRU']:+8.4f}" if str(s) in curve else f"{'.':>8s}"
        for s in SIZES))
    for weight in NOISE_WEIGHTS:
        cells = [f"{curve[str(s)]['noise_probe'][str(weight)]:+8.4f}"
                 if str(s) in curve else f"{'.':>8s}" for s in SIZES]
        print(f"{'noise ' + str(weight):<18}" + "  ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
