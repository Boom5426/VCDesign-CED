#!/usr/bin/env python3
"""Do the masking phase's support terciles survive the leverage diagnostic's defect?

Phase A found that the primary leverage is dominated by feature columns the training
atlas never varies, whose standardized scale is the 1e-8 floor, and that removing them
reverses which unseen candidates look best supported.  Phase B splits its masked
candidates by that same primary leverage.  This recomputes those terciles with the
degenerate columns removed and re-evaluates, using the effect cosines the masking run
cached rather than refitting anything, so the arms are numerically the ones that phase
produced.

This is a sensitivity, not a replacement.  The pre-registered primary stays as it is.
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
from . import contract as ov
from .natural import leverage_from_matrix, paired_all, quantile_strata


def run(args: argparse.Namespace) -> dict:
    packs = {name: hn.load_pack(name, args.packs) for name in fc.CONTEXTS}
    masking = json.loads(Path(args.record).read_text())
    source = Path(args.record).parent
    record: dict = {"schema": "VCDESIGN_OPEN_VOCAB_V1_LEVERAGE_SENSITIVITY",
                    "note": ("the pre-registered primary leverage is unchanged; this drops "
                             "only the feature columns whose training standard deviation is "
                             "at the 1e-8 clip, which the frozen ridge gives exactly zero "
                             "weight and which therefore cannot move any prediction"),
                    "contexts": {}}
    for held in fc.CONTEXTS:
        pack = packs[held]
        block = masking["cross_context"][held]
        training = tuple(block["training"])
        pool = hn.training_pool(packs, training)
        keep = np.flatnonzero(np.asarray(pool["present"], dtype=bool))
        features = pool["features"][keep]
        degenerate = features.std(axis=0) <= 1e-8
        healthy = np.flatnonzero(~degenerate)
        penalty = float(block["seen_predictor"]["penalty"])
        restricted = leverage_from_matrix(features, pack.features, penalty, healthy)

        with np.load(source / f"support_{held}.npz", allow_pickle=False) as table:
            primary = np.asarray(table["leverage"], dtype=np.float64)
        with np.load(source / f"effect_cosine_{held}.npz", allow_pickle=False) as cached:
            eligible = np.asarray(cached["eligible"], dtype=bool)
            arms = {"BASE": list(pack.base)}
            for label, key in (("EFFECT_SEEN", "seen"), ("EFFECT_MASKED", "masked"),
                               ("EFFECT_SIZE_CONTROL", "size_control")):
                arms[label] = [ev.fuse(pack.base[view], np.asarray(cached[f"{key}_{view}"]))
                               for view in (0, 1)]

        rows = np.flatnonzero(eligible)
        entry: dict = {"degenerate_feature_columns": int(degenerate.sum()),
                       "feature_width": int(features.shape[1]),
                       "penalty": penalty,
                       "spearman_primary_vs_restricted": float(np.corrcoef(
                           np.argsort(np.argsort(primary[rows])),
                           np.argsort(np.argsort(restricted[rows])))[0, 1]),
                       "primary_max": float(primary[rows].max()),
                       "restricted_max": float(restricted[rows].max()),
                       "strata": {}}
        strata = quantile_strata(restricted[rows], ov.MASK_SUPPORT_STRATA, ov.MASK_STRATUM_NAMES)
        for name, inside in strata.items():
            mask = np.zeros(pack.ids.size, dtype=bool)
            mask[rows[inside]] = True
            graded = ev.evaluate_arms(arms, pack.utility, pack.ids, pack.self_positions, mask)
            if "skipped" in graded:
                entry["strata"][name] = graded
                continue
            vectors = graded.pop("_vectors")
            entry["strata"][name] = {
                "count": int(mask.sum()),
                "leverage_median": float(np.median(restricted[rows][inside])),
                "masked_vs_base": ev.paired(vectors["EFFECT_MASKED"]["MBRU"],
                                            vectors["BASE"]["MBRU"])["median"],
                "seen_vs_base": ev.paired(vectors["EFFECT_SEEN"]["MBRU"],
                                          vectors["BASE"]["MBRU"])["median"],
                "exposure_size_controlled": ev.paired(
                    vectors["EFFECT_SIZE_CONTROL"]["MBRU"],
                    vectors["EFFECT_MASKED"]["MBRU"])["median"]}
        record["contexts"][held] = entry

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record["runtime"] = {"timestamp": datetime.now(timezone.utc).isoformat(),
                         "python": platform.python_version(), "numpy": np.__version__}
    output.write_text(json.dumps(record, indent=2, sort_keys=True, default=float) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="leverage sensitivity for the masking terciles")
    parser.add_argument("--packs", required=True)
    parser.add_argument("--record", required=True)
    parser.add_argument("--output", required=True)
    record = run(parser.parse_args())
    for held, entry in record["contexts"].items():
        print(f"\n[{held}] {entry['degenerate_feature_columns']} degenerate columns of "
              f"{entry['feature_width']}, primary max {entry['primary_max']:.3g} against "
              f"restricted {entry['restricted_max']:.3g}, rank correlation "
              f"{entry['spearman_primary_vs_restricted']:+.3f}")
        for name, block in entry["strata"].items():
            if "count" not in block:
                continue
            print(f"  {name:16s} n={block['count']:4d} lev {block['leverage_median']:.4g}  "
                  f"masked {block['masked_vs_base']:+.4f}  seen {block['seen_vs_base']:+.4f}  "
                  f"exposure {block['exposure_size_controlled']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
