#!/usr/bin/env python3
"""Build the external RPE1 design benchmark.  Trains nothing and ranks nothing.

The recipe is the anchor's, applied to a different context: library-size 1e4
normalization, log1p, per-batch non-targeting control means, and a per-identity
control reference weighted by that identity's cell incidence inside the view's own
batches.  The A/B batch split uses the same hash rule, namespace and seed as the
frozen K562 split, so the notion of "two independent measurement views" is the same
object in both contexts rather than a new one invented here.

Eligibility is the frozen ``QUERY_ELIGIBILITY_V1`` rule applied unchanged.  The
identity split into external fit, select and check roles is a hash of the identity
name, fixed here, before any model result exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import anndata
import numpy as np

from ..utility_gt_v2 import contract as gt
from . import contract as ec


CHUNK = 16384


def _hash_bucket(namespace: str, seed: int, name: str, modulus: int) -> int:
    token = f"{namespace}|{seed}|{name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % modulus


def view_side(batch: str) -> int:
    """The anchor's frozen A/B bit, reimplemented from the recorded rule."""
    token = f"{ec.VIEW_NAMESPACE}|{ec.VIEW_SEED}|{batch}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big") & 1


def quarter_of(batch: str) -> int:
    """The anchor's four-way refinement, so the action can be estimated cross-fitted."""
    token = f"{ec.QUAD_NAMESPACE}|{ec.VIEW_SEED}|{batch}".encode("utf-8")
    refinement = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") & 1
    return view_side(batch) * 2 + refinement


def role_of(identity: str) -> str:
    bucket = _hash_bucket(ec.SPLIT_NAMESPACE, ec.SPLIT_SEED, identity, 10)
    edge = 0
    for role, width in ec.SPLIT_FRACTIONS.items():
        edge += width
        if bucket < edge:
            return role
    raise RuntimeError("split fractions must sum to ten")


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    handle = anndata.read_h5ad(args.source, backed="r")
    perturbation = handle.obs["perturbation"].astype(str).to_numpy()
    batch = handle.obs["batch"].astype(str).to_numpy()
    genes = np.asarray(handle.var_names, dtype=str)
    observations, features = int(handle.n_obs), int(handle.n_vars)

    batches = np.unique(batch)
    sides = np.asarray([view_side(str(name)) for name in batches], dtype=np.int8)
    quarters = np.asarray([quarter_of(str(name)) for name in batches], dtype=np.int8)
    if not np.array_equal(quarters // 2, sides):
        raise RuntimeError("the quarter assignment is not a refinement of the A/B split")
    batch_index = {name: index for index, name in enumerate(batches.tolist())}
    column = np.asarray([batch_index[name] for name in batch.tolist()], dtype=np.int64)

    identities = np.asarray(sorted(set(perturbation.tolist()) - {ec.CONTROL_LABEL}), dtype=str)
    identity_index = {name: index for index, name in enumerate(identities.tolist())}
    row = np.asarray([identity_index.get(name, -1) for name in perturbation.tolist()], dtype=np.int64)

    incidence = np.zeros((identities.size, batches.size), dtype=np.int64)
    keep = row >= 0
    np.add.at(incidence, (row[keep], column[keep]), 1)
    control_per_batch = np.zeros(batches.size, dtype=np.int64)
    np.add.at(control_per_batch, column[~keep], 1)
    if int((control_per_batch == 0).sum()):
        raise RuntimeError("a batch carries no control cell; the per-batch reference is undefined")

    cells_by_side = np.stack([incidence[:, sides == side].sum(axis=1) for side in (0, 1)], axis=1)
    eligible_depth = (cells_by_side >= ec.MIN_VIEW_CELLS).all(axis=1)

    sums = np.zeros((2, identities.size, features), dtype=np.float64)
    counts = np.zeros((2, identities.size), dtype=np.int64)
    quarter_sums = np.zeros((4, identities.size, features), dtype=np.float64)
    quarter_counts = np.zeros((4, identities.size), dtype=np.int64)
    control_sums = np.zeros((batches.size, features), dtype=np.float64)
    control_counts = np.zeros(batches.size, dtype=np.int64)
    bad = 0
    for start in range(0, observations, CHUNK):
        stop = min(start + CHUNK, observations)
        values = np.asarray(handle.X[start:stop], dtype=np.float32)
        library = values.sum(axis=1)
        good = np.isfinite(library) & (library > 0.0)
        bad += int((~good).sum())
        values[good] = np.log1p(values[good] / library[good, None] * ec.LIBRARY_TARGET)
        values[~good] = 0.0
        local_row, local_column = row[start:stop], column[start:stop]
        local_side = sides[local_column]
        local_quarter = quarters[local_column]
        for quarter in (0, 1, 2, 3):
            picked = np.flatnonzero(good & (local_row >= 0) & (local_quarter == quarter))
            if picked.size:
                np.add.at(quarter_sums[quarter], local_row[picked], values[picked])
                np.add.at(quarter_counts[quarter], local_row[picked], 1)
        for side in (0, 1):
            picked = np.flatnonzero(good & (local_row >= 0) & (local_side == side))
            if picked.size:
                np.add.at(sums[side], local_row[picked], values[picked])
                np.add.at(counts[side], local_row[picked], 1)
        picked = np.flatnonzero(good & (local_row < 0))
        if picked.size:
            np.add.at(control_sums, local_column[picked], values[picked])
            np.add.at(control_counts, local_column[picked], 1)
    handle.file.close()

    control_mean = control_sums / np.maximum(control_counts, 1)[:, None]
    responses = np.zeros((2, identities.size, features), dtype=np.float32)
    sources = np.zeros((2, identities.size, features), dtype=np.float32)
    usable = eligible_depth & (counts > 0).all(axis=0)
    for side in (0, 1):
        weights = incidence.astype(np.float64).copy()
        weights[:, sides != side] = 0.0
        total = weights.sum(axis=1, keepdims=True)
        safe = total[:, 0] > 0
        weights[safe] /= total[safe]
        mean = np.zeros_like(sums[side])
        mean[counts[side] > 0] = (sums[side][counts[side] > 0]
                                  / counts[side][counts[side] > 0][:, None])
        reference = weights @ control_mean
        sources[side] = reference.astype(np.float32)
        responses[side] = (mean - reference).astype(np.float32)
    responses[:, ~usable] = 0.0
    sources[:, ~usable] = 0.0

    four_way = (np.stack([incidence[:, quarters == q].sum(axis=1) for q in range(4)], axis=1)
                >= ec.MIN_VIEW_CELLS).all(axis=1)
    quarter_responses = np.zeros((4, identities.size, features), dtype=np.float32)
    for quarter in range(4):
        weights = incidence.astype(np.float64).copy()
        weights[:, quarters != quarter] = 0.0
        total = weights.sum(axis=1, keepdims=True)
        safe = total[:, 0] > 0
        weights[safe] /= total[safe]
        mean = np.zeros_like(quarter_sums[quarter])
        rows = quarter_counts[quarter] > 0
        mean[rows] = quarter_sums[quarter][rows] / quarter_counts[quarter][rows][:, None]
        quarter_responses[quarter] = (mean - weights @ control_mean).astype(np.float32)
    quarter_responses[:, ~(usable & four_way)] = 0.0

    gram = gt.cross_view_gram(responses[0][usable].astype(np.float64),
                              responses[1][usable].astype(np.float64))
    gate = gt.eligibility(gram, ec.ELIGIBILITY_FDR)
    eligible = np.zeros(identities.size, dtype=bool)
    eligible[np.flatnonzero(usable)] = gate.eligible

    roles = np.asarray([role_of(name) for name in identities.tolist()], dtype=str)

    record = {
        "schema": "VCDESIGN_EXTERNAL_CONTEXT_V1_BUILD",
        "context": ec.PRIMARY_CONTEXT, "anchor": ec.ANCHOR_CONTEXT,
        "perturbation_type": ec.PERTURBATION_TYPE, "trained_anything": False,
        "source": {"path": str(args.source), "n_obs": observations, "n_vars": features},
        "batches": {"total": int(batches.size),
                    "side_A": int((sides == 0).sum()), "side_B": int((sides == 1).sum()),
                    "rule": "the anchor's frozen A/B hash, namespace and seed, applied to RPE1 batch names",
                    "control_cells": int(control_counts.sum()),
                    "batches_without_controls": 0},
        "identities": {"total": int(identities.size),
                       "depth_eligible_both_views": int(eligible_depth.sum()),
                       "usable": int(usable.sum()),
                       "min_view_cells": ec.MIN_VIEW_CELLS,
                       "cells_per_identity_median": float(np.median(cells_by_side.sum(axis=1)))},
        "query_gate": {"rule": "QUERY_ELIGIBILITY_V1 applied unchanged", "fdr": ec.ELIGIBILITY_FDR,
                       "eligible": int(eligible.sum()),
                       "eligibility_fraction_of_usable": float(gate.fraction)},
        "role_split": {"namespace": ec.SPLIT_NAMESPACE, "seed": ec.SPLIT_SEED,
                       "counts": {role: int((roles == role).sum()) for role in ec.SPLIT_FRACTIONS}},
        "four_way": {"rule": "the anchor's QUAD refinement namespace and seed",
                     "batches_per_quarter": [int((quarters == q).sum()) for q in range(4)],
                     "identities_four_way_eligible": int(four_way.sum()),
                     "usable_and_four_way": int((usable & four_way).sum())},
        "bad_library_cells_excluded": int(bad),
        "runtime": {"timestamp": datetime.now(timezone.utc).isoformat(),
                    "python": platform.python_version(), "numpy": np.__version__},
    }
    np.save(output / "side_responses.npy", responses)
    np.save(output / "side_sources.npy", sources)
    np.save(output / "control_mean.npy", control_mean.astype(np.float32))
    np.save(output / "quarter_responses.npy", quarter_responses)
    np.save(output / "batch_quarter.npy", quarters)
    np.save(output / "four_way_eligible.npy", four_way)
    np.save(output / "identity_axis.npy", identities)
    np.save(output / "gene_axis.npy", genes)
    np.save(output / "batch_side.npy", sides)
    np.save(output / "batch_axis.npy", batches)
    np.save(output / "incidence.npy", incidence)
    np.save(output / "cells_by_side.npy", cells_by_side)
    np.save(output / "usable.npy", usable)
    np.save(output / "eligible.npy", eligible)
    np.save(output / "roles.npy", roles)
    (output / "EXTERNAL_BUILD_MANIFEST.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the external RPE1 design benchmark")
    parser.add_argument("--source", default=ec.PRIMARY_SOURCE)
    parser.add_argument("--output", required=True)
    record = run(parser.parse_args())
    print(f"{record['context']}: {record['source']['n_obs']} cells, {record['source']['n_vars']} genes")
    print(f"batches {record['batches']['total']} split {record['batches']['side_A']}/{record['batches']['side_B']}, "
          f"{record['batches']['control_cells']} control cells")
    print(f"identities {record['identities']['total']}, usable {record['identities']['usable']}, "
          f"eligible {record['query_gate']['eligible']} "
          f"({record['query_gate']['eligibility_fraction_of_usable']:.4f} of usable)")
    print(f"role split {record['role_split']['counts']}")
    print(f"quarters {record['four_way']['batches_per_quarter']}, "
          f"usable and four-way {record['four_way']['usable_and_four_way']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
