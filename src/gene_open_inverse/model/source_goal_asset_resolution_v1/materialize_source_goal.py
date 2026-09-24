#!/usr/bin/env python3
"""Gate 2: materialize quarantined paired source/goal assets after Gate 1 pass.

This is an asset-builder program, not model code.  It does not compute utility,
scores, rankings, or any CHECK metric.  It writes full endpoint assets only
when their all-row subtraction reproduces the pre-existing frozen response
cache at the declared tolerance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .audit_pairing_semantics import CONTROL, normalise_library_size_log1p, sha256


def _add_grouped(values: np.ndarray, group: np.ndarray, dest: np.ndarray, count: np.ndarray) -> None:
    """Accumulate rows by group, retaining the frozen cache builder's pattern."""
    if len(group) == 0:
        return
    order = np.argsort(group, kind="stable")
    sorted_group = group[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_group)) + 1]
    unique = sorted_group[starts]
    dest[unique] += np.add.reduceat(values[order], starts, axis=0)
    count[unique] += np.diff(np.r_[starts, len(order)])


def _source_weights(incidence: np.ndarray, sides: np.ndarray) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for view in (0, 1):
        weights = incidence.astype(np.float64)
        weights[:, sides != view] = 0.0
        denominator = weights.sum(axis=1, keepdims=True)
        if np.any(denominator <= 0.0):
            raise RuntimeError(f"an eligible identity has no perturbation cells in view {view}")
        result.append(weights / denominator)
    return result


def _write_row_metadata(
    path: Path,
    *,
    ids: np.ndarray,
    batches: np.ndarray,
    weights: list[np.ndarray],
    source_control_counts: np.ndarray,
    goal_counts: np.ndarray,
) -> None:
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for view, label in enumerate(("A", "B")):
        for index, identity in enumerate(ids.tolist()):
            nonzero = np.flatnonzero(weights[view][index] > 0.0)
            rows.append({
                "identity": identity,
                "identity_index": index,
                "view": label,
                "source_definition": "view-specific perturbation-cell-incidence weighted batch-control pseudobulk",
                "source_control_batch_ids": "|".join(batches[nonzero].tolist()),
                "source_control_weights": "|".join(f"{weights[view][index, batch]:.17g}" for batch in nonzero),
                "source_control_profile_sha256": hashlib.sha256(np.ascontiguousarray(weights[view][index]).tobytes()).hexdigest(),
                "source_control_cells_by_batch": "|".join(str(int(source_control_counts[batch])) for batch in nonzero),
                "goal_perturbation_cell_count": int(goal_counts[view, index]),
            })
    pd.DataFrame(rows).to_parquet(path, index=False)


def _checksum_lines(paths: list[Path]) -> str:
    return "".join(f"{sha256(path)}  {path.name}\n" for path in paths)


def main() -> int:
    import h5py

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--cache-ids", required=True, type=Path)
    parser.add_argument("--gate1", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--generation-code-commit", required=True)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing asset directory: {args.out}")
    gate1 = json.loads(args.gate1.read_text())
    if gate1.get("status") != "GATE_1_PASS_PAIRING_SEMANTICS_RESOLVED":
        raise RuntimeError("Gate 2 requires a passed Gate 1 pairing-semantics audit")
    if "`view_specific_batch_weighted`" not in gate1.get("gate_decision", ""):
        raise RuntimeError("Gate 1 did not establish the required view-specific source recipe")

    metadata = json.loads(args.metadata.read_text())
    raw_hash = sha256(args.raw)
    if raw_hash != metadata["source_sha256"] or raw_hash != metadata["verified_source_sha256"]:
        raise RuntimeError("raw source SHA-256 does not match frozen metadata")
    cache_hash = sha256(args.cache)
    cache = np.load(args.cache, mmap_mode="r", allow_pickle=False)
    ids = np.load(args.cache_ids, allow_pickle=False).astype(str)
    eligible = np.asarray(metadata["eligible_identities"], dtype=str)
    features = np.asarray(metadata["feature_identities"], dtype=str)
    batches = np.asarray(metadata["batch_identities"], dtype=str)
    sides = np.asarray(metadata["batch_sides"], dtype=np.int8)
    if cache.shape != (2, len(eligible), len(features)) or not np.array_equal(ids, eligible):
        raise RuntimeError("frozen response cache and metadata identity/feature axes disagree")

    args.out.mkdir(parents=True)
    source_partial = args.out / "x_source_ab.partial.npy"
    goal_partial = args.out / "x_goal_ab.partial.npy"
    source_out = args.out / "x_source_ab.npy"
    goal_out = args.out / "x_goal_ab.npy"
    source = np.lib.format.open_memmap(source_partial, mode="w+", dtype=np.float32, shape=cache.shape)
    goal = np.lib.format.open_memmap(goal_partial, mode="w+", dtype=np.float32, shape=cache.shape)

    with h5py.File(args.raw, "r") as raw_file:
        matrix = raw_file["X"]
        if not np.array_equal(raw_file["var"][raw_file["var"].attrs["_index"]][:].astype(str), features):
            raise RuntimeError("raw gene order does not exactly match frozen metadata")
        perturbation_group = raw_file["obs"]["perturbation"]
        perturbation_categories = perturbation_group["categories"][:].astype(str)
        perturbation_codes = perturbation_group["codes"][:]
        raw_batches = raw_file["obs"]["batch"][:].astype(str)
        batch_index = {value: index for index, value in enumerate(batches.tolist())}
        bidx = np.asarray([batch_index.get(value, -1) for value in raw_batches], dtype=np.int32)
        if np.any(bidx < 0):
            raise RuntimeError("raw observation has a batch outside frozen metadata")
        eligible_index = {value: index for index, value in enumerate(eligible.tolist())}
        category_to_eligible = np.asarray([eligible_index.get(value, -1) for value in perturbation_categories], dtype=np.int32)
        eidx = np.full(len(perturbation_codes), -1, dtype=np.int32)
        valid_code = perturbation_codes >= 0
        eidx[valid_code] = category_to_eligible[perturbation_codes[valid_code]]
        control_code = int(np.flatnonzero(perturbation_categories == CONTROL)[0])

        control_sums = np.zeros((len(batches), len(features)), dtype=np.float32)
        control_counts = np.zeros(len(batches), dtype=np.int64)
        goal_sums = np.zeros(cache.shape, dtype=np.float32)
        goal_counts = np.zeros((2, len(eligible)), dtype=np.int64)
        incidence = np.zeros((len(eligible), len(batches)), dtype=np.int32)
        eligible_rows = eidx >= 0
        np.add.at(incidence, (eidx[eligible_rows], bidx[eligible_rows]), 1)
        if np.any(incidence.sum(axis=1) == 0):
            raise RuntimeError("an eligible identity has no raw perturbation cells")

        for start in range(0, matrix.shape[0], 2048):
            end = min(start + 2048, matrix.shape[0])
            values, valid = normalise_library_size_log1p(np.asarray(matrix[start:end], dtype=np.float32))
            local_codes = perturbation_codes[start:end]
            local_eligible = eidx[start:end]
            local_batches = bidx[start:end]
            local_views = sides[local_batches]
            control_take = valid & (local_codes == control_code)
            if np.any(control_take):
                _add_grouped(values[control_take], local_batches[control_take], control_sums, control_counts)
            for view in (0, 1):
                take = valid & (local_eligible >= 0) & (local_views == view)
                if np.any(take):
                    _add_grouped(values[take], local_eligible[take], goal_sums[view], goal_counts[view])
            if start % (2048 * 128) == 0:
                print(f"gate2_raw_scan_rows={end}/{matrix.shape[0]}", flush=True)

    if np.any(control_counts == 0) or np.any(goal_counts == 0):
        raise RuntimeError("missing finite control or perturbation rows prevents endpoint materialization")
    controls = control_sums.astype(np.float64) / control_counts[:, None]
    weights = _source_weights(incidence, sides)
    delta_sum = 0.0
    element_count = 0
    max_error = 0.0
    close_count = 0
    for start in range(0, len(eligible), 128):
        end = min(start + 128, len(eligible))
        for view in (0, 1):
            source_block = (weights[view][start:end] @ controls).astype(np.float32)
            goal_block = (goal_sums[view, start:end].astype(np.float64) / goal_counts[view, start:end, None]).astype(np.float32)
            source[view, start:end] = source_block
            goal[view, start:end] = goal_block
            delta_block = goal_block - source_block
            cache_block = np.asarray(cache[view, start:end], dtype=np.float32)
            error = np.abs(delta_block.astype(np.float64) - cache_block.astype(np.float64))
            max_error = max(max_error, float(error.max()))
            delta_sum += float(error.sum(dtype=np.float64))
            element_count += int(error.size)
            close_count += int(np.isclose(delta_block, cache_block, atol=args.atol, rtol=args.rtol).sum())
    source.flush()
    goal.flush()
    del source, goal
    mean_error = delta_sum / element_count
    fraction_close = close_count / element_count
    if fraction_close != 1.0:
        raise RuntimeError(
            f"full endpoint invariant failed: max_abs_error={max_error:.8g}, mean_abs_error={mean_error:.8g}, "
            f"fraction_close={fraction_close:.12f}"
        )
    source_partial.rename(source_out)
    goal_partial.rename(goal_out)
    gene_order = args.out / "gene_order.txt"
    gene_order.write_text("\n".join(features.tolist()) + "\n")
    row_metadata = args.out / "source_goal_row_metadata.parquet"
    _write_row_metadata(
        row_metadata,
        ids=ids,
        batches=batches,
        weights=weights,
        source_control_counts=control_counts,
        goal_counts=goal_counts,
    )
    manifest = {
        "schema": "SOURCE_GOAL_ASSET_RESOLUTION_V1_GATE_2",
        "status": "FROZEN_SOURCE_GOAL_ASSETS_COMPLETE",
        "raw_source": {"path": str(args.raw.resolve()), "sha256": raw_hash, "shape": [int(len(eidx)), int(len(features))]},
        "frozen_delta_cache": {"path": str(args.cache.resolve()), "sha256": cache_hash, "shape": list(cache.shape),
                                "cache_ids_path": str(args.cache_ids.resolve()), "cache_ids_sha256": sha256(args.cache_ids)},
        "preprocessing": "per-cell library-size normalization to 10000; log1p; valid cells only; float32 accumulation and endpoint storage",
        "aggregation": {
            "goal": "mean normalized perturbation expression, separately in A/B batch-disjoint views",
            "source": "view-specific perturbation-cell-incidence weighted mean of batch-specific normalized control pseudobulks",
            "source_varies_by_query": True,
            "source_control_profile_count_per_view": int(len(eligible)),
        },
        "gene_order": {"path": str(gene_order.resolve()), "count": int(len(features)), "exact_raw_metadata_match": True},
        "arrays": {
            "x_source_ab": {"path": str(source_out.resolve()), "shape": list(cache.shape), "dtype": "float32"},
            "x_goal_ab": {"path": str(goal_out.resolve()), "shape": list(cache.shape), "dtype": "float32"},
            "row_metadata": {"path": str(row_metadata.resolve()), "rows": int(2 * len(ids))},
        },
        "delta_invariant": {"expression": "x_goal_ab - x_source_ab == frozen response cache", "atol": args.atol, "rtol": args.rtol,
                            "max_abs_error": max_error, "mean_abs_error": mean_error, "fraction_close": fraction_close,
                            "all_elements_close": True},
        "gate1": {"path": str(args.gate1.resolve()), "sha256": sha256(args.gate1), "status": gate1["status"]},
        "generation": {"code_commit": args.generation_code_commit, "code_path": str(Path(__file__).resolve()), "code_sha256": sha256(Path(__file__))},
        "access_firewall": {
            "builder_scope": "raw endpoint asset generation only; no utility, model fitting, scoring, ranking, or CHECK evaluation",
            "model_scope": "model code may materialize G_fit/G_select only; G_check endpoint rows remain inaccessible until a separately authorized final evaluation",
        },
    }
    manifest_path = args.out / "source_goal_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    checksums = args.out / "SHA256SUMS"
    checksums.write_text(_checksum_lines([source_out, goal_out, row_metadata, gene_order, manifest_path]))
    print(f"gate2_complete={args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
