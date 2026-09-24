#!/usr/bin/env python3
"""Gate 1: audit source/goal pairing semantics without materializing endpoints.

This program reads raw expression only to recover batch-specific *control*
means and deterministic sampled perturbation goals.  It never computes utility,
loads a model checkpoint, or ranks candidates.  Its sampled cache comparison is
an asset-recipe check, not an evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


CONTROL = "control"
EPS = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalise_library_size_log1p(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exactly reproduce the frozen cache's per-cell normalization primitive."""
    values = np.asarray(values, dtype=np.float32)
    totals = values.sum(axis=1)
    valid = np.isfinite(totals) & (totals > 0.0)
    values[valid] = np.log1p(values[valid] / totals[valid, None] * 1e4)
    values[~valid] = np.nan
    return values, valid


def _hash_rows(values: np.ndarray) -> list[str]:
    contiguous = np.ascontiguousarray(values)
    return [hashlib.sha256(row.tobytes()).hexdigest() for row in contiguous]


def _source_summary(
    weights: np.ndarray,
    control_gram: np.ndarray,
    control_rank: int,
    feature_count: int,
    *,
    pair_seed: int,
    pair_count: int,
) -> dict[str, Any]:
    """Summarize sources through the control Gram matrix, without endpoints.

    For source ``S=W C``, all distances and total variance are determined by
    ``G=C C^T``.  If C has full row rank, distinct rows of W are exactly
    distinct (real-valued) source profiles, so no [9205, 8248] endpoint needs
    to be materialized merely to audit diversity.
    """
    n, _ = weights.shape
    profile_counts: Counter[str] = Counter()
    profile_counts.update(_hash_rows(weights))
    centered = weights - weights.mean(axis=0, keepdims=True)
    total_variance = float(np.einsum("ni,ij,nj->", centered, control_gram, centered, optimize=True) / n)
    rng = np.random.Generator(np.random.PCG64(pair_seed))
    left = rng.integers(0, n, size=pair_count)
    right = rng.integers(0, n, size=pair_count)
    nonself = left != right
    left, right = left[nonself], right[nonself]
    distances: list[np.ndarray] = []
    for start in range(0, len(left), 256):
        end = min(len(left), start + 256)
        delta = weights[left[start:end]] - weights[right[start:end]]
        distances.append(np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", delta, control_gram, delta, optimize=True), 0.0)))
    distance = np.concatenate(distances) if distances else np.empty(0, dtype=np.float64)
    return {
        "queries": int(n),
        "distinct_control_mixture_profiles": int(len(profile_counts)),
        "distinct_control_mixture_profile_fraction": float(len(profile_counts) / n),
        "max_perturbations_per_control_mixture": int(max(profile_counts.values())),
        "control_mean_matrix_rank": int(control_rank),
        "source_profile_identity_proven": bool(control_rank == weights.shape[1]),
        "distinct_source_expression_profiles_float32": int(len(profile_counts)) if control_rank == weights.shape[1] else None,
        "distinct_source_expression_profile_fraction": float(len(profile_counts) / n) if control_rank == weights.shape[1] else None,
        "coordinate_variance_mean": float(total_variance / feature_count),
        "sample_pair_count": int(len(distance)),
        "sample_pair_mean_l2_distance": float(distance.mean()),
        "sample_pair_median_l2_distance": float(np.median(distance)),
        "sample_pair_max_l2_distance": float(distance.max()),
    }


def _cache_comparison(
    goals: np.ndarray,
    sources: list[np.ndarray],
    cache: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, source in zip(("all_batch_weighted", "view_specific_batch_weighted"), sources):
        directions: dict[str, Any] = {}
        for view, label in enumerate(("A", "B")):
            delta = goals[view] - source[view]
            reference = cache[view].astype(np.float64, copy=False)
            error = np.abs(delta - reference)
            close = np.isclose(delta, reference, atol=atol, rtol=rtol)
            directions[label] = {
                "max_abs_error": float(error.max()),
                "mean_abs_error": float(error.mean()),
                "fraction_close": float(close.mean()),
                "all_close": bool(close.all()),
            }
        result[name] = directions
    return result


def _format_markdown(report: dict[str, Any]) -> str:
    source = report["source_diversity"]
    lines = [
        "# SOURCE_GOAL_ASSET_RESOLUTION_V1 — Gate 1 pairing-semantics audit",
        "",
        f"**Status:** `{report['status']}`  ",
        "**Scope:** raw-data semantic audit only; no utility, model, ranking, or G_CHECK response evaluation was run.",
        "",
        "## Raw and frozen contract",
        "",
        f"- Raw source: `{report['raw']['path']}`",
        f"- Raw SHA-256 verified: `{report['raw']['sha256']}`",
        f"- Frozen response cache: `{report['frozen_cache']['path']}`",
        f"- Frozen cache SHA-256 verified: `{report['frozen_cache']['sha256']}`",
        f"- Shape: raw `{report['raw']['shape']}`, cache `{report['frozen_cache']['shape']}`.",
        f"- Gene order: exact match to the frozen metadata feature order = `{report['raw']['gene_order_exact_match']}`.",
        "",
        "## Pairing semantics",
        "",
        report["pairing_statement"],
        "",
        "The audit independently tests two plausible source constructions against deterministic FIT/SELECT-only sampled rows from the frozen delta cache. Gate 2 is allowed only when one construction passes every sampled element at the declared tolerance; full-corpus delta consistency remains Gate 2's invariant.",
        "",
        "| Source construction | View | max abs error | mean abs error | fraction close | all close |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for construction, directions in report["sampled_cache_recipe_comparison"].items():
        for direction, values in directions.items():
            lines.append(
                f"| {construction} | {direction} | {values['max_abs_error']:.8g} | "
                f"{values['mean_abs_error']:.8g} | {values['fraction_close']:.8f} | {values['all_close']} |"
            )
    lines += [
        "",
        "## Source diversity",
        "",
        "| Construction / view | Distinct source profiles / queries | Mean coordinate variance | Mean sampled L2 distance | Median sampled L2 distance | Max perturbations per source mixture |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in source.items():
        distinct = values["distinct_source_expression_profiles_float32"]
        fraction = values["distinct_source_expression_profile_fraction"]
        profile_text = "unresolved" if distinct is None else f"{distinct}/{values['queries']} ({fraction:.4f})"
        lines.append(
            f"| {name} | {profile_text} | "
            f"{values['coordinate_variance_mean']:.8g} | {values['sample_pair_mean_l2_distance']:.8g} | "
            f"{values['sample_pair_median_l2_distance']:.8g} | {values['max_perturbations_per_control_mixture']} |"
        )
    lines += [
        "",
        "## Gate decision",
        "",
        report["gate_decision"],
        "",
        "No endpoint arrays were written by Gate 1. If Gate 2 is authorized, it must reproduce the passing construction over all 9,205 identities, write quarantined assets and metadata, and report full-corpus `max_abs_error`, `mean_abs_error`, and `fraction_close` before any model code receives FIT/SELECT arrays.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    import h5py

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--cache-ids", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path,
                        help="Used only to choose a deterministic FIT/SELECT sample; no G_CHECK array is materialized.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--pair-count", type=int, default=10_000)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite existing audit directory: {args.out}")
    if args.sample_count < 2 or args.pair_count < 2:
        raise ValueError("sample-count and pair-count must each be at least 2")

    metadata = json.loads(args.metadata.read_text())
    freeze = json.loads(args.freeze.read_text())
    raw_hash = sha256(args.raw)
    if raw_hash != metadata["source_sha256"] or raw_hash != metadata["verified_source_sha256"]:
        raise RuntimeError("raw source SHA-256 does not match frozen metadata")
    cache_hash = sha256(args.cache)
    cache = np.load(args.cache, mmap_mode="r", allow_pickle=False)
    cache_ids = np.load(args.cache_ids, allow_pickle=False).astype(str)
    eligible = np.asarray(metadata["eligible_identities"], dtype=str)
    feature_ids = np.asarray(metadata["feature_identities"], dtype=str)
    batches = np.asarray(metadata["batch_identities"], dtype=str)
    sides = np.asarray(metadata["batch_sides"], dtype=np.int8)
    if cache.shape != (2, len(eligible), len(feature_ids)):
        raise RuntimeError(f"unexpected frozen cache shape {cache.shape}")
    if not np.array_equal(cache_ids, eligible):
        raise RuntimeError("cache identity axis does not match frozen metadata")
    if set(freeze["roles"]) < {"G_fit", "G_select"}:
        raise RuntimeError("freeze lacks G_fit/G_select roles")

    # Use only FIT/SELECT identities for the sampled endpoint/cache comparison.
    fit_select = np.asarray(freeze["roles"]["G_fit"] + freeze["roles"]["G_select"], dtype=str)
    index = {identity: i for i, identity in enumerate(eligible.tolist())}
    sample_ids = np.asarray(sorted(fit_select.tolist(), key=lambda x: hashlib.sha256(
        f"SOURCE_GOAL_ASSET_RESOLUTION_V1|20260916|{x}".encode()).hexdigest())[:args.sample_count], dtype=str)
    sample_global = np.asarray([index[x] for x in sample_ids], dtype=np.int64)
    sample_lookup = {identity: i for i, identity in enumerate(sample_ids.tolist())}

    with h5py.File(args.raw, "r") as raw_file:
        matrix = raw_file["X"]
        raw_shape = tuple(int(x) for x in matrix.shape)
        raw_genes = raw_file["var"][raw_file["var"].attrs["_index"]][:].astype(str)
        if not np.array_equal(raw_genes, feature_ids):
            raise RuntimeError("raw gene order does not exactly match frozen metadata")
        perturbation_group = raw_file["obs"]["perturbation"]
        perturbation_categories = perturbation_group["categories"][:].astype(str)
        perturbation_codes = perturbation_group["codes"][:]
        batch = raw_file["obs"]["batch"][:].astype(str)
        batch_index = {value: i for i, value in enumerate(batches.tolist())}
        bidx = np.asarray([batch_index.get(value, -1) for value in batch], dtype=np.int32)
        if np.any(bidx < 0):
            raise RuntimeError("raw observation has a batch absent from frozen metadata")
        eligible_index = {value: i for i, value in enumerate(eligible.tolist())}
        category_to_eligible = np.asarray([eligible_index.get(value, -1) for value in perturbation_categories], dtype=np.int32)
        eidx = np.full(len(perturbation_codes), -1, dtype=np.int32)
        valid_code = perturbation_codes >= 0
        eidx[valid_code] = category_to_eligible[perturbation_codes[valid_code]]
        all_noncontrol = set(np.asarray(metadata["all_noncontrol_identities"], dtype=str).tolist())
        if not set(eligible.tolist()).issubset(all_noncontrol):
            raise RuntimeError("eligible identities are not contained in frozen non-control identities")

        # Derive the control and sampled-goal primitives in one sequential raw
        # scan.  H5AD point indexing over sparse sampled rows repeatedly reads
        # compressed chunks; sequential chunks are both deterministic and avoid
        # accidentally turning this Gate 1 audit into an uncontrolled I/O job.
        controls = np.zeros((len(batches), len(feature_ids)), dtype=np.float32)
        control_counts = np.zeros(len(batches), dtype=np.int64)
        sampled_sums = np.zeros((2, len(sample_ids), len(feature_ids)), dtype=np.float32)
        sampled_counts = np.zeros((2, len(sample_ids)), dtype=np.int64)
        sampled_global_lookup = np.full(len(eligible), -1, dtype=np.int32)
        sampled_global_lookup[sample_global] = np.arange(len(sample_ids), dtype=np.int32)
        control_code = int(np.flatnonzero(perturbation_categories == CONTROL)[0])
        incidence = np.zeros((len(eligible), len(batches)), dtype=np.int32)
        eligible_rows = eidx >= 0
        np.add.at(incidence, (eidx[eligible_rows], bidx[eligible_rows]), 1)
        if np.any(incidence.sum(axis=1) == 0):
            raise RuntimeError("an eligible identity has no raw perturbation cells")
        for start in range(0, matrix.shape[0], 2048):
            end = min(start + 2048, matrix.shape[0])
            local_codes = perturbation_codes[start:end]
            local_eligible = eidx[start:end]
            local_batch = bidx[start:end]
            local_sample = sampled_global_lookup[np.maximum(local_eligible, 0)]
            take_control = local_codes == control_code
            take_sample = local_sample >= 0
            take = take_control | take_sample
            if not np.any(take):
                continue
            local_rows = np.flatnonzero(take)
            values, valid = normalise_library_size_log1p(np.asarray(matrix[start:end], dtype=np.float32)[local_rows])
            selected_control = take_control[local_rows] & valid
            if np.any(selected_control):
                np.add.at(controls, local_batch[local_rows[selected_control]], values[selected_control])
                np.add.at(control_counts, local_batch[local_rows[selected_control]], 1)
            selected_sample = take_sample[local_rows] & valid
            if np.any(selected_sample):
                sample_pos = local_sample[local_rows[selected_sample]]
                view = sides[local_batch[local_rows[selected_sample]]]
                for direction in (0, 1):
                    direction_take = view == direction
                    if np.any(direction_take):
                        np.add.at(sampled_sums[direction], sample_pos[direction_take], values[selected_sample][direction_take])
                        np.add.at(sampled_counts[direction], sample_pos[direction_take], 1)
            if start % (2048 * 128) == 0:
                print(f"gate1_raw_scan_rows={end}/{matrix.shape[0]}", flush=True)
        if np.any(control_counts == 0):
            raise RuntimeError("a batch has no finite control rows")
        controls = controls / control_counts[:, None]
        all_weights = incidence.astype(np.float64)
        all_weights /= all_weights.sum(axis=1, keepdims=True)
        view_weights: list[np.ndarray] = []
        for view in (0, 1):
            weight = incidence.astype(np.float64)
            weight[:, sides != view] = 0.0
            denominator = weight.sum(axis=1, keepdims=True)
            if np.any(denominator <= 0.0):
                raise RuntimeError(f"an eligible identity has no view-{view} perturbation cells")
            view_weights.append(weight / denominator)
        if np.any(sampled_counts == 0):
            raise RuntimeError("a sampled FIT/SELECT identity lacks a valid goal in one view")
        sampled_goals = sampled_sums.astype(np.float64) / sampled_counts[:, :, None]

    sampled_sources_all = [all_weights[sample_global] @ controls for _ in (0, 1)]
    sampled_sources_view = [view_weights[view][sample_global] @ controls for view in (0, 1)]
    comparison = _cache_comparison(
        sampled_goals,
        [np.stack(sampled_sources_all), np.stack(sampled_sources_view)],
        np.asarray(cache[:, sample_global, :], dtype=np.float32),
        atol=args.atol,
        rtol=args.rtol,
    )
    passing = [
        name for name, directions in comparison.items()
        if all(values["all_close"] for values in directions.values())
    ]
    control_gram = controls @ controls.T
    control_rank = int(np.linalg.matrix_rank(control_gram))
    source_diversity = {
        "all_batch_weighted_shared_across_views": _source_summary(all_weights, control_gram, control_rank, len(feature_ids), pair_seed=20260916, pair_count=args.pair_count),
        "view_A_batch_weighted": _source_summary(view_weights[0], control_gram, control_rank, len(feature_ids), pair_seed=20260916, pair_count=args.pair_count),
        "view_B_batch_weighted": _source_summary(view_weights[1], control_gram, control_rank, len(feature_ids), pair_seed=20260917, pair_count=args.pair_count),
    }
    if len(passing) == 1:
        status = "GATE_1_PASS_PAIRING_SEMANTICS_RESOLVED"
        decision = (
            f"PASS: `{passing[0]}` reproduces every sampled FIT/SELECT delta element at atol={args.atol:g}, "
            f"rtol={args.rtol:g}. Gate 2 may materialize quarantined endpoint assets using this exact recipe, "
            "then must validate the invariant on all identities before release."
        )
    elif not passing:
        status = "GATE_1_BLOCKED_NO_CANDIDATE_SOURCE_RECIPE_MATCHED"
        decision = (
            "BLOCKED: neither audited source construction reproduces the sampled frozen cache. Do not materialize "
            "endpoints or alter preprocessing; first recover the exact historical cache-generation implementation."
        )
    else:
        status = "GATE_1_BLOCKED_AMBIGUOUS_SOURCE_RECIPE"
        decision = (
            "BLOCKED: more than one source construction reproduces the sampled cache. Do not materialize endpoints "
            "until an authoritative cache-generation record disambiguates the source definition."
        )
    report = {
        "schema": "SOURCE_GOAL_ASSET_RESOLUTION_V1_GATE_1",
        "status": status,
        "raw": {"path": str(args.raw.resolve()), "sha256": raw_hash, "shape": list(raw_shape),
                "gene_order_exact_match": True, "aggregation": "per-cell library-size 10000 then log1p; mean pseudobulk"},
        "frozen_cache": {"path": str(args.cache.resolve()), "sha256": cache_hash, "shape": list(cache.shape),
                         "identity_axis_sha256": hashlib.sha256("\n".join(cache_ids.tolist()).encode()).hexdigest()},
        "pairing_statement": (
            "Each endpoint candidate has a deterministic perturbation pseudobulk goal in each A/B batch-disjoint view. "
            "The source candidate is a deterministic weighted mixture of batch-specific control pseudobulks; this audit "
            "tests whether its weights are pooled across views or restricted to the endpoint view."
        ),
        "sample": {"identities": sample_ids.tolist(), "count": int(len(sample_ids)), "roles": ["G_fit", "G_select"],
                   "selection": "SHA-256 deterministic identity order with salt SOURCE_GOAL_ASSET_RESOLUTION_V1|20260916"},
        "tolerance": {"atol": args.atol, "rtol": args.rtol},
        "sampled_cache_recipe_comparison": comparison,
        "source_diversity": source_diversity,
        "gate_decision": decision,
    }
    args.out.mkdir(parents=True)
    (args.out / "pairing_semantics_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.out / "PAIRING_SEMANTICS_AUDIT.md").write_text(_format_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
