#!/usr/bin/env python3
"""Phase 2: reproduce the frozen measurement contract in a new context.

This is the anchor's recipe applied to a different dataset, not a recipe chosen for
that dataset: library-size 1e4 normalization, log1p, per-batch control means, a
per-identity control reference weighted by that identity's cell incidence inside the
view's own batches, the same A/B batch hash with the same namespace and seed, the
same quarter refinement, the same minimum cell threshold and the same
``QUERY_ELIGIBILITY_V1`` gate.  Nothing in it is retuned on the new data.

It trains nothing and ranks nothing.
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
from . import contract as fc
from . import dualguide as dg


CHUNK = 16384


def view_side(batch: str) -> int:
    """The anchor's frozen A/B bit, applied to this context's batch names."""
    token = f"{fc.VIEW_NAMESPACE}|{fc.VIEW_SEED}|{batch}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big") & 1


def quarter_of(batch: str) -> int:
    """The anchor's four-way refinement, so an action can be estimated cross-fitted."""
    token = f"{fc.QUAD_NAMESPACE}|{fc.VIEW_SEED}|{batch}".encode("utf-8")
    refinement = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") & 1
    return view_side(batch) * 2 + refinement


def accumulate(handle, row: np.ndarray, column: np.ndarray, sides: np.ndarray,
               quarters: np.ndarray, identities: int, batches: int, features: int,
               cell_rows: np.ndarray, gene_keep: np.ndarray) -> dict:
    """One streaming pass: per-side, per-quarter and per-batch-control sums.

    ``cell_rows`` maps each retained cell back to its row in the source file, so the
    stream skips excluded cells instead of assigning them to an identity.
    ``gene_keep`` drops ambiguous gene columns.  The library size is computed on the
    full row before that restriction, because the normalization denominator is a
    property of the cell and not of whichever genes this programme happens to keep.
    """
    sums = np.zeros((2, identities, features), dtype=np.float64)
    counts = np.zeros((2, identities), dtype=np.int64)
    quarter_sums = np.zeros((4, identities, features), dtype=np.float64)
    quarter_counts = np.zeros((4, identities), dtype=np.int64)
    control_sums = np.zeros((batches, features), dtype=np.float64)
    control_counts = np.zeros(batches, dtype=np.int64)
    bad = 0
    observations = int(cell_rows.size)
    contiguous = bool(cell_rows.size == int(handle.n_obs))
    for start in range(0, observations, CHUNK):
        stop = min(start + CHUNK, observations)
        block = handle.X[start:stop] if contiguous else handle.X[cell_rows[start:stop]]
        values = np.asarray(block.toarray() if hasattr(block, "toarray") else block, dtype=np.float32)
        library = values.sum(axis=1)
        good = np.isfinite(library) & (library > 0.0)
        bad += int((~good).sum())
        values[good] = np.log1p(values[good] / library[good, None] * fc.LIBRARY_TARGET)
        values[~good] = 0.0
        values = np.ascontiguousarray(values[:, gene_keep])
        local_row, local_column = row[start:stop], column[start:stop]
        local_side, local_quarter = sides[local_column], quarters[local_column]
        for quarter in range(4):
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
    return {"sums": sums, "counts": counts, "quarter_sums": quarter_sums,
            "quarter_counts": quarter_counts, "control_sums": control_sums,
            "control_counts": control_counts, "bad": bad}


def _weighted_reference(incidence: np.ndarray, keep_columns: np.ndarray,
                        control_mean: np.ndarray) -> np.ndarray:
    weights = incidence.astype(np.float64).copy()
    weights[:, ~keep_columns] = 0.0
    total = weights.sum(axis=1, keepdims=True)
    safe = total[:, 0] > 0
    weights[safe] /= total[safe]
    return weights @ control_mean


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    handle = anndata.read_h5ad(args.source, backed="r")
    # The collapsed perturbation column is optional under dual_guide: the identity is
    # a property of the pair, and some releases ship only the pair.
    perturbation = (handle.obs[args.perturbation_column].astype(str).to_numpy()
                    if args.perturbation_column and args.perturbation_column in handle.obs
                    else np.full(int(handle.n_obs), "__UNDECLARED__", dtype=object).astype(str))
    batch = handle.obs[args.batch_column].astype(str).to_numpy()
    # The gene axis is whichever column the audit established carries symbols.  It is
    # passed in explicitly rather than assumed, because a release keyed by ENSEMBL
    # identifier would otherwise produce an axis that intersects nothing.
    if args.gene_symbol_column:
        raw_genes = handle.var[args.gene_symbol_column].astype(str).to_numpy()
    else:
        raw_genes = np.asarray(handle.var_names, dtype=str)
    # A duplicated symbol is dropped entirely rather than deduplicated.  Downstream the
    # axis is addressed by name, so a repeated symbol means one column silently wins and
    # the other disappears, and there is no way to tell which is the intended gene.
    # Dropping both is the only choice that does not quietly pick one.
    symbols, counts = np.unique(raw_genes, return_counts=True)
    duplicated = set(symbols[counts > 1].tolist())
    gene_keep = np.asarray([name not in duplicated for name in raw_genes.tolist()], dtype=bool)
    genes = raw_genes[gene_keep]
    dropped_genes = {"rule": "symbols appearing more than once are dropped, both copies",
                     "duplicated_symbols": int(len(duplicated)),
                     "columns_dropped": int((~gene_keep).sum()),
                     "examples": sorted(duplicated)[:10]}
    features = int(gene_keep.sum())
    handle_rows = int(handle.n_obs)

    control_labels = tuple(args.control_labels)
    excluded: dict = {}
    if args.identity_mode == "dual_guide":
        # The identity of a dual-guide cell is a property of the pair.  It is resolved
        # here by the same module the audit reported from, so what the builder uses is
        # what the audit recorded, and cross-gene or unparsed pairs leave the
        # single-gene programme entirely rather than being folded into one of its genes.
        if not args.guide_column:
            raise RuntimeError("identity mode dual_guide requires --guide-column")
        guides = handle.obs[args.guide_column].astype(str).to_numpy()
        universe = set(genes.tolist()) | set(np.unique(perturbation).tolist())
        resolution = dg.resolve(guides, universe)
        kind = resolution["kind"]
        control_mask = kind == dg.NON_TARGETING
        keep_cell = (kind == dg.SINGLE_GENE) | control_mask
        excluded = {"rule": "cross-gene and unparsed dual-guide pairs are excluded from "
                            "the single-gene programme and counted here",
                    "cross_gene_cells": int((kind == dg.CROSS_GENE).sum()),
                    "unparsed_cells": int((kind == dg.UNPARSED).sum()),
                    "excluded_fraction": float((~keep_cell).mean()),
                    "guide_column": args.guide_column}
        perturbation = np.where(control_mask, "__CONTROL__", resolution["identity"])
        control_labels = ("__CONTROL__",)
        perturbation = perturbation[keep_cell]
        batch = batch[keep_cell]
        cell_rows = np.flatnonzero(keep_cell)
    else:
        if not args.perturbation_column:
            raise RuntimeError("identity mode column requires --perturbation-column")
        control_mask = np.isin(perturbation, np.asarray(control_labels, dtype=str))
        cell_rows = np.arange(perturbation.size, dtype=np.int64)
        excluded = {"rule": "identity taken directly from the declared column",
                    "excluded_fraction": 0.0}
    control_mask = np.isin(perturbation, np.asarray(control_labels, dtype=str))
    if not control_mask.any():
        raise RuntimeError(f"no cell carries a declared control label {control_labels}")

    batches = np.unique(batch)
    sides = np.asarray([view_side(str(name)) for name in batches.tolist()], dtype=np.int8)
    quarters = np.asarray([quarter_of(str(name)) for name in batches.tolist()], dtype=np.int8)
    if not np.array_equal(quarters // 2, sides):
        raise RuntimeError("the quarter assignment is not a refinement of the A/B split")
    batch_position = {name: index for index, name in enumerate(batches.tolist())}
    column = np.asarray([batch_position[name] for name in batch.tolist()], dtype=np.int64)

    identities = np.asarray(sorted(set(perturbation[~control_mask].tolist())), dtype=str)
    identity_position = {name: index for index, name in enumerate(identities.tolist())}
    row = np.asarray([-1 if flag else identity_position[name]
                      for name, flag in zip(perturbation.tolist(), control_mask.tolist())],
                     dtype=np.int64)

    incidence = np.zeros((identities.size, batches.size), dtype=np.int64)
    keep = row >= 0
    np.add.at(incidence, (row[keep], column[keep]), 1)
    control_per_batch = np.zeros(batches.size, dtype=np.int64)
    np.add.at(control_per_batch, column[~keep], 1)
    empty = np.flatnonzero(control_per_batch == 0)
    if empty.size:
        raise RuntimeError(f"{empty.size} batches carry no control cell; "
                           "the per-batch reference is undefined and the contract forbids pooling")

    cells_by_side = np.stack([incidence[:, sides == side].sum(axis=1) for side in (0, 1)], axis=1)
    eligible_depth = (cells_by_side >= fc.MIN_VIEW_CELLS).all(axis=1)

    store = accumulate(handle, row, column, sides, quarters,
                       identities.size, batches.size, features, cell_rows, gene_keep)
    handle.file.close()

    control_mean = store["control_sums"] / np.maximum(store["control_counts"], 1)[:, None]
    responses = np.zeros((2, identities.size, features), dtype=np.float32)
    sources = np.zeros((2, identities.size, features), dtype=np.float32)
    usable = eligible_depth & (store["counts"] > 0).all(axis=0)
    for side in (0, 1):
        mean = np.zeros_like(store["sums"][side])
        rows = store["counts"][side] > 0
        mean[rows] = store["sums"][side][rows] / store["counts"][side][rows][:, None]
        reference = _weighted_reference(incidence, sides == side, control_mean)
        sources[side] = reference.astype(np.float32)
        responses[side] = (mean - reference).astype(np.float32)
    responses[:, ~usable] = 0.0
    sources[:, ~usable] = 0.0

    by_quarter = np.stack([incidence[:, quarters == q].sum(axis=1) for q in range(4)], axis=1)
    four_way = (by_quarter >= fc.MIN_VIEW_CELLS).all(axis=1)
    quarter_responses = np.zeros((4, identities.size, features), dtype=np.float32)
    for quarter in range(4):
        mean = np.zeros_like(store["quarter_sums"][quarter])
        rows = store["quarter_counts"][quarter] > 0
        mean[rows] = store["quarter_sums"][quarter][rows] / store["quarter_counts"][quarter][rows][:, None]
        reference = _weighted_reference(incidence, quarters == quarter, control_mean)
        quarter_responses[quarter] = (mean - reference).astype(np.float32)
    quarter_responses[:, ~(usable & four_way)] = 0.0

    gram = gt.cross_view_gram(responses[0][usable].astype(np.float64),
                              responses[1][usable].astype(np.float64))
    gate = gt.eligibility(gram, fc.ELIGIBILITY_FDR)
    eligible = np.zeros(identities.size, dtype=bool)
    eligible[np.flatnonzero(usable)] = gate.eligible
    record = {
        "schema": "VCDESIGN_FOUR_CONTEXT_V1_BUILD",
        "context": args.context, "perturbation_type": fc.PERTURBATION_TYPE,
        "inherited_from": fc.INHERITED_FROM, "trained_anything": False,
        "source": {"path": str(args.source), "n_obs_total": int(handle_rows),
                   "n_obs_retained": int(row.size), "n_vars": features,
                   "perturbation_column": args.perturbation_column,
                   "batch_column": args.batch_column, "identity_mode": args.identity_mode,
                   "gene_symbol_column": args.gene_symbol_column,
                   "n_vars_in_file": int(handle.n_vars),
                   "control_labels": list(control_labels)},
        "dual_guide_exclusion": excluded,
        "gene_axis": dropped_genes,
        "batches": {"total": int(batches.size), "side_A": int((sides == 0).sum()),
                    "side_B": int((sides == 1).sum()),
                    "batches_per_quarter": [int((quarters == q).sum()) for q in range(4)],
                    "control_cells": int(store["control_counts"].sum()),
                    "batches_without_controls": 0,
                    "control_cells_per_batch_median": float(np.median(control_per_batch))},
        "identities": {"nominal": int(identities.size),
                       "depth_eligible_both_views": int(eligible_depth.sum()),
                       "usable": int(usable.sum()), "min_view_cells": fc.MIN_VIEW_CELLS,
                       "cells_per_identity_median": float(np.median(cells_by_side.sum(axis=1)))},
        "query_gate": {"rule": "QUERY_ELIGIBILITY_V1 applied unchanged",
                       "fdr": fc.ELIGIBILITY_FDR, "eligible": int(eligible.sum()),
                       "eligibility_fraction_of_usable": float(gate.fraction)},
        "training_and_query_rule": {"train": fc.TRAIN_RULE.get(args.context),
                                    "query": fc.QUERY_RULE.get(args.context),
                                    "candidates": fc.CANDIDATE_RULE},
        "four_way": {"identities_four_way_eligible": int(four_way.sum()),
                     "usable_and_four_way": int((usable & four_way).sum())},
        "bad_library_cells_excluded": int(store["bad"]),
        "runtime": {"timestamp": datetime.now(timezone.utc).isoformat(),
                    "python": platform.python_version(), "numpy": np.__version__},
    }
    np.save(output / "side_responses.npy", responses)
    np.save(output / "side_sources.npy", sources)
    np.save(output / "control_mean.npy", control_mean.astype(np.float32))
    np.save(output / "quarter_responses.npy", quarter_responses)
    np.save(output / "batch_quarter.npy", quarters)
    np.save(output / "batch_side.npy", sides)
    np.save(output / "batch_axis.npy", batches)
    np.save(output / "four_way_eligible.npy", four_way)
    np.save(output / "identity_axis.npy", identities)
    np.save(output / "gene_axis.npy", genes)
    np.save(output / "incidence.npy", incidence)
    np.save(output / "cells_by_side.npy", cells_by_side)
    np.save(output / "control_per_batch.npy", control_per_batch)
    np.save(output / "usable.npy", usable)
    np.save(output / "eligible.npy", eligible)
    (output / "FOUR_CONTEXT_BUILD_MANIFEST.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2: build a context benchmark")
    parser.add_argument("--source", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--perturbation-column", dest="perturbation_column", default=None,
                        help="optional under dual_guide, where identity comes from the pair")
    parser.add_argument("--batch-column", dest="batch_column", required=True)
    parser.add_argument("--control-labels", dest="control_labels", nargs="+", required=True)
    parser.add_argument("--identity-mode", dest="identity_mode", default="column",
                        choices=("column", "dual_guide"),
                        help="how a cell's intervention identity is established")
    parser.add_argument("--guide-column", dest="guide_column", default=None)
    parser.add_argument("--gene-symbol-column", dest="gene_symbol_column", default=None,
                        help="var column carrying gene symbols, when var_names does not")
    parser.add_argument("--output", required=True)
    record = run(parser.parse_args())
    print(f"{record['context']}: {record['source']['n_obs_retained']} of "
          f"{record['source']['n_obs_total']} cells retained, "
          f"{record['source']['n_vars']} genes")
    if record["dual_guide_exclusion"].get("excluded_fraction"):
        print(f"excluded {record['dual_guide_exclusion']['excluded_fraction']:.4f} of cells: "
              f"{record['dual_guide_exclusion'].get('cross_gene_cells', 0)} cross-gene, "
              f"{record['dual_guide_exclusion'].get('unparsed_cells', 0)} unparsed")
    print(f"batches {record['batches']['total']} split {record['batches']['side_A']}/"
          f"{record['batches']['side_B']}, quarters {record['batches']['batches_per_quarter']}, "
          f"{record['batches']['control_cells']} control cells")
    print(f"identities nominal {record['identities']['nominal']}, usable {record['identities']['usable']}, "
          f"eligible {record['query_gate']['eligible']}")
    print(f"usable and four-way {record['four_way']['usable_and_four_way']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
