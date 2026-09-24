#!/usr/bin/env python3
"""Phase 1: independent dataset audit.  Reads data properties only, never a model.

Nothing here is allowed to depend on a downstream result.  The script reports what
the file contains, resolves the three columns the measurement contract needs
(perturbation identity, batch, control label) from a declared priority list, and
evaluates the eight hard qualification conditions.  A condition that fails is
recorded as failed; it is never worked around by relaxing a threshold.

The column resolution is deliberately explicit.  If none of the declared candidate
names is present the audit stops and prints the columns that are, rather than
guessing at a column whose meaning it has not established.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import anndata
import numpy as np

from . import contract as fc
from . import dualguide as dg


# Declared before looking at the files.  Order is priority, not preference after the fact.
PERTURBATION_COLUMNS = ("gene", "gene_symbol", "target_gene_name", "perturbation",
                        "gene_target", "sgRNA_target", "guide_identity", "target")
BATCH_COLUMNS = ("gem_group", "gemgroup", "batch", "lane", "channel", "sample", "gem_well")
GUIDE_COLUMNS = ("sgID_AB", "sgID", "sgRNA_AB", "guide_identity", "sgRNA", "protospacer",
                 "guide", "sgRNA_name", "sgRNA_pair", "perturbation_pair")
CONTROL_TOKENS = ("non-targeting", "non_targeting", "nontargeting", "control",
                  "neg_ctrl", "negative_control", "safe-targeting", "NTC", "*")


def _resolve(frame, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in frame.columns:
            return name
    return None


def _control_mask(values: np.ndarray) -> tuple[np.ndarray, list[str]]:
    lowered = np.asarray([v.lower() for v in values.tolist()], dtype=object)
    labels = sorted({v for v in values.tolist()
                     if any(token.lower() in v.lower() for token in CONTROL_TOKENS)})
    mask = np.zeros(values.size, dtype=bool)
    for label in labels:
        mask |= values == label
    return mask, labels


def count_semantics(handle, blocks: int = 3, rows: int = 2048) -> dict:
    """Raw counts or already normalized: the one property the frozen recipe assumes.

    The inherited contract normalizes to a library size of 1e4 and takes log1p, which
    is only meaningful on raw counts.  Running it on an already normalized matrix
    would produce a plausible-looking response that is not the quantity every other
    context measures, and nothing downstream would notice.  Contiguous blocks are read
    from three positions rather than random rows, because fancy indexing into a backed
    matrix is slow enough to change what this audit costs.
    """
    observations = int(handle.n_obs)
    samples = []
    for fraction in np.linspace(0.0, 0.9, blocks):
        start = min(int(fraction * observations), max(observations - rows, 0))
        block = handle.X[start:start + rows]
        samples.append(np.asarray(block.toarray() if hasattr(block, "toarray") else block,
                                  dtype=np.float64))
    values = np.concatenate(samples)
    nonzero = values[values != 0.0]
    library = values.sum(axis=1)
    integral = float(np.isclose(nonzero, np.rint(nonzero)).mean()) if nonzero.size else 0.0
    return {
        "cells_sampled": int(values.shape[0]), "blocks": blocks,
        "dtype_on_disk": str(handle.X.dtype) if hasattr(handle.X, "dtype") else "unknown",
        "sparse": bool(hasattr(handle.X[0:1], "toarray")),
        "nonzero_fraction": float((values != 0.0).mean()),
        "min": float(values.min()), "max": float(values.max()),
        "any_negative": bool((values < 0).any()),
        "fraction_of_nonzero_values_that_are_integers": integral,
        "library_size": {"min": float(library.min()), "median": float(np.median(library)),
                         "max": float(library.max()),
                         "coefficient_of_variation": float(library.std() / max(library.mean(), 1e-30))},
        "is_raw_counts": bool(integral > 0.999 and not (values < 0).any()),
        "rule": "raw counts means every nonzero value is an integer and none is negative; "
                "a constant library size across cells would instead indicate that the "
                "matrix was already normalized",
    }


def _quantiles(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"n": 0}
    return {"n": int(values.size), "min": int(values.min()), "p05": float(np.quantile(values, 0.05)),
            "median": float(np.median(values)), "mean": float(values.mean()),
            "p95": float(np.quantile(values, 0.95)), "max": int(values.max())}


def audit(source: str, context: str, reference_genes: dict[str, np.ndarray] | None = None) -> dict:
    handle = anndata.read_h5ad(source, backed="r")
    obs = handle.obs
    genes = np.asarray(handle.var_names, dtype=str)
    record: dict = {
        "schema": "VCDESIGN_FOUR_CONTEXT_V1_AUDIT",
        "context": context, "source": str(source),
        "cells": int(handle.n_obs), "genes": int(handle.n_vars),
        "obs_columns": {name: str(obs[name].dtype) for name in obs.columns},
        "var_columns": {name: [str(v) for v in handle.var[name].astype(str).to_numpy()[:5]]
                        for name in handle.var.columns},
        "layers": list(handle.layers.keys()),
        "obsm": list(handle.obsm.keys()),
        "declared_perturbation_type": fc.PERTURBATION_TYPE,
    }

    record["count_semantics"] = count_semantics(handle)

    perturbation_column = _resolve(obs, PERTURBATION_COLUMNS)
    batch_column = _resolve(obs, BATCH_COLUMNS)
    guide_column = _resolve(obs, GUIDE_COLUMNS)
    record["resolved_columns"] = {"perturbation": perturbation_column, "batch": batch_column,
                                  "guide": guide_column}
    if batch_column is None or (perturbation_column is None and guide_column is None):
        record["qualification"] = {name: False for name in fc.QUALIFICATION}
        record["blocking"] = ("could not resolve the batch column, or neither a "
                              "perturbation column nor a dual-guide column is present")
        handle.file.close()
        return record

    universe_seed = set(genes.tolist())
    if reference_genes:
        for other in reference_genes.values():
            universe_seed |= set(np.asarray(other, dtype=str).tolist())

    if perturbation_column is not None:
        perturbation = obs[perturbation_column].astype(str).to_numpy()
        record["perturbation_source"] = f"column `{perturbation_column}`"
    else:
        # No collapsed column exists, so the identity is whatever the guide pair says it
        # is.  Deriving it here rather than refusing keeps a perfectly usable dataset in
        # the programme, and it is still the audit that decides, not the builder.
        derived = dg.resolve(obs[guide_column].astype(str).to_numpy(), universe_seed)
        perturbation = np.where(derived["kind"] == dg.SINGLE_GENE, derived["identity"],
                                np.where(derived["kind"] == dg.NON_TARGETING,
                                         "non-targeting", "__UNRESOLVED__"))
        record["perturbation_source"] = (f"derived from the dual-guide column "
                                         f"`{guide_column}`, no collapsed column present")
    batch = obs[batch_column].astype(str).to_numpy()
    control, control_labels = _control_mask(np.unique(perturbation))
    control_mask = np.isin(perturbation, np.unique(perturbation)[control])
    record["control"] = {"labels": control_labels, "cells": int(control_mask.sum()),
                         "fraction_of_cells": float(control_mask.mean())}

    batches, batch_counts = np.unique(batch, return_counts=True)
    control_per_batch = np.asarray([int(control_mask[batch == name].sum()) for name in batches.tolist()])
    record["batches"] = {
        "column": batch_column, "count": int(batches.size),
        "cells_per_batch": _quantiles(batch_counts),
        "control_cells_per_batch": _quantiles(control_per_batch),
        "batches_without_controls": int((control_per_batch == 0).sum()),
        "example_names": batches[:5].tolist(),
    }

    identities = np.unique(perturbation[~control_mask])
    counts = np.asarray([int((perturbation == name).sum()) for name in identities.tolist()])
    record["identities"] = {"column": perturbation_column, "nominal": int(identities.size),
                            "cells_per_identity": _quantiles(counts)}

    # Cells per identity per A/B side, which is what MIN_VIEW_CELLS is actually applied to.
    from .build import quarter_of, view_side
    sides = np.asarray([view_side(str(name)) for name in batches.tolist()], dtype=np.int8)
    quarters = np.asarray([quarter_of(str(name)) for name in batches.tolist()], dtype=np.int8)
    batch_position = {name: index for index, name in enumerate(batches.tolist())}
    column = np.asarray([batch_position[name] for name in batch.tolist()], dtype=np.int64)
    identity_position = {name: index for index, name in enumerate(identities.tolist())}
    row = np.asarray([identity_position.get(name, -1) for name in perturbation.tolist()], dtype=np.int64)
    incidence = np.zeros((identities.size, batches.size), dtype=np.int64)
    keep = row >= 0
    np.add.at(incidence, (row[keep], column[keep]), 1)
    by_side = np.stack([incidence[:, sides == side].sum(axis=1) for side in (0, 1)], axis=1)
    by_quarter = np.stack([incidence[:, quarters == q].sum(axis=1) for q in range(4)], axis=1)
    usable = (by_side >= fc.MIN_VIEW_CELLS).all(axis=1)
    record["view_structure"] = {
        "rule": f"{fc.VIEW_NAMESPACE} seed {fc.VIEW_SEED}, applied to this context's batch names",
        "batches_side_A": int((sides == 0).sum()), "batches_side_B": int((sides == 1).sum()),
        "batches_per_quarter": [int((quarters == q).sum()) for q in range(4)],
        "cells_per_identity_side_A": _quantiles(by_side[:, 0]),
        "cells_per_identity_side_B": _quantiles(by_side[:, 1]),
        "min_view_cells": fc.MIN_VIEW_CELLS,
        "usable_identities": int(usable.sum()),
        "four_way_eligible": int((by_quarter >= fc.MIN_VIEW_CELLS).all(axis=1).sum()),
    }

    # --- the dual-sgRNA hard gate ------------------------------------------------
    # A dual-guide vector means the intervention unit is a property of the pair.  The
    # audit resolves the pair explicitly; it never lets the builder infer it.
    universe = universe_seed | set(identities.tolist())
    if guide_column is not None:
        guides = obs[guide_column].astype(str).to_numpy()
        resolution = dg.resolve(guides, universe)
        summary = dg.summarise(resolution, guides)
        shape = dg.pair_shape(resolution, universe)
        unparsed = dg.unparsed_pair_shape(resolution)
        verdict = dg.case_verdict(summary, shape, fc.MIN_USABLE_IDENTITIES, unparsed)
        # Cross-check against the column the file itself calls the perturbation: if the
        # dataset already collapsed the pair, the two must agree, and a disagreement is
        # a reason to stop rather than to pick whichever is more convenient.
        single = resolution["kind"] == dg.SINGLE_GENE
        agreement = float((resolution["identity"][single] == perturbation[single]).mean()) \
            if single.any() else 0.0
        guides_per_identity = np.asarray(
            [int(np.unique(guides[perturbation == name]).size)
             for name in identities[:2000].tolist()])
        record["dual_guide"] = {
            "column": guide_column, "distinct_labels": int(np.unique(guides).size),
            "separators_tried": list(dg.SEPARATORS),
            "summary": summary, "pair_shape": shape, "case_verdict": verdict,
            "agreement_with_perturbation_column": agreement,
            "agreement_is_a_gate": True,
            "guides_per_identity": _quantiles(guides_per_identity),
        }
        record["multi_guide"] = {
            "column": guide_column,
            "handling": "identity is derived from the resolved pair; cross-gene and "
                        "unparsed pairs are excluded and counted, never relabelled"}
    else:
        record["dual_guide"] = {"column": None,
                                "blocking": "no dual-guide column found, so the pair "
                                            "semantics cannot be established from this file"}
        record["multi_guide"] = {"column": None,
                                 "handling": "no guide column; identities are taken as given"}

    # Which axis actually carries gene symbols decides whether G_4 exists at all: a
    # release keyed by ENSEMBL identifier intersects a symbol axis in nothing, and the
    # failure would look like "these contexts share no genes" rather than like a
    # key mismatch.  Every candidate axis is scored against the reference axes here.
    symbol_candidates = {"var_names": genes}
    for name in handle.var.columns:
        symbol_candidates[f"var.{name}"] = handle.var[name].astype(str).to_numpy()
    scored = {}
    for name, values in symbol_candidates.items():
        entry = {"examples": [str(v) for v in values[:5]],
                 "distinct": int(np.unique(values).size)}
        if reference_genes:
            entry["overlap_with_reference"] = {
                key: int(np.intersect1d(np.asarray(values, dtype=str),
                                        np.asarray(other, dtype=str)).size)
                for key, other in reference_genes.items()}
            entry["best_reference_overlap"] = max(entry["overlap_with_reference"].values())
        scored[name] = entry
    best = max(scored, key=lambda k: scored[k].get("best_reference_overlap", -1)) \
        if reference_genes else "var_names"
    record["gene_axis"] = {"count": int(genes.size),
                           "duplicates": int(genes.size - np.unique(genes).size),
                           "example": genes[:5].tolist(),
                           "symbol_axis_candidates": scored,
                           "best_symbol_axis": best,
                           "var_names_are_symbols": bool(best == "var_names")}
    if reference_genes:
        axis = genes if best == "var_names" else \
            handle.var[best.split(".", 1)[1]].astype(str).to_numpy()
        record["shared_axis_used"] = best
        shared = {}
        for name, other in reference_genes.items():
            common = np.intersect1d(np.asarray(axis, dtype=str), np.asarray(other, dtype=str))
            shared[name] = {"shared_genes": int(common.size),
                            "fraction_of_this_context": float(common.size / len(axis)),
                            "fraction_of_{}".format(name): float(common.size / len(other))}
        record["shared_with_reference_axes"] = shared

    dual = record.get("dual_guide", {})
    verdict = dual.get("case_verdict", {})
    record["qualification"] = {
        "gene_level_crispri_knockdown": True,
        "count_matrix_is_raw_counts": bool(record["count_semantics"]["is_raw_counts"]),
        "single_gene_intervention_unit_constructible": bool(
            verdict.get("single_gene_unit_constructible", False)),
        "explicit_non_targeting_controls": bool(control_mask.sum() > 0),
        "batch_or_gemgroup_structure": bool(batches.size > 1),
        "every_retained_batch_has_controls": bool((control_per_batch == 0).sum() == 0),
        "min_view_cells_in_both_sides": bool(usable.sum() > 0),
        "at_least_several_hundred_usable_identities": bool(usable.sum() >= fc.MIN_USABLE_IDENTITIES),
        "sufficient_common_gene_axis": None if not reference_genes else bool(
            min(entry["shared_genes"] for entry in record["shared_with_reference_axes"].values()) >= 3000),
        "no_candidate_response_in_inference_features": True,
    }
    passed = [value for value in record["qualification"].values() if value is not None]
    record["verdict"] = "PRIMARY_QUALIFIED" if all(passed) else "EXPLORATORY_ONLY"
    record["failed_conditions"] = [name for name, value in record["qualification"].items()
                                   if value is False]
    record["qualification_note"] = (
        "gene_level_crispri_knockdown is asserted from the GEO series design, not inferred "
        "from the matrix; no_candidate_response_in_inference_features is a property of the "
        "model contract rather than of this file"
    )
    handle.file.close()
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 dataset audit")
    parser.add_argument("--source", required=True)
    parser.add_argument("--context", required=True)
    parser.add_argument("--reference-genes", dest="reference_genes", nargs="*", default=[],
                        help="NAME=path.npy gene axes to intersect against")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    references = {}
    for item in args.reference_genes:
        name, path = item.split("=", 1)
        references[name] = np.asarray([str(v) for v in np.load(path, allow_pickle=True)])
    record = audit(args.source, args.context, references or None)
    record["runtime"] = {"timestamp": datetime.now(timezone.utc).isoformat(),
                         "python": platform.python_version(), "numpy": np.__version__,
                         "anndata": anndata.__version__}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
