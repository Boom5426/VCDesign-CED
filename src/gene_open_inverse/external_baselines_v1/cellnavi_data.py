#!/usr/bin/env python3
"""CellNavi input files for one held context (runs in the project environment).

Training file: raw-count cells of every identity the held context's ``RIDGE_UNIT`` SEEN fit may train
on (K562: anchor ``G_fit``; RPE1, HepG2, Jurkat: every usable identity), from the three training
contexts only, capped at ``CELLNAVI_CELLS_PER_CLASS`` hashed cells per (context, identity).  The label
is the identity symbol; classes are their sorted union.  No control cell is used (CellNavi's released
code never reads controls).

Query file: the held context's query identities, up to ``CELLNAVI_QUERY_CELLS`` hashed cells per
(query, view) from that view's batches, so a view-v row's CellNavi input comes from the same batches
that build its view-v query direction.

Gene axis: the released CellNavi graph's genes in its column order (the official reader indexes every
one by name).  Counts are filled for graph genes in ``G_4``; every other graph gene is zero and so is
never tokenized.  Every arm in this mission therefore sees the same measured gene universe.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import cells
from . import contract as cc


def stratum_identities(c018_run: str, four_context_run: str, held: str) -> set[str]:
    """The held context's C-016 SEEN stratum, from C-018's saved arrays."""
    with np.load(Path(c018_run) / f"effects_{held}.npz", allow_pickle=False) as store:
        eligible = store["eligible"]
    with np.load(Path(four_context_run) / "packs" / f"{held}.npz", allow_pickle=False) as store:
        return set(store["ids"].astype(str)[eligible].tolist())


def graph_genes(dist_csv: Path) -> np.ndarray:
    with open(dist_csv) as handle:
        header = handle.readline().rstrip("\n").split(",")
    return np.asarray(header[1:], dtype=str)


def _matrix(source: cells.Source, rows: np.ndarray, universe: np.ndarray, measured: np.ndarray) -> sp.csr_matrix:
    """Raw counts of ``rows`` (any order) on the graph gene axis, graph genes outside ``G_4`` zero.

    Rows are sorted once over the whole selection and read in contiguous sorted chunks, then put back
    in the caller's order; chunking before sorting would make every chunk span the whole file.
    """
    position = {g: i for i, g in enumerate(source.genes.tolist()) if source.gene_keep[i]}
    target = {g: j for j, g in enumerate(universe.tolist())}
    into = np.asarray([target[g] for g in measured.tolist()])
    take = np.asarray([position[g] for g in measured.tolist()])
    order = np.argsort(rows, kind="stable")
    ordered = rows[order]
    blocks = []
    for start in range(0, ordered.size, 16384):
        counts = source.counts(ordered[start:start + 16384])
        out = np.zeros((counts.shape[0], universe.size), dtype=np.float32)
        out[:, into] = counts[:, take]
        blocks.append(sp.csr_matrix(out))
    stacked = sp.vstack(blocks).tocsr()
    back = np.empty_like(order)
    back[order] = np.arange(order.size)
    return stacked[back]


def _write(path: Path, matrix, labels, index, universe, extra: dict | None = None) -> None:
    obs = pd.DataFrame({"perturbation": pd.Categorical(labels)}, index=index)
    for key, value in (extra or {}).items():
        obs[key] = value
    anndata.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=universe)).write_h5ad(path)


def run(args: argparse.Namespace) -> dict:
    held = args.held
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    keep_identities = (stratum_identities(args.c018_run, args.four_context_run, held) if args.classes == "stratum"
                       else set(np.load(Path(args.four_context_run) / "packs" / f"{held}.npz",
                                        allow_pickle=False)["ids"].astype(str).tolist())
                       | {i for c in cc.TRAINING_CONTEXTS if c != held
                          for i in cells.load_pack_axes(args.four_context_run, c)["ids"].tolist()})
    universe = graph_genes(Path(args.dist_csv))
    g4 = set(cells.g4(args.four_context_run).tolist())
    measured = np.asarray([g for g in universe.tolist() if g in g4])

    matrices, labels, index, per_context = [], [], [], {}
    for context in cc.TRAINING_CONTEXTS:
        if context == held:
            continue
        axes = cells.load_pack_axes(args.four_context_run, context)
        identities = sorted(set(axes["ids"][axes["fit_mask"]].tolist()) & keep_identities)
        source = cells.Source(context)
        chosen, tags = [], []
        for identity in identities:
            rows = source.select(identity, cc.CELLNAVI_CELLS_PER_CLASS, cc.CELLNAVI_CELL_NAMESPACE)
            chosen.append(rows)
            tags += [identity] * rows.size
        rows = np.concatenate(chosen)
        matrices.append(_matrix(source, rows, universe, measured))
        labels += tags
        index += [f"{context}|{source.names[r]}" for r in rows.tolist()]
        per_context[context] = {"identities": len(identities), "cells": int(rows.size)}
    matrix = sp.vstack(matrices).tocsr()
    classes = sorted(set(labels))
    _write(output / "train.h5ad", matrix, pd.Categorical(labels, categories=classes), index, universe)
    _write(output / "placeholder.h5ad", matrix[:4], pd.Categorical(labels[:4], categories=classes), index[:4], universe)

    # query cells of the held context, per (query, view)
    axes = cells.load_pack_axes(args.four_context_run, held)
    queries = axes["ids"][axes["self_positions"]]
    source = cells.Source(held)
    side = np.asarray([cells.view_side(b) for b in np.unique(source.batches).tolist()])
    side_of = dict(zip(np.unique(source.batches).tolist(), side.tolist()))
    cell_side = np.asarray([side_of[b] for b in source.batches.tolist()])
    build = Path(args.four_context_run) / held
    build_ids = np.load(build / "identity_axis.npy", allow_pickle=True).astype(str)
    by_side = np.load(build / "cells_by_side.npy")
    build_pos = {v: i for i, v in enumerate(build_ids.tolist())}
    rows, qlab, views, keys = [], [], [], []
    short = []
    for q in queries.tolist():
        for view in (0, 1):
            pool = np.flatnonzero((source.labels == q) & (cell_side == view))
            if pool.size != int(by_side[build_pos[q], view]):
                raise RuntimeError(f"{held} {q} view {view}: {pool.size} cells here, {by_side[build_pos[q], view]} in the build")
            chosen = source.select(q, cc.CELLNAVI_QUERY_CELLS, cc.CELLNAVI_CELL_NAMESPACE + "_QUERY", rows=pool)
            if chosen.size == 0:
                raise RuntimeError(f"{held} query {q} has no view-{view} cell")
            if chosen.size < cc.CELLNAVI_QUERY_CELLS:
                short.append([q, view, int(chosen.size)])
            rows.append(chosen)
            qlab += [q] * chosen.size
            views += [view] * chosen.size
    rows = np.concatenate(rows)
    qmatrix = _matrix(source, rows, universe, measured)
    _write(output / "query.h5ad", qmatrix, qlab, [f"{held}|{v}|{source.names[r]}" for r, v in zip(rows.tolist(), views)],
           universe, {"query": qlab, "view": views})
    record = {"schema": "VCDESIGN_EBC_V1_CELLNAVI_DATA", "held": held, "read_G_check": False,
              "class_rule": args.classes, "class_rule_text": cc.CELLNAVI_CLASS_RULES[args.classes],
              "training_contexts": per_context, "classes": len(classes), "train_cells": int(matrix.shape[0]),
              "graph_genes": int(universe.size), "graph_genes_in_G4": int(measured.size),
              "query_rows": int(2 * queries.size), "query_cells": int(rows.size),
              "query_views_below_cap": short, "timestamp": datetime.now(timezone.utc).isoformat()}
    (output / "classes.json").write_text(json.dumps(classes) + "\n")
    (output / "cellnavi_data.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="CellNavi input files for one held context")
    parser.add_argument("--held", required=True, choices=cc.PRIMARY_CONTEXTS)
    parser.add_argument("--dist-csv", dest="dist_csv", required=True)
    parser.add_argument("--four-context-run", dest="four_context_run", default=cc.FOUR_CONTEXT_RUN)
    parser.add_argument("--classes", choices=("all_training", "stratum"), default="all_training")
    parser.add_argument("--c018-run", dest="c018_run", default=cc.C018_RUN)
    parser.add_argument("--output", required=True)
    record = run(parser.parse_args())
    print(json.dumps({k: v for k, v in record.items() if k != "query_views_below_cap"}, indent=1))
    print(f"query views below cap: {len(record['query_views_below_cap'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
