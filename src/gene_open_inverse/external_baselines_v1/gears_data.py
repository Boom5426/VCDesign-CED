#!/usr/bin/env python3
"""Build GEARS's training AnnData from K562 ``G_fit`` single cells (runs in the project environment).

Only anchor ``G_fit`` identities (the rows ``RIDGE_UNIT`` may train on in K562) that lie in GEARS's
default perturbation graph are kept, each capped at ``GEARS_CELLS_PER_IDENTITY`` cells chosen by a
hashed cell key, plus ``GEARS_CONTROL_CELLS`` hashed control cells.  Expression is log1p(CP10k) on
the full row, restricted to the frozen four-context axis ``G_4``.  Masking happens later, per fit,
by splits; this file is the one shared cell pool.  No other K562 identity's expression is read.
"""
from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import cells
from . import contract as cc


def pert_graph(gears_data: Path) -> set[str]:
    """GEARS's default perturbation graph: ``gene2go_all`` restricted to its essential gene list."""
    gene2go = pickle.load(open(gears_data / "gene2go_all.pkl", "rb"))
    essential = pickle.load(open(gears_data / "essential_all_data_pert_genes.pkl", "rb"))
    return {g for g in essential if g in gene2go}


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    axes = cells.load_pack_axes(args.four_context_run, "K562")
    g_fit = sorted(set(axes["ids"][axes["fit_mask"]].tolist()))
    graph = pert_graph(Path(args.gears_data))
    kept = [g for g in g_fit if g in graph]
    source = cells.Source("K562")
    labels = set(source.labels.tolist())
    missing = [g for g in kept if g not in labels]
    if missing:
        raise RuntimeError(f"{len(missing)} G_fit identities have no K562 cell label, e.g. {missing[:5]}")
    genes = cells.g4(args.four_context_run)

    rows, condition = [], []
    per_identity = {}
    for identity in kept:
        chosen = source.select(identity, cc.GEARS_CELLS_PER_IDENTITY, cc.GEARS_CELL_NAMESPACE)
        rows.append(chosen)
        condition += [f"{identity}+ctrl"] * chosen.size
        per_identity[identity] = int(chosen.size)
    control = source.select(source.control, cc.GEARS_CONTROL_CELLS, cc.GEARS_CELL_NAMESPACE)
    rows.append(control)
    condition += ["ctrl"] * control.size
    rows = np.concatenate(rows)
    order = np.argsort(rows, kind="stable")
    if np.unique(rows).size != rows.size:
        raise RuntimeError("a cell was selected twice")
    values = source.lognorm(rows[order], genes)
    back = np.empty_like(order)
    back[order] = np.arange(order.size)
    values = values[back]

    adata = anndata.AnnData(
        X=sp.csr_matrix(values.astype(np.float32)),
        obs=pd.DataFrame({"condition": condition, "cell_type": "K562"},
                         index=[f"K562|{source.names[r]}" for r in rows.tolist()]),
        var=pd.DataFrame({"gene_name": genes}, index=genes))
    adata.write_h5ad(output / "k562_gfit_gears.h5ad")
    record = {
        "schema": "VCDESIGN_EBC_V1_GEARS_DATA", "read_G_check": False,
        "source": source.path, "identity_rule": "anchor G_fit identities in GEARS's default perturbation graph",
        "g_fit": len(g_fit), "kept": len(kept), "not_in_pert_graph": sorted(set(g_fit) - set(kept)),
        "cells_per_identity_cap": cc.GEARS_CELLS_PER_IDENTITY, "control_cells": int(control.size),
        "cells": int(rows.size), "genes": int(genes.size),
        "cells_per_identity": {"min": int(min(per_identity.values())), "median": float(np.median(list(per_identity.values()))),
                               "max": int(max(per_identity.values()))},
        "timestamp": datetime.now(timezone.utc).isoformat()}
    (output / "gears_data.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    (output / "identities.json").write_text(json.dumps(kept) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="GEARS K562 G_fit training data")
    parser.add_argument("--four-context-run", dest="four_context_run", default=cc.FOUR_CONTEXT_RUN)
    parser.add_argument("--gears-data", dest="gears_data", required=True)
    parser.add_argument("--output", required=True)
    record = run(parser.parse_args())
    print(json.dumps({k: v for k, v in record.items() if k != "not_in_pert_graph"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
