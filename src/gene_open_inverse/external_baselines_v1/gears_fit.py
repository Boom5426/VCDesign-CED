#!/usr/bin/env python3
"""GEARS on K562 G_fit, official package, interface adapter only (runs in the ``pertbench`` env).

Stages
  process  GEARS's own ``new_data_process`` once over the shared cell pool (DE genes, cell graphs)
  fit      one fit: SEEN, or the MASKED fit of one C-016 fold under ``GEARS_MASK_RULE``.  Masked identities enter no
           split, no loader, no validation and no co-expression graph; GEARS's ``no_test`` mode is
           used so no test pass runs.  Official hyper-parameters, GEARS's own best-validation epoch.
           Then the predicted shift ``x_hat_c - x_s`` for every candidate identity of the three
           primary pools that lies in GEARS's perturbation graph.
The model output is ``f(pert) + x`` (``gears/model.py``), so the shift is the same for every control
cell; the adapter checks that on ``GEARS_PREDICTION_CONTROLS`` cells rather than assuming it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import contract as cc

DATASET = "k562_gfit"


def _bucket(value: str, namespace: str, modulus: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{namespace}:{value}".encode()).digest()[:8], "big") % modulus


class SeriesSafeAnnData:
    """Environment shim, no numerical change.  GEARS 0.1.2 indexes ``adata.X`` with a pandas boolean
    Series (``gears.py`` line 87), which the scipy in this environment rejects.  GEARS receives this
    proxy, whose ``X`` converts such a key to the numpy array older scipy converted it to; every other
    attribute is the wrapped AnnData's own.  (Assigning a subclass to ``adata.X`` does not survive,
    because AnnData re-wraps it.)"""

    def __init__(self, adata):
        import scipy.sparse as sp

        class SeriesSafeCSR(sp.csr_matrix):
            def __getitem__(self, key):
                return super().__getitem__(key.to_numpy() if hasattr(key, "to_numpy") else key)

        self._adata = adata
        self.X = SeriesSafeCSR(adata.X)

    def __getattr__(self, name):
        return getattr(self._adata, name)

    def __getitem__(self, key):
        return self._adata[key]


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_set(c018_run: str, four_context_run: str, held: str, fold: int) -> list[str]:
    """``M_{held,fold}``: C-016 eligible identities of the held pool in this fold (C-018's arrays)."""
    with np.load(Path(c018_run) / f"effects_{held}.npz", allow_pickle=False) as store:
        eligible, folds = store["eligible"], store["folds"]
    with np.load(Path(four_context_run) / "packs" / f"{held}.npz", allow_pickle=False) as store:
        ids = store["ids"].astype(str)
    return sorted(ids[eligible & (folds == fold)].tolist())


def fold_union(c018_run: str, four_context_run: str, fold: int) -> list[str]:
    """``GEARS_MASK_RULE``: the union of the three held pools' C-016 masked sets for one fold."""
    return sorted(set().union(*(masked_set(c018_run, four_context_run, h, fold) for h in cc.PRIMARY_CONTEXTS)))


def make_split(kept: list[str], excluded: list[str]) -> tuple[dict, list[str], list[str]]:
    """GEARS's custom split: hashed 10% validation, the rest training, masked identities nowhere.

    ``test`` repeats ``val`` only because GEARS's custom-split format requires the key; the fit
    switches to ``no_test`` before building loaders, so it is never iterated.
    """
    allowed = sorted(set(kept) - set(excluded))
    val = [g for g in allowed if _bucket(g, cc.GEARS_VAL_NAMESPACE, cc.GEARS_VAL_MODULUS) == 0]
    train = [g for g in allowed if g not in set(val)]
    split = {"train": [f"{g}+ctrl" for g in train] + ["ctrl"], "val": [f"{g}+ctrl" for g in val],
             "test": [f"{g}+ctrl" for g in val]}
    if set(excluded) & {c.split("+")[0] for part in split.values() for c in part}:
        raise RuntimeError("a masked identity reached a GEARS split")
    return split, train, val


def candidate_universe(four_context_run: str) -> list[str]:
    out = set()
    for held in cc.PRIMARY_CONTEXTS:
        with np.load(Path(four_context_run) / "packs" / f"{held}.npz", allow_pickle=False) as store:
            out |= set(store["ids"].astype(str).tolist())
    return sorted(out)


def process(args: argparse.Namespace) -> None:
    import anndata
    from gears import PertData
    _seed_all(cc.GEARS_SEED)
    root = Path(args.gears_root)
    if (root / DATASET).exists():
        raise FileExistsError(f"{root / DATASET} exists")
    adata = anndata.read_h5ad(Path(args.data_dir) / "k562_gfit_gears.h5ad")
    pert_data = PertData(str(root))
    pert_data.new_data_process(dataset_name=DATASET, adata=adata)


def repro(args: argparse.Namespace) -> dict:
    """GEARS's official tutorial, run as published; the gate is its printed test headline."""
    import io
    import re
    import sys
    from gears import GEARS, PertData
    target = cc.GEARS_REPRO
    root = Path(args.run_dir) / ("gears_repro" if args.default_pert_graph else "gears_repro_tutorial_graph")
    root.mkdir(parents=True, exist_ok=False)
    _seed_all(target["split_seed"])
    capture, stderr = io.StringIO(), sys.stderr

    class Tee:
        def write(self, text):
            capture.write(text)
            return stderr.write(text)

        def flush(self):
            stderr.flush()

    sys.stderr = Tee()
    try:
        pert_data = PertData(str(root), default_pert_graph=args.default_pert_graph)
        pert_data.load(data_name=target["dataset"])
        pert_data.prepare_split(split=target["split"], seed=target["split_seed"])
        pert_data.get_dataloader(batch_size=target["batch_size"], test_batch_size=target["test_batch_size"])
        composition = {k: len(v) for k, v in pert_data.subgroup["test_subgroup"].items()}
        pert_data.adata = SeriesSafeAnnData(pert_data.adata)
        model = GEARS(pert_data, device="cuda")
        model.model_initialize(hidden_size=target["hidden_size"])
        model.train(epochs=target["epochs"], lr=target["lr"])
    finally:
        sys.stderr = stderr
    text = capture.getvalue()
    headline = re.search(r"Test Top 20 DE MSE: ([0-9.]+)", text)
    metrics = {m.group(1): float(m.group(2)) for m in re.finditer(r"^(test_[A-Za-z0-9_]+): ([-0-9.eE]+)$", text, re.M)}
    value = float(headline.group(1)) if headline else None
    record = {"stage": "GEARS_OFFICIAL_TUTORIAL_REPRODUCTION", "package": cc.GEARS_PACKAGE, "target": target,
              "test_composition": composition, "test_top20_de_mse": value, "printed_test_metrics": metrics,
              "passes": bool(value is not None and composition == target["target_composition"]
                             and abs(value - target["target_test_top20_de_mse"]) <= target["tolerance"]),
              "shim": "SeriesSafeAnnData (environment only)", "timestamp": datetime.now(timezone.utc).isoformat()}
    record["default_pert_graph"] = bool(args.default_pert_graph)
    passed = "GEARS_OFFICIAL_REPRODUCTION_PASSED" + ("" if args.default_pert_graph else "_ON_TUTORIAL_GRAPH")
    record["verdict"] = passed if record["passes"] else "GEARS_OFFICIAL_REPRODUCTION_FAILED"
    (root / "reproduction.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def fit(args: argparse.Namespace) -> dict:
    from gears import GEARS, PertData
    from gears.utils import create_cell_graph_for_prediction
    from torch_geometric.data import Batch

    name = cc.SEEN_POOL if args.fold is None else f"FOLD_{args.fold}"
    output = Path(args.run_dir) / "gears" / name
    output.mkdir(parents=True, exist_ok=False)
    seed = cc.GEARS_SEED + (0 if args.fold is None else 1 + args.fold)
    _seed_all(seed)

    kept = json.loads((Path(args.data_dir) / "identities.json").read_text())
    excluded = [] if args.fold is None else fold_union(args.c018_run, args.four_context_run, args.fold)
    split, train, val = make_split(kept, excluded)
    split_path = output / "split.pkl"
    pickle.dump(split, open(split_path, "wb"))

    pert_data = PertData(str(Path(args.gears_root)))
    pert_data.load(data_path=str(Path(args.gears_root) / DATASET))
    pert_data.prepare_split(split="custom", seed=seed, split_dict_path=str(split_path))
    pert_data.split = "no_test"           # train and val loaders only; no test pass over any cell
    pert_data.train_gene_set_size = 0.75  # only enters GEARS's cache file name, with the seed
    pert_data.get_dataloader(batch_size=cc.GEARS_HYPER["batch_size"],
                             test_batch_size=cc.GEARS_HYPER["test_batch_size"])
    pert_data.adata = SeriesSafeAnnData(pert_data.adata)
    model = GEARS(pert_data, device="cuda")
    model.model_initialize(hidden_size=cc.GEARS_HYPER["hidden_size"])
    model.train(epochs=cc.GEARS_HYPER["epochs"], lr=cc.GEARS_HYPER["lr"],
                weight_decay=cc.GEARS_HYPER["weight_decay"])
    torch.save(model.best_model.state_dict(), output / "best_model.pt")

    # predicted shift for every candidate identity in the perturbation graph
    net = model.best_model.to("cuda").eval()
    ctrl = pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
    names = [str(v) for v in ctrl.obs_names]
    chosen = np.sort(np.argsort([_bucket(v, cc.GEARS_CELL_NAMESPACE + "_PREDICT", 2 ** 61) for v in names])
                     [:cc.GEARS_PREDICTION_CONTROLS])
    basal = np.asarray(ctrl.X[chosen].toarray() if hasattr(ctrl.X, "toarray") else ctrl.X[chosen], dtype=np.float32)
    pert_list = model.pert_list
    universe = candidate_universe(args.four_context_run)
    covered = [g for g in universe if g in set(pert_list)]
    delta = np.zeros((len(covered), basal.shape[1]), dtype=np.float32)
    spread = 0.0
    with torch.no_grad():
        for start in range(0, len(covered), 64):
            genes = covered[start:start + 64]
            graphs = []
            for g in genes:
                pert_idx = [pert_list.index(g)]
                graphs += [create_cell_graph_for_prediction(basal[i], pert_idx, [g]) for i in range(basal.shape[0])]
            batch = Batch.from_data_list(graphs).to("cuda")
            pred = net(batch).reshape(len(genes), basal.shape[0], -1).cpu().numpy()
            shift = pred - basal[None]
            spread = max(spread, float(np.abs(shift - shift[:, :1]).max()))
            delta[start:start + len(genes)] = shift.mean(axis=1)
    if spread > cc.GEARS_INVARIANCE_TOLERANCE:
        raise RuntimeError(f"GEARS shift depends on the control cell ({spread:.2e}); the adapter's premise fails")
    np.savez(output / "effects.npz", identities=np.asarray(covered), delta=delta,
             genes=np.asarray(pert_data.adata.var["gene_name"].astype(str)),
             excluded=np.asarray(excluded, dtype=str), training_identities=np.asarray(train + val))
    record = {"fit": name, "seed": seed, "fold": args.fold, "mask_rule": cc.GEARS_MASK_RULE,
              "excluded": len(excluded), "train_identities": len(train), "val_identities": len(val),
              "excluded_in_kept": len(set(excluded) & set(kept)),
              "covered_candidates": len(covered), "universe": len(universe),
              "control_invariance_max_abs": spread, "hyper": cc.GEARS_HYPER,
              "package": cc.GEARS_PACKAGE, "timestamp": datetime.now(timezone.utc).isoformat()}
    (output / "record.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="GEARS adapter")
    parser.add_argument("--stage", choices=("process", "fit", "repro"), required=True)
    parser.add_argument("--gears-root", dest="gears_root", required=True)
    parser.add_argument("--data-dir", dest="data_dir")
    parser.add_argument("--run-dir", dest="run_dir")
    parser.add_argument("--fold", type=int, default=None, help="omit for the SEEN fit")
    parser.add_argument("--tutorial-graph", dest="default_pert_graph", action="store_false",
                        help="repro only: default_pert_graph=False (Deviation 4)")
    parser.add_argument("--c018-run", dest="c018_run", default=cc.C018_RUN)
    parser.add_argument("--four-context-run", dest="four_context_run", default=cc.FOUR_CONTEXT_RUN)
    args = parser.parse_args()
    if args.stage == "process":
        process(args)
    elif args.stage == "repro":
        print(json.dumps(repro(args), indent=1))
    else:
        print(json.dumps(fit(args), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
