#!/usr/bin/env python3
"""CellNavi inference witness (runs in the ``cellnavi`` env).  It does not import ``cellnavi_fit``.

For a few hashed (query, view) rows it rebuilds the model from the step-1000 checkpoint, runs the
official validation tokenization on exactly those rows' cells (a query-file subset, with the upstream
int64 mask gather rather than the run's int8 adapter, so the adapter is checked too), averages ``log_softmax`` itself, and compares with the saved
``scores.npz``.  This proves the saved scores are the model's, row mapping included.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="CellNavi inference witness")
    parser.add_argument("--run-dir", dest="run_dir", required=True)
    parser.add_argument("--held", required=True)
    parser.add_argument("--code-dir", dest="code_dir", required=True)
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--subdir", default="cellnavi", help="cellnavi, or cellnavi_native for Deviation 5")
    args = parser.parse_args()
    sys.path.insert(0, args.code_dir)
    import anndata
    import torch
    import torch.nn as nn

    root = Path(args.run_dir) / args.subdir / args.held
    target = root / "VERIFICATION_CELLNAVI.json"
    if target.exists():
        raise FileExistsError(target)
    saved = np.load(root / "scores.npz", allow_pickle=False)
    classes = saved["classes"].astype(str).tolist()
    keys = list(zip(saved["queries"].astype(str).tolist(), saved["views"].tolist()))
    pick = sorted(keys, key=lambda k: hashlib.sha256(f"WITNESS:{k[0]}:{k[1]}".encode()).hexdigest())[:args.rows]

    data = root / "data"
    query = anndata.read_h5ad(data / "query.h5ad")
    mask = np.zeros(query.n_obs, dtype=bool)
    for q, v in pick:
        mask |= (query.obs["query"].astype(str).to_numpy() == q) & (query.obs["view"].astype(int).to_numpy() == v)
    subset = query[mask].copy()
    witness_dir = root / "witness"
    (witness_dir / "work").mkdir(parents=True, exist_ok=False)
    subset.write_h5ad(witness_dir / "subset.h5ad")
    for name in ("dist_t_matrix.csv", "adj_t_matrix.csv", "gene_name.txt"):
        (witness_dir / name).symlink_to((data / name).resolve())
    config = json.loads((root / "config.json").read_text())
    config.update(dataset_dir=str(witness_dir) + "/", test_data="subset.h5ad")
    (witness_dir / "config.json").write_text(json.dumps(config))
    os.chdir(witness_dir / "work")

    import cellnavi.model.finetune_model as fm
    from cellnavi.data_provider.dataset import ValidationDataset
    model = fm.FinetuneModel()
    model.fc[-1] = nn.Linear(model.pretrain.d_model * 4, len(classes))
    state = torch.load(data / "finetune" / "model" / "checkpoint-step-1000.pth", map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in state.items()})
    model = model.eval().cuda()
    dataset = ValidationDataset()
    names = dataset._dataset.cell_name
    obs = subset.obs.loc[names]
    sums = {k: np.zeros(len(classes)) for k in pick}
    counts = {k: 0 for k in pick}
    with torch.no_grad():
        for i in range(len(dataset)):
            batch = dataset[i]            # already batched (batch of one), as the official loader yields
            logits = model({k: v.cuda() for k, v in batch.items()})["pred"][0].float()
            key = (str(obs["query"].iloc[i]), int(obs["view"].iloc[i]))
            sums[key] += torch.log_softmax(logits, dim=0).cpu().numpy()
            counts[key] += 1
    facts = {}
    for key in pick:
        row = keys.index(key)
        ours = sums[key] / counts[key]
        facts[f"{key[0]}|v{key[1]}"] = {"cells": counts[key], "saved_cells": int(saved["cells"][row]),
                                        "max_abs": float(np.abs(ours - saved["mean_log_prob"][row]).max())}
    ok = all(f["cells"] == f["saved_cells"] and f["max_abs"] < 1e-3 for f in facts.values())
    out = {"held": args.held, "variant": args.subdir, "rows": facts, "tolerance": 1e-3,
           "tolerance_reason": "GPU kernel nondeterminism across processes in fp32 attention",
           "all_pass": bool(ok)}
    target.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
