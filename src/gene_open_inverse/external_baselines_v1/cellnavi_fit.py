#!/usr/bin/env python3
"""CellNavi, official code at ``CELLNAVI_COMMIT``, interface adapter only (runs in the ``cellnavi`` env).

Stages
  repro  the official reproduction gate: the released tutorial checkpoint on the released test file,
         evaluated by the notebook's gene-name route; must match accuracy 0.510 and weighted F1 0.498
  train  fine-tune from ``pretrain_weights.pth`` with the official trainer and the official tutorial
         recipe on one held context's training file.  The only change is the class-head width
         (``n_cls`` is hard-coded to 2058 upstream); it is set to the number of training classes
  infer  the official validation tokenization (no down-sampling, no truncation) over the held
         context's query cells; per cell ``log_softmax`` over classes, averaged per (query, view)
Nothing here reads a held-context utility.  The class head is the official MLP.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import contract as cc


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _link(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        link.symlink_to(target)


def _workspace(root: Path, files: dict, train: str, test: str) -> Path:
    """The directory layout the official code expects: ``work/`` is the cwd, ``../config.json``."""
    data = root / "data"
    for name, target in files.items():
        _link(Path(target), data / name)
    (data / "finetune" / "model").mkdir(parents=True, exist_ok=True)
    (data / "log").mkdir(parents=True, exist_ok=True)
    (root / "work").mkdir(parents=True, exist_ok=True)
    config = dict(cc.CELLNAVI_RECIPE, dataset_dir=str(data) + "/", log_dir=str(data / "log"),
                  model_dir=str(data / "finetune" / "model"), pretrain_model_dir=str(data / "pretrain"),
                  train_data=train, test_data=test, dist_graph="dist_t_matrix.csv", adj_graph="adj_t_matrix.csv")
    (root / "config.json").write_text(json.dumps(config, indent=4) + "\n")
    os.chdir(root / "work")
    return data


def _patch_classes(n_cls: int) -> None:
    """Widen the official class head; everything else in ``FinetuneModel`` is untouched."""
    import torch.nn as nn
    import cellnavi.model.finetune_model as fm
    original = fm.FinetuneModel.__init__

    def init(self):
        original(self)
        inplane = self.pretrain.d_model
        self.n_cls = n_cls
        self.fc[-1] = nn.Linear(inplane * 4, n_cls)

    fm.FinetuneModel.__init__ = init


def _patch_io() -> dict:
    """I/O-only adapter, identical tensors: the graph-distance mask gather.

    Upstream builds each cell's mask as ``mtx[u][:, u]`` on an int64 11,936 x 11,936 matrix, which
    copies about 195 MB per cell (0.33 s measured) and starves the GPU.  The distance codes are
    -1..4, so an int8 copy gathered with ``np.ix_`` returns the same values (0.043 s).  A self-check
    compares both gathers on a hashed index set before training or inference starts.
    """
    import cellnavi.data_provider.dataset as ds
    original_init, original_mask = ds.H5ADReader.__init__, ds.H5ADReader.get_mask
    record = {}

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.mtx.min() < -128 or self.mtx.max() > 127:
            raise RuntimeError("distance codes do not fit int8")
        probe = np.sort(np.random.default_rng(cc.GEARS_SEED).choice(self.mtx.shape[0], 512, replace=False))
        reference = original_mask(self, probe)
        self.mtx_int8 = np.ascontiguousarray(self.mtx.astype(np.int8))
        if not np.array_equal(self.mtx_int8[np.ix_(probe, probe)].astype(self.mtx.dtype), reference):
            raise RuntimeError("int8 gather differs from the upstream gather")
        record["self_check"] = "passed"

    def get_mask(self, u_cols):
        return self.mtx_int8[np.ix_(u_cols, u_cols)].astype(np.int64)

    ds.H5ADReader.__init__ = init
    ds.H5ADReader.get_mask = get_mask
    return record


def _load(model_dir: Path, step: int):
    import torch
    from cellnavi.model.finetune_model import FinetuneModel
    model = FinetuneModel()
    checkpoint = torch.load(model_dir / f"checkpoint-step-{step}.pth", map_location="cpu", weights_only=True)
    model.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in checkpoint["state_dict"].items()})
    return model.eval().cuda()


def _loader(workers: int):
    import torch
    from cellnavi.data_provider.dataset import ValidationDataset
    dataset = ValidationDataset()
    return dataset, torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=False,
                                                num_workers=workers, pin_memory=True)


def repro(args: argparse.Namespace) -> dict:
    import anndata
    import torch
    from sklearn.metrics import accuracy_score, f1_score
    tutorial, pretrain = Path(args.tutorial_dir), Path(args.pretrain_dir)
    root = Path(args.run_dir) / "cellnavi" / "repro"
    root.mkdir(parents=True, exist_ok=False)
    hashes = {"pretrain": _md5(pretrain / "pretrain_weights.pth"),
              "tutorial_checkpoint": _md5(tutorial / "checkpoint-step-1000.pth")}
    if hashes != {"pretrain": cc.CELLNAVI_PRETRAIN_MD5, "tutorial_checkpoint": cc.CELLNAVI_TUTORIAL_CKPT_MD5}:
        raise RuntimeError(f"released asset hash mismatch: {hashes}")
    files = {n: tutorial / n for n in ("Re-stimulated_t_example_train.h5ad", "Resting_t_example_test.h5ad",
                                       "dist_t_matrix.csv", "adj_t_matrix.csv")}
    files["gene_name.txt"] = pretrain / "gene_name.txt"
    files["pretrain/pretrain_weights.pth"] = pretrain / "pretrain_weights.pth"
    files["finetune/model/checkpoint-step-1000.pth"] = tutorial / "checkpoint-step-1000.pth"
    data = _workspace(root, files, "Re-stimulated_t_example_train.h5ad", "Resting_t_example_test.h5ad")
    train_categories = anndata.read_h5ad(data / "Re-stimulated_t_example_train.h5ad", backed="r").obs[
        "perturbation"].astype("category").cat.categories.tolist()
    model = _load(data / "finetune" / "model", 1000)
    dataset, loader = _loader(args.workers)
    predicted = []
    with torch.no_grad():
        for batch in loader:
            logits = model({k: v.cuda() for k, v in batch.items()})["pred"][0, :len(train_categories)]
            predicted.append(train_categories[int(logits.argmax())])
    truth = anndata.read_h5ad(data / "Resting_t_example_test.h5ad", backed="r").obs["perturbation"].astype(str).to_numpy()
    accuracy = float(accuracy_score(truth, predicted))
    f1 = float(f1_score(truth, predicted, average="weighted"))
    target = cc.CELLNAVI_REPRO_TARGET
    record = {"stage": "CELLNAVI_OFFICIAL_REPRODUCTION", "commit": cc.CELLNAVI_COMMIT, "hashes": hashes,
              "cells": len(predicted), "classes": len(train_categories), "accuracy": accuracy, "weighted_f1": f1,
              "target": target, "tolerance": cc.CELLNAVI_REPRO_TOLERANCE,
              "passes": bool(len(predicted) == target["cells"] and len(train_categories) == target["classes"]
                             and abs(accuracy - target["accuracy"]) <= cc.CELLNAVI_REPRO_TOLERANCE
                             and abs(f1 - target["weighted_f1"]) <= cc.CELLNAVI_REPRO_TOLERANCE),
              "route": "notebook gene-name route: argmax over the train file's categories, compared by name",
              "torch": torch.__version__, "timestamp": datetime.now(timezone.utc).isoformat()}
    record["verdict"] = "CELLNAVI_OFFICIAL_REPRODUCTION_PASSED" if record["passes"] else "CELLNAVI_OFFICIAL_REPRODUCTION_FAILED"
    (root / "reproduction.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def _fit_files(args: argparse.Namespace) -> dict:
    data_dir, tutorial, pretrain = Path(args.data_dir), Path(args.tutorial_dir), Path(args.pretrain_dir)
    return {"train.h5ad": data_dir / "train.h5ad", "placeholder.h5ad": data_dir / "placeholder.h5ad",
            "query.h5ad": data_dir / "query.h5ad", "dist_t_matrix.csv": tutorial / "dist_t_matrix.csv",
            "adj_t_matrix.csv": tutorial / "adj_t_matrix.csv", "gene_name.txt": pretrain / "gene_name.txt",
            "pretrain/pretrain_weights.pth": pretrain / "pretrain_weights.pth"}


def train(args: argparse.Namespace) -> None:
    import torch
    classes = json.loads((Path(args.data_dir) / "classes.json").read_text())
    root = Path(args.run_dir) / args.subdir / args.held
    resuming = args.resume and any((root / "data" / "finetune" / "model").glob("checkpoint-step-*.pth"))
    root.mkdir(parents=True, exist_ok=resuming)   # the official trainer resumes from its latest checkpoint
    _workspace(root, _fit_files(args), "train.h5ad", "placeholder.h5ad")
    seed = (cc.GEARS_SEED + 100 + cc.PRIMARY_CONTEXTS.index(args.held) + (1000 if resuming else 0)
            + (200 if args.subdir != "cellnavi" else 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    _patch_classes(len(classes))
    _patch_io()
    os.environ.update({"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1", "MASTER_ADDR": "127.0.0.1",
                       "MASTER_PORT": str(29500 + cc.PRIMARY_CONTEXTS.index(args.held) + (10 if args.subdir != "cellnavi" else 0))})
    from cellnavi.trainer import Trainer
    torch.distributed.init_process_group(backend="nccl", init_method=f"tcp://127.0.0.1:{os.environ['MASTER_PORT']}",
                                         world_size=1, rank=0)
    torch.cuda.set_device(0)
    with Trainer(0, 0) as trainer:
        trainer.train()
    torch.distributed.destroy_process_group()
    (root / "train_record.json").write_text(json.dumps(
        {"held": args.held, "classes": len(classes), "seed": seed, "resumed": bool(resuming),
         "variant": args.subdir, "recipe": cc.CELLNAVI_RECIPE,
         "commit": cc.CELLNAVI_COMMIT, "deviation": "class head width set to the number of training classes",
         "io_adapter": "int8 np.ix_ distance-mask gather, identical values, self-checked",
         "timestamp": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")


def infer(args: argparse.Namespace) -> dict:
    import anndata
    import torch
    classes = json.loads((Path(args.data_dir) / "classes.json").read_text())
    root = Path(args.run_dir) / args.subdir / args.held
    data = _workspace(root, _fit_files(args), "train.h5ad", "query.h5ad")
    output = root / "scores.npz"
    if output.exists():
        raise FileExistsError(output)
    _patch_classes(len(classes))
    _patch_io()
    model = _load(data / "finetune" / "model", cc.CELLNAVI_EVAL_STEP)
    obs = anndata.read_h5ad(data / "query.h5ad", backed="r").obs
    keys = list(zip(obs["query"].astype(str).tolist(), obs["view"].astype(int).tolist()))
    rows = sorted(set(keys), key=lambda k: (k[1], k[0]))
    index = {k: i for i, k in enumerate(rows)}
    total = np.zeros((len(rows), len(classes)), dtype=np.float64)
    count = np.zeros(len(rows), dtype=np.int64)
    top1 = []
    dataset, loader = _loader(args.workers)
    if len(dataset) != len(keys):
        raise RuntimeError("the official dataset did not keep every query cell")
    with torch.no_grad():
        for position, batch in enumerate(loader):
            logits = model({k: v.cuda() for k, v in batch.items()})["pred"][0].float()
            logp = torch.log_softmax(logits, dim=0).cpu().numpy().astype(np.float64)
            row = index[keys[position]]
            total[row] += logp
            count[row] += 1
            top1.append(int(np.argmax(logp)))
    np.savez(output, queries=np.asarray([k[0] for k in rows]), views=np.asarray([k[1] for k in rows]),
             classes=np.asarray(classes), mean_log_prob=total / count[:, None], cells=count,
             cell_top1=np.asarray(top1), cell_query=np.asarray([k[0] for k in keys]),
             cell_view=np.asarray([k[1] for k in keys]))
    record = {"held": args.held, "variant": args.subdir, "rows": len(rows), "cells": int(count.sum()), "classes": len(classes),
              "checkpoint_step": cc.CELLNAVI_EVAL_STEP, "score": cc.CELLNAVI_SCORE,
              "timestamp": datetime.now(timezone.utc).isoformat()}
    (root / "infer_record.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="CellNavi adapter")
    parser.add_argument("--stage", choices=("repro", "train", "infer"), required=True)
    parser.add_argument("--run-dir", dest="run_dir", required=True)
    parser.add_argument("--tutorial-dir", dest="tutorial_dir", required=True)
    parser.add_argument("--pretrain-dir", dest="pretrain_dir", required=True)
    parser.add_argument("--code-dir", dest="code_dir", required=True, help="the official clone at CELLNAVI_COMMIT")
    parser.add_argument("--data-dir", dest="data_dir")
    parser.add_argument("--held", choices=cc.PRIMARY_CONTEXTS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="continue from the latest official checkpoint")
    parser.add_argument("--subdir", default="cellnavi", help="cellnavi, or cellnavi_native for Deviation 5")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.code_dir)))
    record = {"repro": repro, "train": train, "infer": infer}[args.stage](args)
    if record:
        print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
