"""Single-cell access shared by the GEARS and CellNavi adapters.

Every selection is deterministic (SHA-256 of a namespaced cell key) and label-driven: rows are chosen
from the ``obs`` identity column first, and only the chosen rows' expression is ever read.  A K562
identity outside the anchor's ``G_fit`` role is never loaded, so no ``G_check`` expression is read.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import anndata
import numpy as np

from . import contract as cc

CHUNK = 8192


def cell_key_rank(keys: list[str], namespace: str) -> np.ndarray:
    """Ascending rank order of ``SHA-256(namespace:key)``; the first ``n`` are the selection."""
    digests = [hashlib.sha256(f"{namespace}:{key}".encode()).digest()[:8] for key in keys]
    return np.argsort(np.asarray([int.from_bytes(d, "big") for d in digests], dtype=np.uint64), kind="stable")


def hash_bucket(value: str, namespace: str, modulus: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{namespace}:{value}".encode()).digest()[:8], "big") % modulus


class Source:
    """One context's raw single-cell file, opened backed and read by selected rows only."""

    def __init__(self, context: str):
        spec = cc.SOURCES[context]
        self.context = context
        self.path = spec["path"]
        self.handle = anndata.read_h5ad(spec["path"], backed="r")
        self.labels = self.handle.obs[spec["perturbation"]].astype(str).to_numpy()
        self.batches = self.handle.obs[spec["batch"]].astype(str).to_numpy()
        self.control = spec["control"]
        self.names = np.asarray(self.handle.obs_names, dtype=str)
        genes = (self.handle.var[spec["symbol"]].astype(str).to_numpy() if spec["symbol"]
                 else np.asarray(self.handle.var_names, dtype=str))
        symbols, counts = np.unique(genes, return_counts=True)
        duplicated = set(symbols[counts > 1].tolist())
        self.gene_keep = np.asarray([g not in duplicated for g in genes.tolist()], dtype=bool)
        self.genes = genes

    def select(self, identity: str, cap: int, namespace: str, rows: np.ndarray | None = None) -> np.ndarray:
        """Up to ``cap`` rows of one identity (or of ``rows`` when given), by hashed cell key."""
        candidates = np.flatnonzero(self.labels == identity) if rows is None else np.asarray(rows)
        if candidates.size <= cap:
            return np.sort(candidates)
        keys = [f"{self.context}|{self.names[r]}" for r in candidates.tolist()]
        return np.sort(candidates[cell_key_rank(keys, namespace)[:cap]])

    def counts(self, rows: np.ndarray) -> np.ndarray:
        """Raw counts of sorted rows, full gene axis, float32."""
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size and np.any(np.diff(rows) <= 0):
            raise ValueError("rows must be strictly increasing")
        blocks = []
        for start in range(0, rows.size, CHUNK):
            block = self.handle.X[rows[start:start + CHUNK]]
            blocks.append(np.asarray(block.toarray() if hasattr(block, "toarray") else block, dtype=np.float32))
        return np.concatenate(blocks) if blocks else np.zeros((0, self.genes.size), dtype=np.float32)

    def lognorm(self, rows: np.ndarray, genes: np.ndarray) -> np.ndarray:
        """log1p(CP10k) with the library taken over the full row, then restricted to ``genes``.

        Same normalization as ``four_context_v1.build``: the denominator is a property of the cell.
        """
        values = self.counts(rows)
        library = values.sum(axis=1)
        if not np.all(np.isfinite(library) & (library > 0)):
            raise ValueError(f"{self.context}: a selected cell has no positive library")
        values = np.log1p(values / library[:, None] * cc.LIBRARY_TARGET)
        position = {g: i for i, g in enumerate(self.genes.tolist()) if self.gene_keep[i]}
        return np.ascontiguousarray(values[:, [position[g] for g in np.asarray(genes, dtype=str).tolist()]])


def view_side(batch: str) -> int:
    """The frozen A/B bit of ``four_context_v1.build`` (imported lazily to keep this module light)."""
    from ..four_context_v1.build import view_side as frozen
    return frozen(batch)


def load_pack_axes(four_context_run: str, context: str) -> dict:
    """Identity axis, training mask and queries of one cached pack, without the heavy arrays."""
    with np.load(Path(four_context_run) / "packs" / f"{context}.npz", allow_pickle=False) as store:
        return {"ids": store["ids"].astype(str), "fit_mask": store["fit_mask"].copy(),
                "self_positions": store["self_positions"].copy(), "present": store["present"].copy()}


def g4(four_context_run: str) -> np.ndarray:
    return np.load(Path(four_context_run) / "genes" / "G_4.npy", allow_pickle=True).astype(str)
