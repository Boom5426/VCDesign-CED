"""A1 anchors for the anchored arms (protocol Deviation 2).

Held and validation candidates are anchored by the frozen A1 recipe fitted on the whole
training part.  Training rows are anchored out of fold: each identity's anchor comes from an
A1 fit, with its own basis and the part's selected penalty, on the identities of the other
four anchor folds.  So the correction network never learns against a ridge that has seen the
row it is correcting.
"""
from __future__ import annotations

import hashlib

import numpy as np

from ..candidate_effect_distillation_v1 import basis as bs
from ..four_context_v1 import predictors as pd
from . import contract as pc
from . import data as dt


def oof_folds(identities: np.ndarray) -> np.ndarray:
    digest = [hashlib.sha256(f"{pc.ANCHOR_OOF_NAMESPACE}:{value}".encode()).digest()
              for value in np.asarray(identities, dtype=str).tolist()]
    return np.asarray([int.from_bytes(value[:8], "big") % pc.ANCHOR_OOF_FOLDS for value in digest], dtype=np.int64)


def _keys(pool: dt.Pool) -> np.ndarray:
    return np.asarray([f"{c}|{pool.identities[i]}" for c, i in zip(pool.row_context, pool.row_identity)])


def fit_a1(pool: dt.Pool):
    """The frozen A1 recipe on a pool's rows, penalty chosen by its own cross validation."""
    return pd.fit(pool.features[pool.row_identity], pool.unit.astype(np.float64), np.ones(pool.rows, dtype=bool),
                  _keys(pool), pool.contexts)


def training_anchors(pool: dt.Pool, penalty: float) -> np.ndarray:
    """Out-of-fold A1 effect for every identity of ``pool``, in its identity order."""
    folds = oof_folds(pool.identities)
    output = np.zeros((pool.identities.size, pool.unit.shape[1]), dtype=np.float32)
    for fold in range(pc.ANCHOR_OOF_FOLDS):
        held = folds == fold
        part = dt.restrict(pool, ~held)
        basis = bs.randomized_basis(part.unit.astype(np.float64))
        path = pd.WeightedRidgePath(part.features[part.row_identity], basis.project(part.unit.astype(np.float64)))
        output[held] = basis.reconstruct(path.predict(pool.features[held], penalty)).astype(np.float32)
    return output
