"""Training pools for PPM, built from the frozen four-context packs by the frozen rules.

A pool is the ``fit_mask`` rows of the training contexts, minus every row whose identity is
masked, keeping only rows with a legal modality: the rows A1 fits on, in A1's order.  Each
row keeps both measured views, because the tokenizer's cross-view term needs them.  Every
derived quantity (standardization, the program initialization basis, the common direction)
is computed from whatever rows a fit is allowed to train on, never from a wider set.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from ..candidate_effect_distillation_v1 import basis as bs
from ..rwed_v1 import estimator as es
from . import contract as pc


@dataclass(frozen=True)
class Pool:
    """Training rows grouped by identity."""

    identities: np.ndarray        # [n_id] str, sorted
    features: np.ndarray          # [n_id, F] raw feature contract, identical across contexts
    row_identity: np.ndarray      # [n_rows] index into identities
    row_context: np.ndarray       # [n_rows] str
    unit: np.ndarray              # [n_rows, G] float32, unit consensus direction (A1's target)
    view_a: np.ndarray            # [n_rows, G] float32, unit view 0
    view_b: np.ndarray            # [n_rows, G] float32, unit view 1
    contexts: tuple

    @property
    def rows(self) -> int:
        return int(self.row_identity.size)


def _unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1)
    if (norms <= pc.ZERO_NORM).any():
        raise ValueError(f"{int((norms <= pc.ZERO_NORM).sum())} training rows have no direction")
    return values / norms[:, None]


def assemble(packs: dict, training: tuple, excluded=()) -> Pool:
    """The present training rows of ``training`` minus every row of an ``excluded`` identity."""
    raw = es.views_pool(packs, training, excluded)
    keep = np.flatnonzero(np.asarray(raw["present"], dtype=bool))
    names = np.asarray([key.split("|", 1)[1] for key in raw["keys"][keep].tolist()], dtype=str)
    identities, first, inverse = np.unique(names, return_index=True, return_inverse=True)
    features = np.asarray(raw["features"], dtype=np.float64)[keep]
    if not np.array_equal(features, features[first][inverse]):
        raise ValueError("an identity carries different static features in different contexts")
    views = raw["views"][:, keep]
    return Pool(identities=identities, features=features[first], row_identity=inverse.astype(np.int64),
                row_context=np.asarray(raw["context_of_row"])[keep].astype(str),
                unit=_unit(raw["responses"][keep]).astype(np.float32),
                view_a=_unit(views[0]).astype(np.float32), view_b=_unit(views[1]).astype(np.float32),
                contexts=tuple(training))


def restrict(pool: Pool, keep_identity: np.ndarray) -> Pool:
    """The pool with only the identities flagged in ``keep_identity``."""
    keep_identity = np.asarray(keep_identity, dtype=bool)
    if keep_identity.shape != pool.identities.shape:
        raise ValueError("one flag per identity is required")
    new_index = np.cumsum(keep_identity) - 1
    rows = np.flatnonzero(keep_identity[pool.row_identity])
    return Pool(identities=pool.identities[keep_identity], features=pool.features[keep_identity],
                row_identity=new_index[pool.row_identity[rows]].astype(np.int64),
                row_context=pool.row_context[rows], unit=pool.unit[rows],
                view_a=pool.view_a[rows], view_b=pool.view_b[rows], contexts=pool.contexts)


def identity_directions(pool: Pool) -> np.ndarray:
    """``u_bar_j``: the unit mean of an identity's unit consensus directions."""
    total = np.zeros((pool.identities.size, pool.unit.shape[1]), dtype=np.float64)
    np.add.at(total, pool.row_identity, pool.unit.astype(np.float64))
    return _unit(total).astype(np.float32)


def validation_mask(identities: np.ndarray) -> np.ndarray:
    """Inner-validation identities, from the symbol alone."""
    digest = [hashlib.sha256(f"{pc.VALIDATION_NAMESPACE}:{value}".encode()).digest()
              for value in np.asarray(identities, dtype=str).tolist()]
    return np.asarray([int.from_bytes(value[:8], "big") % pc.VALIDATION_MODULUS == 0
                       for value in digest], dtype=bool)


class Standardizer:
    """Feature centring and scaling from training rows, exactly as the frozen ridge does it."""

    def __init__(self, pool: Pool):
        rows = pool.features[pool.row_identity]
        self.mean = rows.mean(axis=0)
        self.scale = np.maximum(rows.std(axis=0), 1e-8)
        self.flag_columns = slice(sum(pc.MODALITY_WIDTHS), sum(pc.MODALITY_WIDTHS) + len(pc.MODALITY_WIDTHS))

    def transform(self, features: np.ndarray) -> np.ndarray:
        return ((np.asarray(features, dtype=np.float64) - self.mean) / self.scale).astype(np.float32)

    def presence(self, features: np.ndarray) -> np.ndarray:
        """The raw presence flags, which gate the modality adapters."""
        return np.asarray(features, dtype=np.float64)[:, self.flag_columns].astype(np.float32)


def program_basis(pool: Pool):
    """A1's rank-256 basis of this pool's unit targets: the program initialization."""
    return bs.randomized_basis(pool.unit.astype(np.float64))


def common_direction(pool: Pool, basis) -> np.ndarray:
    """A1's unit intercept on this pool (the C-018 common-direction definition)."""
    target_mean = basis.project(pool.unit.astype(np.float64)).mean(axis=0)
    direction = basis.reconstruct(target_mean[None, :])[0]
    return direction / np.linalg.norm(direction)
