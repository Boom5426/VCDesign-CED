"""The RWED estimator: A1 rotated into a common axis and candidate-specific axes, with a
per-axis ridge penalty set by training-only cross-view reproducibility.

Everything here is a pure function of training rows.  The held context enters nothing.
The estimator's output is a predicted effect vector; how it is scored is not decided here,
and nothing in this module produces a per-candidate reliability, weight or confidence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..candidate_effect_distillation_v1 import contract as ced
from ..candidate_effect_distillation_v1 import predictor as pr
from ..four_context_v1 import evaluate as ev
from ..four_context_v1 import predictors as pd
from . import contract as rc

ZERO_NORM = 1e-12


def views_pool(packs: dict, names: tuple[str, ...], excluded=()) -> dict:
    """Training rows of ``names`` with both measured views kept.

    The rows, their order and the masking rule are those of ``harness.training_pool`` and
    ``masking.masked_training_pool``: each context's ``fit_mask`` rows in order, minus every
    row whose identity is excluded.  The consensus response is the frozen
    ``evaluate.consensus`` of the two views, so the run can assert it equals the frozen pool.
    """
    excluded = set(np.asarray(excluded, dtype=str).tolist())
    features, views, present, keys, context = [], [], [], [], []
    for name in names:
        pack = packs[name]
        rows = np.flatnonzero(pack.fit_mask)
        rows = rows[np.asarray([value not in excluded for value in pack.ids[rows].tolist()], dtype=bool)]
        features.append(pack.features[rows])
        views.append(np.asarray(pack.responses[:, rows], dtype=np.float64))
        present.append(pack.present[rows])
        keys.append(np.asarray([f"{name}|{value}" for value in pack.ids[rows].tolist()], dtype=str))
        context.append(np.full(rows.size, name))
    views = np.concatenate(views, axis=1)
    return {"features": np.concatenate(features), "views": views,
            "responses": ev.consensus(views), "present": np.concatenate(present),
            "keys": np.concatenate(keys), "context_of_row": np.concatenate(context),
            "contexts": tuple(names)}


def unit_rows(values: np.ndarray, rows: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1)
    if (norms[rows] <= ZERO_NORM).any():
        raise ValueError("a present training row has a zero response and no direction")
    return values / np.maximum(norms, ZERO_NORM)[:, None]


class RotatedBasis:
    """A1's response basis read in rotated coordinates: ``reconstruct(t) = B (t Q^T)``."""

    def __init__(self, basis, rotation: np.ndarray):
        self.inner = basis
        self.rotation = np.asarray(rotation, dtype=np.float64)

    @property
    def explained(self) -> float:
        return self.inner.explained

    def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
        return self.inner.reconstruct(np.asarray(coefficients, dtype=np.float64) @ self.rotation.T)


def rotation(targets: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    """``Q = [a, W]`` in A1's coefficient space.

    ``a`` is the unit intercept.  ``W`` are the principal axes of the targets once ``a`` is
    removed, from an exact SVD, so the residual axes are ordered by how much candidate
    variation they carry.  A final QR guards orthonormality; its signs are fixed so that
    the first column is ``a`` itself.
    """
    targets = np.asarray(targets, dtype=np.float64)
    common = np.asarray(intercept, dtype=np.float64)
    common = common / np.linalg.norm(common)
    residual = targets - np.outer(targets @ common, common)
    _, _, right = np.linalg.svd(residual, full_matrices=False)
    axes = right[: targets.shape[1] - 1].T
    q, r = np.linalg.qr(np.column_stack([common, axes]))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs[None, :]


def axis_reliability(view_a: np.ndarray, view_b: np.ndarray, contexts: np.ndarray,
                     axes: np.ndarray) -> np.ndarray:
    """Per-axis Pearson correlation of the two views, centred within each context."""
    first = np.asarray(view_a, dtype=np.float64) @ axes
    second = np.asarray(view_b, dtype=np.float64) @ axes
    contexts = np.asarray(contexts)
    for name in np.unique(contexts):
        inside = contexts == name
        first[inside] -= first[inside].mean(axis=0)
        second[inside] -= second[inside].mean(axis=0)
    denominator = np.sqrt((first * first).sum(axis=0) * (second * second).sum(axis=0))
    return (first * second).sum(axis=0) / np.maximum(denominator, 1e-300)


def multipliers(rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``m_common = 1``, ``m_k = Rbar / R_k`` for the residual axes, ``R = 2 rho / (1 + rho)``."""
    clipped = np.clip(np.asarray(rho, dtype=np.float64), rc.RHO_FLOOR, 1.0)
    consensus = 2.0 * clipped / (1.0 + clipped)
    output = np.ones_like(consensus)
    residual = consensus[1:]
    output[1:] = np.exp(np.log(residual).mean()) / residual
    return output, consensus


def column_predict(path, features: np.ndarray, penalties: np.ndarray) -> np.ndarray:
    """Ridge prediction of every target column with its own penalty, from one SVD."""
    singular = path.singular[:, None]
    scale = singular / (singular ** 2 + np.asarray(penalties, dtype=np.float64)[None, :])
    weights = path.right.T @ (scale * path.projected)
    centred = (np.asarray(features, dtype=np.float64) - path.feature_mean) / path.feature_scale
    return centred @ weights + path.target_mean


def select_penalty(features: np.ndarray, targets: np.ndarray, truth: np.ndarray, basis,
                   penalty_multipliers: np.ndarray) -> dict:
    """The frozen selection procedure with per-axis multipliers inside it.

    Same grid, same five row-index folds, same criterion as
    ``four_context_v1.predictors._cross_validate``: the mean held-out gene-space cosine of the
    reconstructed prediction against the unit target.  With every multiplier 1 it returns that
    function's curve; a test asserts it.
    """
    folds = pr.identity_folds(features.shape[0])
    cosine = {penalty: [] for penalty in ced.RIDGE_PENALTIES}
    for held in folds:
        train = np.setdiff1d(np.arange(features.shape[0]), held)
        path = pd.WeightedRidgePath(features[train], targets[train], np.ones(train.size))
        for penalty in ced.RIDGE_PENALTIES:
            reconstructed = basis.reconstruct(column_predict(path, features[held],
                                                             penalty * penalty_multipliers))
            norm = np.linalg.norm(reconstructed, axis=1) * np.linalg.norm(truth[held], axis=1)
            cosine[penalty].append(np.einsum("ij,ij->i", reconstructed, truth[held])
                                   / np.maximum(norm, 1e-30))
    record = {"held_out_cosine": {str(p): float(np.concatenate(v).mean()) for p, v in cosine.items()}}
    record["selected_penalty"] = float(max(ced.RIDGE_PENALTIES,
                                           key=lambda p: record["held_out_cosine"][str(p)]))
    record["held_out_cosine_at_selected"] = record["held_out_cosine"][str(record["selected_penalty"])]
    return record


@dataclass(frozen=True)
class RWEDPredictor:
    basis: RotatedBasis
    path: object
    penalty: float
    multipliers: np.ndarray
    rows: int
    identities: int
    cross_validation: dict
    reliability: dict

    def effect(self, features: np.ndarray, present: np.ndarray) -> np.ndarray:
        predicted = self.basis.reconstruct(column_predict(self.path, features,
                                                          self.penalty * self.multipliers))
        predicted[~np.asarray(present, dtype=bool)] = 0.0
        return predicted


def build(pool: dict, a1, penalty_multipliers: np.ndarray | None = None) -> RWEDPredictor:
    """RWED on one training pool, anchored on that pool's own A1 fit.

    ``penalty_multipliers`` exists only so the tests can force every multiplier to 1 and
    check that the estimator is then A1; the run never passes it.
    """
    keep = np.flatnonzero(np.asarray(pool["present"], dtype=bool))
    unit = unit_rows(pool["responses"], keep)[keep]
    view_a = unit_rows(pool["views"][0], keep)[keep]
    view_b = unit_rows(pool["views"][1], keep)[keep]
    basis = a1.basis
    targets = basis.project(unit)
    turn = rotation(targets, a1.path.target_mean)
    rotated = targets @ turn
    contexts = np.asarray(pool["context_of_row"])[keep]
    rho = axis_reliability(basis.project(view_a), basis.project(view_b), contexts, turn)
    rho_full = axis_reliability(basis.project(view_a), basis.project(view_b), contexts,
                                np.eye(turn.shape[0]))
    computed, consensus = multipliers(rho)
    used = computed if penalty_multipliers is None else np.asarray(penalty_multipliers, dtype=np.float64)
    rotated_basis = RotatedBasis(basis, turn)
    record = select_penalty(pool["features"][keep], rotated, unit, rotated_basis, used)
    path = pd.WeightedRidgePath(pool["features"][keep], rotated, np.ones(keep.size))
    identities = np.unique([key.split("|", 1)[1] for key in np.asarray(pool["keys"])[keep].tolist()])
    return RWEDPredictor(
        basis=rotated_basis, path=path, penalty=float(record["selected_penalty"]),
        multipliers=used, rows=int(keep.size), identities=int(identities.size),
        cross_validation=record,
        reliability={"rho": rho, "rho_full_basis": rho_full, "consensus_reliability": consensus,
                     "rotation": turn})
