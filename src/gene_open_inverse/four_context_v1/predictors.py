"""The V1 recipe on an arbitrary pool of ``(context, identity)`` rows.

Nothing about the recipe changes across folds, arms or atlas sizes: the same rank,
the same oversampling, the same power iterations, the same penalty grid and the same
identity-fold cross validation.  That is the point.  If a pooled atlas wins, it has
to win at fixed method.

Row weighting.  A candidate measured in three contexts contributes three rows, so a
frequently measured candidate has three times the leverage of a rarely measured one.
The protocol asks for both readings, so the primary fit keeps every row at equal
weight and ``candidate_balanced_weights`` supplies the sensitivity in which each
candidate carries one unit of weight regardless of how many contexts measured it.
The choice between them is never made on a held-context result.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..candidate_effect_distillation_v1 import basis as bs
from ..candidate_effect_distillation_v1 import contract as ced
from ..candidate_effect_distillation_v1 import predictor as pr


class WeightedRidgePath:
    """One SVD, every penalty for free, with per-row weights.

    With equal weights this reduces to the frozen ``RidgePath`` exactly: the
    weighted mean becomes the mean, the weighted scale becomes the standard
    deviation, and the row scaling becomes a constant that cancels between the
    normal equations' two sides.  The test suite asserts that equivalence rather
    than assuming it.
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray, weights: np.ndarray | None = None):
        features = np.asarray(features, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        if weights is None:
            weights = np.ones(features.shape[0], dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (features.shape[0],) or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("weights must be one non-negative value per row with positive total")
        share = weights / weights.sum()
        self.feature_mean = share @ features
        centred = features - self.feature_mean
        self.feature_scale = np.maximum(np.sqrt(share @ (centred ** 2)), 1e-8)
        self.target_mean = share @ targets
        # Two different normalizations, deliberately.  The means and the scale use
        # weights that sum to one, so they are proper weighted moments.  The row
        # scaling uses weights whose *mean* is one, so that equal weights leave the
        # normal equations untouched and the penalty grid keeps the meaning it has in
        # the frozen RidgePath.  Dividing by the sum instead would multiply the
        # effective penalty by the row count and quietly make every fit a different
        # model from the one this programme claims to reuse.
        row = weights / weights.mean()
        scaled = (centred / self.feature_scale) * np.sqrt(row)[:, None]
        self.left, self.singular, self.right = np.linalg.svd(scaled, full_matrices=False)
        self.projected = self.left.T @ ((targets - self.target_mean) * np.sqrt(row)[:, None])

    def weights_at(self, penalty: float) -> np.ndarray:
        scale = self.singular / (self.singular ** 2 + penalty)
        return self.right.T @ (scale[:, None] * self.projected)

    def predict(self, features: np.ndarray, penalty: float) -> np.ndarray:
        centred = (np.asarray(features, dtype=np.float64) - self.feature_mean) / self.feature_scale
        return centred @ self.weights_at(penalty) + self.target_mean


@dataclass(frozen=True)
class EffectPredictor:
    """A fitted static-knowledge to gene-space-effect map."""

    basis: object
    path: WeightedRidgePath
    penalty: float
    rows: int
    identities: int
    contexts: tuple[str, ...]
    held_out_cosine: float
    cross_validation: dict

    def effect(self, features: np.ndarray, present: np.ndarray) -> np.ndarray:
        """``r_hat_c = B z_hat_c``.  A candidate with no legal modality gets exactly zero."""
        predicted = self.basis.reconstruct(self.path.predict(features, self.penalty))
        predicted[~np.asarray(present, dtype=bool)] = 0.0
        return predicted


def candidate_balanced_weights(keys: np.ndarray) -> np.ndarray:
    """One unit of weight per candidate identity, split across the contexts measuring it."""
    identities = np.asarray([key.split("|", 1)[1] for key in np.asarray(keys, dtype=str).tolist()])
    values, counts = np.unique(identities, return_counts=True)
    lookup = dict(zip(values.tolist(), counts.tolist()))
    return np.asarray([1.0 / lookup[name] for name in identities.tolist()], dtype=np.float64)


def _cross_validate(features: np.ndarray, targets: np.ndarray, truth: np.ndarray, basis,
                    weights: np.ndarray, penalties=ced.RIDGE_PENALTIES) -> dict:
    """Held-out cosine per penalty, by identity fold, weighted exactly as the final fit."""
    folds = pr.identity_folds(features.shape[0])
    cosine = {penalty: [] for penalty in penalties}
    squared = {penalty: [] for penalty in penalties}
    for held in folds:
        train = np.setdiff1d(np.arange(features.shape[0]), held)
        path = WeightedRidgePath(features[train], targets[train], weights[train])
        for penalty in penalties:
            predicted = path.predict(features[held], penalty)
            reconstructed = basis.reconstruct(predicted)
            norm = np.linalg.norm(reconstructed, axis=1) * np.linalg.norm(truth[held], axis=1)
            cosine[penalty].append(np.einsum("ij,ij->i", reconstructed, truth[held])
                                   / np.maximum(norm, 1e-30))
            squared[penalty].append(((predicted - targets[held]) ** 2).mean(axis=1))
    record = {
        "folds": len(folds), "fold_sizes": [int(part.size) for part in folds],
        "held_out_cosine": {str(p): float(np.concatenate(v).mean()) for p, v in cosine.items()},
        "held_out_squared_error": {str(p): float(np.concatenate(v).mean()) for p, v in squared.items()},
    }
    record["selected_penalty"] = float(max(penalties, key=lambda p: record["held_out_cosine"][str(p)]))
    record["selected_penalty_by_squared_error"] = float(
        min(penalties, key=lambda p: record["held_out_squared_error"][str(p)]))
    record["held_out_cosine_at_selected"] = record["held_out_cosine"][str(record["selected_penalty"])]
    return record


def fit(features: np.ndarray, responses: np.ndarray, present: np.ndarray, keys: np.ndarray,
        contexts: tuple[str, ...], rank: int = ced.BASIS_RANK,
        weights: np.ndarray | None = None) -> EffectPredictor:
    """Fit the V1 recipe on one training pool.  Reads no held-context response."""
    keep = np.flatnonzero(np.asarray(present, dtype=bool))
    if keep.size <= rank:
        raise ValueError(f"pool of {keep.size} usable rows cannot support a rank-{rank} basis")
    basis = bs.randomized_basis(responses[keep], rank=rank)
    targets = basis.project(responses[keep])
    row_weights = np.ones(keep.size) if weights is None else np.asarray(weights)[keep]
    record = _cross_validate(features[keep], targets, responses[keep], basis, row_weights)
    penalty = float(record["selected_penalty"])
    identities = np.unique([key.split("|", 1)[1] for key in np.asarray(keys, dtype=str)[keep].tolist()])
    return EffectPredictor(
        basis=basis, path=WeightedRidgePath(features[keep], targets, row_weights), penalty=penalty,
        rows=int(keep.size), identities=int(identities.size), contexts=tuple(contexts),
        held_out_cosine=float(record["held_out_cosine_at_selected"]), cross_validation=record)
