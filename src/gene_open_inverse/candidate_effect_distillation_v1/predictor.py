"""Static knowledge to effect, as a linear map chosen entirely inside G_fit.

The map is a multi-output ridge onto the response basis.  It is deliberately the
simplest thing that can work: if a linear map from STRING and MAPKG carries design
signal, that is the finding, and if it does not, a larger model would only make the
negative harder to attribute.

Every choice that could be tuned is made by identity-level cross validation on
``G_fit``.  A candidate never appears in both the training and validation side of a
fold, and no ``G_select`` quantity is consulted at any point.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import contract as ced


@dataclass(frozen=True)
class StaticFeatures:
    """Concatenated legal descriptors with their present flags."""

    values: np.ndarray
    present_any: np.ndarray
    names: tuple[str, ...]

    def fit_rows(self) -> np.ndarray:
        return np.flatnonzero(self.present_any)


def static_features(role, modalities: tuple[str, ...]) -> StaticFeatures:
    """Zero-filled descriptor blocks plus one present flag per modality."""
    blocks, flags = [], []
    for name in modalities:
        values = np.asarray(role.capabilities[name], dtype=np.float64)
        present = np.asarray(role.capability_present[name], dtype=bool)
        blocks.append(values * present[:, None])
        flags.append(present.astype(np.float64))
    stacked = np.concatenate(blocks + [np.stack(flags, axis=1)], axis=1)
    return StaticFeatures(values=stacked, present_any=np.stack(flags, axis=1).any(axis=1),
                          names=tuple(modalities))


class RidgePath:
    """One SVD, every penalty for free."""

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.feature_mean = features.mean(axis=0)
        self.feature_scale = np.maximum(features.std(axis=0), 1e-8)
        self.target_mean = targets.mean(axis=0)
        centred = (features - self.feature_mean) / self.feature_scale
        self.left, self.singular, self.right = np.linalg.svd(centred, full_matrices=False)
        self.projected = self.left.T @ (targets - self.target_mean)

    def weights(self, penalty: float) -> np.ndarray:
        scale = self.singular / (self.singular ** 2 + penalty)
        return self.right.T @ (scale[:, None] * self.projected)

    def predict(self, features: np.ndarray, penalty: float) -> np.ndarray:
        centred = (np.asarray(features, dtype=np.float64) - self.feature_mean) / self.feature_scale
        return centred @ self.weights(penalty) + self.target_mean


def identity_folds(count: int, folds: int = ced.RIDGE_FOLDS,
                   seed: int = ced.RIDGE_FOLD_SEED) -> list[np.ndarray]:
    """Disjoint identity blocks.  Each candidate is one identity, so this is exact."""
    generator = np.random.default_rng(seed)
    order = generator.permutation(count)
    return [np.sort(part) for part in np.array_split(order, folds)]


def _gene_space_cosine(coefficients: np.ndarray, basis, truth: np.ndarray) -> np.ndarray:
    predicted = basis.reconstruct(coefficients)
    norm = np.linalg.norm(predicted, axis=1) * np.linalg.norm(truth, axis=1)
    return np.einsum("ij,ij->i", predicted, truth) / np.maximum(norm, 1e-30)


def cross_validate(features: np.ndarray, targets: np.ndarray, truth: np.ndarray, basis,
                   penalties: tuple[float, ...] = ced.RIDGE_PENALTIES) -> dict:
    """Held-out cosine and squared error for every penalty, by identity fold."""
    folds = identity_folds(features.shape[0])
    cosine = {penalty: [] for penalty in penalties}
    squared = {penalty: [] for penalty in penalties}
    for held in folds:
        train = np.setdiff1d(np.arange(features.shape[0]), held)
        path = RidgePath(features[train], targets[train])
        for penalty in penalties:
            predicted = path.predict(features[held], penalty)
            cosine[penalty].append(_gene_space_cosine(predicted, basis, truth[held]))
            squared[penalty].append(((predicted - targets[held]) ** 2).mean(axis=1))
    record = {
        "folds": len(folds), "fold_sizes": [int(part.size) for part in folds],
        "penalties": list(penalties),
        "held_out_cosine": {str(penalty): float(np.concatenate(values).mean())
                            for penalty, values in cosine.items()},
        "held_out_squared_error": {str(penalty): float(np.concatenate(values).mean())
                                   for penalty, values in squared.items()},
    }
    record["selected_penalty"] = float(max(penalties, key=lambda p: record["held_out_cosine"][str(p)]))
    record["selected_penalty_by_squared_error"] = float(
        min(penalties, key=lambda p: record["held_out_squared_error"][str(p)]))
    record["held_out_cosine_at_selected"] = record["held_out_cosine"][str(record["selected_penalty"])]
    distribution = np.concatenate(cosine[record["selected_penalty"]])
    record["held_out_cosine_distribution"] = {
        "median": float(np.median(distribution)), "mean": float(distribution.mean()),
        "p05": float(np.quantile(distribution, 0.05)), "p95": float(np.quantile(distribution, 0.95)),
        "fraction_non_positive": float((distribution <= 0).mean())}
    return record
