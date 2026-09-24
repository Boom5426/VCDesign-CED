"""The response basis and the response target, both from G_fit only.

``B`` spans the response space itself rather than deviations from the mean
response, which is the protocol's choice and is what makes ``r_hat = B z_hat`` a
complete reconstruction rather than an offset from one.  The consequence is that
the leading direction is the common perturbation response, so a predictor that
learns nothing candidate-specific will still produce a plausible-looking vector.
``common_direction_share`` exists to make that visible rather than flattering.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import contract as ced


@dataclass(frozen=True)
class ResponseBasis:
    """``B`` (8248 x K), its singular values, and what it explains."""

    basis: np.ndarray
    singular_values: np.ndarray
    total_energy: float

    @property
    def explained(self) -> float:
        return float((self.singular_values ** 2).sum() / self.total_energy)

    def project(self, responses: np.ndarray) -> np.ndarray:
        return np.asarray(responses, dtype=np.float64) @ self.basis

    def reconstruct(self, coefficients: np.ndarray) -> np.ndarray:
        return np.asarray(coefficients, dtype=np.float64) @ self.basis.T


def consensus_response(transitions: np.ndarray) -> np.ndarray:
    """``r_bar_c``, the mean of the two batch-disjoint view responses."""
    values = np.asarray(transitions, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != 2:
        raise ValueError("transitions must be [2, candidates, genes]")
    return 0.5 * (values[0] + values[1])


def view_reliability(transitions: np.ndarray) -> dict:
    """How reproducible a single candidate's response is across the two views."""
    values = np.asarray(transitions, dtype=np.float64)
    left, right = values[0], values[1]
    norm = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    cosine = np.einsum("ij,ij->i", left, right) / np.maximum(norm, 1e-30)
    return {"cross_view_cosine": {"median": float(np.median(cosine)), "mean": float(cosine.mean()),
                                  "p05": float(np.quantile(cosine, 0.05)),
                                  "p95": float(np.quantile(cosine, 0.95)),
                                  "fraction_non_positive": float((cosine <= 0).mean())}}


def randomized_basis(responses: np.ndarray, rank: int = ced.BASIS_RANK,
                     seed: int = ced.BASIS_SEED) -> ResponseBasis:
    """Randomized SVD of the G_fit consensus responses; the right factor is the basis."""
    values = np.asarray(responses, dtype=np.float64)
    rows, columns = values.shape
    if rank >= min(rows, columns):
        raise ValueError("rank must be smaller than both dimensions")
    generator = np.random.default_rng(seed)
    sketch = values @ generator.standard_normal((columns, rank + ced.BASIS_OVERSAMPLE))
    basis, _ = np.linalg.qr(sketch)
    for _ in range(ced.BASIS_POWER_ITERATIONS):
        basis, _ = np.linalg.qr(values.T @ basis)
        basis, _ = np.linalg.qr(values @ basis)
    small = basis.T @ values
    _, singular, right = np.linalg.svd(small, full_matrices=False)
    return ResponseBasis(basis=np.ascontiguousarray(right[:rank].T),
                         singular_values=np.ascontiguousarray(singular[:rank]),
                         total_energy=float((values * values).sum()))


def common_direction_share(responses: np.ndarray) -> dict:
    """How much of the response energy is one direction shared by every candidate.

    If the mean response explains most of the energy then any predictor, however
    weak, reconstructs something that looks like a response, and a cosine against
    it ranks candidates almost identically for every query.  This number is the
    context in which the design result must be read.
    """
    values = np.asarray(responses, dtype=np.float64)
    mean = values.mean(axis=0)
    unit = mean / max(float(np.linalg.norm(mean)), 1e-30)
    projected = values @ unit
    total = float((values * values).sum())
    norms = np.linalg.norm(values, axis=1)
    return {
        "share_of_response_energy_on_the_mean_direction": float((projected ** 2).sum() / total),
        "cosine_to_the_mean_direction": {
            "median": float(np.median(projected / np.maximum(norms, 1e-30))),
            "p05": float(np.quantile(projected / np.maximum(norms, 1e-30), 0.05)),
            "p95": float(np.quantile(projected / np.maximum(norms, 1e-30), 0.95))},
    }
