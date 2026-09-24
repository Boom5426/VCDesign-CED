"""Frozen Phase-1 eligibility rule and the Phase-2 latent candidate value.

Metric space
------------
The primary metric space is ``W = I``.  A noise-weighted ``W`` is deliberately
absent from this module: it is a later sensitivity analysis and must not touch a
primary verdict, because a ``W`` chosen while looking at downstream scores would
make the whole comparison circular.

Eligibility statistic
---------------------
A query is only a question if its transition is measurable at all.  The shared
latent transition energy

    E_q = <d_A^q, d_B^q>

is unbiased for ``||d*_q||^2`` because the two views are measured on disjoint
batches, so the noise cross term vanishes in expectation.  Its null needs no
extra computation: row ``q`` of the cross-view Gram already holds
``<d_A^q, d_B^j>`` for every mismatched partner ``j``, which is the same
statistic under the hypothesis that the identity pairing carries no signal.
Holding ``d_A^q`` fixed makes the null conditional on that identity's own
magnitude instead of pooling across identities of very different scale.

Candidate value
---------------
    V_qc = 2 <r*_c, d*_q> - ||r*_c||^2

is the reduction in squared endpoint distance that candidate ``c`` buys for
query ``q``, written without a denominator:

    V_qc = ||d*_q||^2 - ||r*_c - d*_q||^2.

The earlier no-op normalization divided by an estimate of ``||d*_q||^2``.  That
is a per-query constant, so it never changed a within-query ranking, but when
the estimate came out non-positive it flipped the sign of every utility in the
row and produced the catastrophic MBRU tail.  Dropping it removes that failure
mode by construction rather than by filtering.  nRU is invariant to a per-query affine
transform of the utility row with a POSITIVE multiplier, so this change leaves
the metric intact.  A negative multiplier reverses the ordering, which is exactly
what the old denominator did whenever the energy estimate came out negative.  The
invariance is exact only while the nRU denominator stays above the contract's
absolute floor of 1e-8; the smallest denominator observed in Phase 2 was 0.038
for the cosine and 4.15 for the latent value, so the floor never bound.

Every estimate is built from two independent measurement views, so no term is a
self inner product of one noisy vector and the estimator is unbiased for the
latent quantity rather than inflated by measurement variance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------- Phase 1 rule
METRIC_SPACE = "IDENTITY"
ELIGIBILITY_STATISTIC = "SHARED_LATENT_TRANSITION_ENERGY"
ELIGIBILITY_NULL = "MISMATCHED_PARTNER_PERMUTATION_ONE_SIDED"
CALIBRATION_ROLE = "G_fit"
PRIMARY_FDR = 0.05
SENSITIVITY_FDR = (0.01, 0.10)
REQUIRE_POSITIVE_ENERGY = True

# The level is prescribed by the frozen instruction, not selected from any
# calibration curve, and it is applied to G_select unchanged.  Nothing in this
# module may read a G_select pass count, a model score, or a reproducibility
# number in order to set it.
GATE_SELECTED_FROM_G_SELECT = False


@dataclass(frozen=True)
class Eligibility:
    """Per-identity eligibility decision and everything needed to audit it."""

    energy: np.ndarray
    p_value: np.ndarray
    q_value: np.ndarray
    passes_fdr: np.ndarray
    positive_energy: np.ndarray
    eligible: np.ndarray

    @property
    def count(self) -> int:
        return int(self.eligible.size)

    @property
    def eligible_count(self) -> int:
        return int(self.eligible.sum())

    @property
    def fraction(self) -> float:
        return float(self.eligible.mean())


def cross_view_gram(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """``M[i, j] = <first_i, second_j>`` under ``W = I``."""
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("the two views must share one identity and feature axis")
    if left.ndim != 2:
        raise ValueError("a view must be a two-dimensional identity-by-feature array")
    return left @ right.T


def permutation_p_value(gram: np.ndarray) -> np.ndarray:
    """One-sided empirical p-value of ``E_q`` against its mismatched partners.

    With ``m = n - 1`` mismatched partners the usual ``(1 + b) / (1 + m)``
    correction is written as ``(1 + b) / n``; the smallest attainable p-value is
    therefore ``1 / n``, which bounds how small an FDR level can ever be met.
    """
    gram = np.asarray(gram, dtype=np.float64)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("the identifiability null needs a square cross-view Gram")
    count = gram.shape[0]
    observed = np.ascontiguousarray(np.diag(gram)).copy()
    mismatched = ~np.eye(count, dtype=bool)
    exceed = (gram >= observed[:, None]) & mismatched
    return (1.0 + exceed.sum(axis=1)) / float(count)


def benjamini_hochberg(p_value: np.ndarray) -> np.ndarray:
    """Step-up BH q-values, monotone by construction."""
    p_value = np.asarray(p_value, dtype=np.float64)
    count = p_value.size
    order = np.argsort(p_value, kind="stable")
    scaled = p_value[order] * count / np.arange(1, count + 1, dtype=np.float64)
    q_value = np.empty(count, dtype=np.float64)
    q_value[order] = np.minimum.accumulate(scaled[::-1])[::-1].clip(max=1.0)
    return q_value


def eligibility(gram: np.ndarray, level: float = PRIMARY_FDR) -> Eligibility:
    """The frozen rule: BH-FDR on the permutation p-value, and a positive energy."""
    energy = np.ascontiguousarray(np.diag(np.asarray(gram, dtype=np.float64))).copy()
    p_value = permutation_p_value(gram)
    q_value = benjamini_hochberg(p_value)
    passes = q_value <= level
    positive = energy > 0.0 if REQUIRE_POSITIVE_ENERGY else np.ones_like(passes)
    return Eligibility(energy, p_value, q_value, passes, positive, passes & positive)


def eligibility_at_fixed_p(gram: np.ndarray, cutoff: float) -> np.ndarray:
    """Sensitivity form: a p-value cutoff frozen elsewhere, applied verbatim."""
    energy = np.ascontiguousarray(np.diag(np.asarray(gram, dtype=np.float64))).copy()
    return (permutation_p_value(gram) <= cutoff) & (energy > 0.0)


def bh_p_cutoff(p_value: np.ndarray, level: float) -> float:
    """Largest p-value BH rejects at ``level``; ``0.0`` when it rejects nothing."""
    p_value = np.asarray(p_value, dtype=np.float64)
    count = p_value.size
    ranked = np.sort(p_value, kind="stable")
    admissible = ranked <= np.arange(1, count + 1, dtype=np.float64) * level / count
    if not admissible.any():
        return 0.0
    return float(ranked[np.flatnonzero(admissible)[-1]])


# ---------------------------------------------------------- Phase 2 value form
def latent_value(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """``V[q, c] = M[c, q] + M[q, c] - M[c, c]`` for ``M = first @ second.T``.

    Row ``q`` is the query, column ``c`` the candidate.  Written out this is
    ``r_c^first . d_q^second + r_c^second . d_q^first - r_c^first . r_c^second``,
    which is the four-way estimate the instruction specifies.  No division, no
    clipping, no per-query rescaling.
    """
    gram = cross_view_gram(first, second)
    diagonal = np.ascontiguousarray(np.diag(gram)).copy()
    return gram.T + gram - diagonal[None, :]


def cosine_value(view: np.ndarray) -> np.ndarray:
    """The incumbent frozen utility: single-view row-normalized cosine."""
    values = np.asarray(view, dtype=np.float64)
    norm = np.linalg.norm(values, axis=1)
    norm[norm == 0.0] = 1.0
    unit = values / norm[:, None]
    return unit @ unit.T


def physical_bound_violation(value: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """``V > E_q`` means the implied squared distance is negative, which cannot be.

    This reads out estimator variance, not a candidate that overshoots the goal.
    """
    return np.asarray(value, dtype=np.float64) > np.asarray(energy, dtype=np.float64)[:, None]
