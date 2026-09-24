"""The frozen action contract, its sufficient statistics, and the four ranking policies.

Action contract
---------------
    a in [0, 1]

Attenuation only.  Amplification is not assumed, and the range is frozen before
any result is seen.  Nothing in this module may widen it.

Deployment utility
------------------
    J_qc(a) = 2 a s_qc - a^2 R_c ,   s_qc = <r_c, d_q> ,   R_c = ||r_c||^2

which is exactly ``||d_q||^2 - ||a r_c - d_q||^2``, the reduction in squared
endpoint distance relative to doing nothing.  ``J > 0`` is closer to the goal
than no-op, ``J = 0`` is equivalent to no-op, ``J < 0`` is worse.  No division by
``||d_q||^2``, no clipping, no cosine or magnitude weighting.

``a = 1`` recovers the fixed-endpoint value of the previous phase, so
``FIXED_ENDPOINT`` is the ``a = 1`` slice of the same family rather than a
separate object.

Why the optimum is worth stating in closed form
-----------------------------------------------
When the stationary point ``s / R`` lies inside the interval the optimum value is

    J(s/R) = s^2 / R = ||d_q||^2 cos^2(r_c, d_q)

so the candidate's magnitude cancels completely: with attenuation available, an
interior candidate is worth exactly its squared directional alignment, scaled by
a per-query constant.  Ranking by that is ranking by ``cos``.  The attenuation
policy can therefore only differ from the direction policy through the boundary
regime ``s >= R``, where a candidate is too weak to reach its own optimum even at
full strength and is penalised for it.  That makes the comparison against
``DIRECTION`` a real test rather than a foregone one, and it is the reason the
two policies must be compared under one shared grading utility.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ACTION_LOWER = 0.0
ACTION_UPPER = 1.0
ACTION_CONTRACT = "ATTENUATION_ONLY_a_in_0_1"
METRIC_SPACE = "IDENTITY"
POLICIES = ("ATTENUATION_AWARE", "DIRECTION", "FIXED_ENDPOINT", "GOAL_INDEPENDENT")
PRIMARY_COMPARISONS = (
    ("ATTENUATION_AWARE", "DIRECTION"),
    ("ATTENUATION_AWARE", "FIXED_ENDPOINT"),
    ("ATTENUATION_AWARE", "GOAL_INDEPENDENT"),
)


@dataclass(frozen=True)
class ViewStatistics:
    """Cross-quarter sufficient statistics from one pair of batch-disjoint quarters."""

    alignment: np.ndarray
    energy: np.ndarray
    target_energy: np.ndarray

    def restrict(self, rows: np.ndarray) -> "ViewStatistics":
        """Keep every candidate but only the query rows this phase evaluates.

        ``energy`` is indexed by candidate and is therefore left whole;
        ``alignment`` and ``target_energy`` are indexed by query and are cut.
        """
        rows = np.asarray(rows, dtype=np.int64)
        return ViewStatistics(alignment=np.ascontiguousarray(self.alignment[rows]),
                              energy=self.energy, target_energy=self.target_energy[rows])

    def value_at(self, action: np.ndarray) -> np.ndarray:
        """``J(a)`` for a per-pair action, evaluated under this view."""
        action = np.asarray(action, dtype=np.float64)
        return 2.0 * action * self.alignment - action * action * self.energy[None, :]


def view_statistics(first: np.ndarray, second: np.ndarray) -> ViewStatistics:
    """``s``, ``R`` and the query's own transition energy from two independent quarters.

    Every term is a product across the two quarters, so no vector multiplies
    itself and none of the three statistics carries a noise-energy bias.
    """
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("the two quarters must share one identity and feature axis")
    gram = left @ right.T
    diagonal = np.ascontiguousarray(np.diag(gram)).copy()
    alignment = 0.5 * (gram.T + gram)
    return ViewStatistics(alignment=alignment, energy=diagonal, target_energy=diagonal)


def optimal_attenuation(statistics: ViewStatistics) -> tuple[np.ndarray, np.ndarray]:
    """Exact maximiser of ``J(a)`` on the closed interval ``[0, 1]``, and its value.

    The one-dimensional quadratic is maximised by comparing the analytic
    stationary point, when it lies strictly inside the interval, against both
    endpoints, and taking the largest objective value.  Written this way the
    behaviour is decided by the same optimisation problem even when noise makes
    ``R <= 0`` and the quadratic is convex, so no separate heuristic is needed:
    a convex quadratic simply has its maximum at an endpoint, and the stationary
    point loses the comparison on value.

    Ties are broken toward the smaller action, so a candidate that cannot beat
    no-op is assigned ``a = 0`` rather than an arbitrary positive dose.
    """
    alignment = statistics.alignment
    energy = np.broadcast_to(statistics.energy[None, :], alignment.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        stationary = np.where(energy != 0.0, alignment / energy, np.nan)
    inside = np.isfinite(stationary) & (stationary > ACTION_LOWER) & (stationary < ACTION_UPPER)
    interior = np.where(inside, stationary, ACTION_LOWER)

    actions = np.stack([
        np.full(alignment.shape, ACTION_LOWER), np.full(alignment.shape, ACTION_UPPER), interior,
    ])
    values = np.stack([statistics.value_at(action) for action in actions])
    best = values.max(axis=0)
    reachable = np.where(values >= best, actions, np.inf)
    action = reachable.min(axis=0)
    return action, statistics.value_at(action)


def cross_fitted_value(action: np.ndarray, grading: ViewStatistics) -> np.ndarray:
    """Realized ``J`` of an action chosen on one view, measured on the other.

    The measurement that chose the action never appears here.  This is the only
    quantity any reported deployment value may come from.
    """
    return grading.value_at(action)


def cosine_value(view: np.ndarray) -> np.ndarray:
    """The incumbent direction utility: single-view row-normalized cosine."""
    values = np.asarray(view, dtype=np.float64)
    norm = np.linalg.norm(values, axis=1)
    norm[norm == 0.0] = 1.0
    unit = values / norm[:, None]
    return unit @ unit.T


def policy_scores(statistics: ViewStatistics, cosine: np.ndarray, attenuated: np.ndarray) -> dict[str, np.ndarray]:
    """Every policy's proposing score, all read off the same proposer view.

    ``GOAL_INDEPENDENT`` is the candidate-only control that reads no query
    information at all, not even the pooled mean target direction.
    """
    shape = statistics.alignment.shape
    return {
        "ATTENUATION_AWARE": attenuated,
        "DIRECTION": cosine,
        "FIXED_ENDPOINT": 2.0 * statistics.alignment - statistics.energy[None, :],
        "GOAL_INDEPENDENT": np.ascontiguousarray(np.broadcast_to(-statistics.energy[None, :], shape)),
    }
