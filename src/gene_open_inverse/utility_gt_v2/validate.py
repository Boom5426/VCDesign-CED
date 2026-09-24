"""Phase-2 endpoint: does a budget prefix survive an independent measurement.

The primary endpoint is not Top-B identity overlap.  Overlap counts how many of
the same candidates come back and is blind to what they are worth, so a pair of
orderings that disagree on which near-equal candidate sits at rank 7 is punished
exactly as hard as a pair that disagrees about a candidate worth ten times the
median.  The deployment question is the second one, so the endpoint is graded in
decision units: rank by one view, grade by the other, and report how much of
that other view's own achievable value the prefix delivers.

Both estimators are compared on the same queries, the same candidates and the
same cells.  The incumbent cosine consumes side A for its first instance and
side B for its second; the latent value consumes ``(Q0, Q1)`` for its first and
``(Q2, Q3)`` for its second.  Those are the same cells, differently used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..model.decision_alignment_v1.contract import (
    BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, BUDGETS, mbru_from_nru, normalized_ranked_utility, query_references,
)
from ..model.open_vocab_dual_encoder_v1.metrics import stable_order


@dataclass(frozen=True)
class GradedRun:
    """Per-query nRU at every budget and MBRU, for one ordering graded by one view."""

    nru: dict[int, np.ndarray]
    mbru: np.ndarray
    denominator: dict[int, np.ndarray]


def legal_orders(value: np.ndarray, ids: np.ndarray, self_position: np.ndarray) -> np.ndarray:
    """Self-excluded descending order of the candidate axis, one row per query.

    ``self_position[i]`` is the column of query ``i`` in the candidate axis, or
    ``-1`` when that query is not itself a candidate.  Exclusion is by position,
    never by score, so a tie can never smuggle the target intervention back in.
    """
    queries, candidates = value.shape
    present = self_position >= 0
    if present.any() and not present.all():
        raise ValueError("every query must either be a candidate or none may be; rows would differ in length")
    keep = candidates - 1 if present.all() else candidates
    output = np.empty((queries, keep), dtype=np.int32)
    for row in range(queries):
        index = int(self_position[row])
        order = stable_order(value[row], ids, excluded_index=None if index < 0 else index)
        order = order[order != index] if index >= 0 else order
        output[row] = order[:keep].astype(np.int32)
    return output


def grade(order: np.ndarray, grading: np.ndarray, ids: np.ndarray, self_position: np.ndarray) -> GradedRun:
    """Grade an ordering under an independent view's value, in nRU and MBRU."""
    queries = order.shape[0]
    nru = {budget: np.empty(queries, dtype=np.float64) for budget in BUDGETS}
    denominator = {budget: np.empty(queries, dtype=np.float64) for budget in BUDGETS}
    mbru = np.empty(queries, dtype=np.float64)
    for row in range(queries):
        references = query_references(grading[row], ids, int(self_position[row]))
        values = normalized_ranked_utility(grading[row, order[row]], references)
        for budget in BUDGETS:
            nru[budget][row] = values[budget]
            denominator[budget][row] = references.denominator[budget]
        mbru[row] = mbru_from_nru(values)
    return GradedRun(nru, mbru, denominator)


def summarize(values: np.ndarray) -> dict:
    return {
        "median": float(np.median(values)), "mean": float(values.mean()),
        "fraction_below_zero": float((values < 0).mean()),
        "p05": float(np.quantile(values, 0.05)), "p95": float(np.quantile(values, 0.95)),
    }


def graded_summary(run: GradedRun) -> dict:
    record = {f"nRU@{budget}": summarize(run.nru[budget]) for budget in BUDGETS}
    record["MBRU"] = summarize(run.mbru)
    record["min_denominator"] = {str(budget): float(run.denominator[budget].min()) for budget in BUDGETS}
    return record


def cluster_bootstrap(difference: np.ndarray, replicates: int = BOOTSTRAP_REPLICATES,
                      seed: int = BOOTSTRAP_SEED) -> dict:
    """Identity-cluster bootstrap over queries; the cluster unit is the query identity."""
    difference = np.asarray(difference, dtype=np.float64)
    generator = np.random.default_rng(seed)
    count = difference.size
    draws = generator.integers(0, count, size=(replicates, count))
    resampled = difference[draws]
    means = resampled.mean(axis=1)
    medians = np.median(resampled, axis=1)
    return {
        "replicates": int(replicates), "seed": int(seed), "clusters": int(count),
        "mean": {"point": float(difference.mean()),
                 "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
                 "fraction_of_replicates_above_zero": float((means > 0).mean())},
        "median": {"point": float(np.median(difference)),
                   "ci95": [float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))],
                   "fraction_of_replicates_above_zero": float((medians > 0).mean())},
    }


def set_overlap(first: np.ndarray, second: np.ndarray) -> dict:
    record = {}
    for budget in BUDGETS:
        overlap = np.asarray([
            len(set(first[row, :budget].tolist()) & set(second[row, :budget].tolist())) / budget
            for row in range(first.shape[0])
        ], dtype=np.float64)
        record[str(budget)] = {"median": float(np.median(overlap)), "mean": float(overlap.mean()),
                               "fraction_zero": float((overlap == 0).mean()),
                               "chance": float(budget) / float(first.shape[1])}
    return record


def boundary_flip(order: np.ndarray, proposing: np.ndarray, grading: np.ndarray) -> dict:
    """How often an independent view reverses the pair that decides the budget.

    Chance is 0.5 by construction, so this is a direct statement about whether
    the candidate at rank ``B`` is distinguishable from the one at rank ``B+1``.
    It needs no noise model.
    """
    record = {}
    for budget in BUDGETS:
        inside = order[:, budget - 1]
        outside = order[:, budget]
        rows = np.arange(order.shape[0])
        margin = proposing[rows, inside] - proposing[rows, outside]
        replicate = grading[rows, inside] - grading[rows, outside]
        record[str(budget)] = {
            "margin_median": float(np.median(margin)),
            "independent_view_reverses": float((replicate < 0).mean()),
        }
    return record


def spearman(first: np.ndarray, second: np.ndarray, self_position: np.ndarray) -> dict:
    """Auxiliary only: whole-row rank correlation, which the budget does not use."""
    values = np.empty(first.shape[0], dtype=np.float64)
    for row in range(first.shape[0]):
        keep = np.ones(first.shape[1], dtype=bool)
        index = int(self_position[row])
        if index >= 0:
            keep[index] = False
        left = _rankdata(first[row, keep])
        right = _rankdata(second[row, keep])
        left = left - left.mean()
        right = right - right.mean()
        scale = np.sqrt((left * left).sum() * (right * right).sum())
        values[row] = float((left * right).sum() / scale) if scale > 0 else 0.0
    return {"median": float(np.median(values)), "mean": float(values.mean())}


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    for index in range(1, values.size + 1):
        if index == values.size or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return ranks


def goal_independent_order(value: np.ndarray, ids: np.ndarray, self_position: np.ndarray) -> np.ndarray:
    """The legal global prior: rank every query by the candidate's mean value.

    This is the control that matters most for a new ground truth.  If a value
    definition is more reproducible mainly because it carries a large
    query-independent component, then a policy that ignores the goal entirely
    will reproduce most of its advantage.  D-005 in this programme's decision log
    exists because random is not the comparison of record; this is.

    The column mean excludes each query's own diagonal entry, which is the
    largest value in its column by construction and would otherwise let the
    self-similarity of 24% of the candidates leak into a policy that is supposed
    to carry no query information at all.

    It is a lower bound on what a goal-blind policy can do: the ordering that
    maximizes MBRU for a single shared permutation would weight the column mean
    by the per-query nRU denominator.  Every advantage computed against it is
    therefore an upper bound on the goal-conditioned content of a definition.
    """
    mask = np.ones(value.shape, dtype=bool)
    present = self_position >= 0
    if present.any():
        mask[np.flatnonzero(present), self_position[present]] = False
    prior = (value * mask).sum(axis=0) / np.maximum(mask.sum(axis=0), 1)
    tiled = np.broadcast_to(prior, value.shape)
    return legal_orders(np.ascontiguousarray(tiled), ids, self_position)
