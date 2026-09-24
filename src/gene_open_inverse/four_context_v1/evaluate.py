"""Grading, stratified by candidate, under the frozen metric contract.

Two things differ from the anchor's evaluator and both are necessary here.

First, a stratum is a subset of the *candidate* axis, not of the queries, so the
score and utility matrices are rectangular: the query stays the same and the pool it
ranks shrinks.  That is what separates candidate generalization from context
generalization.  Restricting the pool also moves the per-query random and oracle
references, which is correct: within a stratum the achievable ceiling is the
stratum's own ceiling, and an absolute number from one stratum is not comparable to
another's.  Only paired differences between arms inside one stratum are.

Second, self-exclusion has to survive the restriction.  A query is dropped from the
pool when it is in it and simply absent when it is not, and the excluded index is
``None`` rather than ``-1`` in the second case, because the frozen ``stable_order``
would read ``-1`` as the last candidate and silently delete a real one.
"""
from __future__ import annotations

import numpy as np

from ..model.decision_alignment_v1.contract import (
    BUDGETS, MAX_BUDGET, mbru_from_nru, normalized_ranked_utility, query_references,
)
from ..model.open_vocab_dual_encoder_v1.metrics import stable_order
from ..utility_gt_v2.validate import cluster_bootstrap, summarize


HIGH_VALUE_QUANTILE = 0.95


def consensus(responses: np.ndarray) -> np.ndarray:
    """``r_bar_c``, the mean of the two batch-disjoint view responses."""
    values = np.asarray(responses, dtype=np.float64)
    return 0.5 * (values[0] + values[1])


def unit(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-30)


def cross_view_utility(responses: np.ndarray, view: int) -> np.ndarray:
    """The anchor's convention: a view-``v`` query is graded by view ``1-v`` responses."""
    other = unit(np.asarray(responses[1 - view], dtype=np.float64))
    return other @ other.T


def effect_cosine(responses: np.ndarray, effect: np.ndarray, view: int) -> np.ndarray:
    """``cos(d_q, r_hat_c)`` with a candidate that has no predicted effect scoring zero."""
    direction = unit(np.asarray(responses[view], dtype=np.float64))
    predicted = unit(np.asarray(effect, dtype=np.float64))
    cosine = direction @ predicted.T
    cosine[:, np.linalg.norm(effect, axis=1) <= 0.0] = 0.0
    return cosine


def row_z(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    return (values - values.mean(axis=1, keepdims=True)) / np.maximum(
        values.std(axis=1, keepdims=True), 1e-12)


def fuse(base: np.ndarray, effect: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """The frozen fusion: z-score additive at unit weight, no fitted coefficient."""
    return row_z(base) + beta * row_z(effect)


def stratum_metrics(scores: np.ndarray, utility: np.ndarray, candidate_ids: np.ndarray,
                    self_positions: np.ndarray) -> dict:
    """Per-query MBRU, Mean@B and HighValueCount@B on one candidate stratum.

    ``scores`` and ``utility`` are ``[queries, candidates_in_stratum]`` and
    ``self_positions[i]`` is where query ``i`` sits inside that stratum, or ``-1``.
    """
    scores = np.asarray(scores, dtype=np.float64)
    utility = np.asarray(utility, dtype=np.float64)
    if scores.shape != utility.shape or scores.shape[1] != len(candidate_ids):
        raise ValueError("scores, utility and the candidate axis must be aligned")
    queries = scores.shape[0]
    mbru = np.empty(queries, dtype=np.float64)
    mean_at = {budget: np.empty(queries) for budget in BUDGETS}
    high_at = {budget: np.empty(queries) for budget in BUDGETS}
    for row in range(queries):
        position = int(self_positions[row])
        excluded = position if position >= 0 else None
        order = stable_order(scores[row], candidate_ids, excluded)
        if excluded is not None:
            order = order[order != position]
        references = query_references(utility[row], candidate_ids, excluded)
        mbru[row] = mbru_from_nru(normalized_ranked_utility(utility[row, order], references))
        legal = np.delete(utility[row], position) if position >= 0 else utility[row]
        threshold = float(np.quantile(legal, HIGH_VALUE_QUANTILE))
        for budget in BUDGETS:
            picked = utility[row, order[:budget]]
            mean_at[budget][row] = float(picked.mean())
            high_at[budget][row] = float((picked >= threshold).sum())
    return {"MBRU": mbru, **{f"Mean@{b}": mean_at[b] for b in BUDGETS},
            **{f"HighValueCount@{b}": high_at[b] for b in BUDGETS}}


def oracle_scores(utility: np.ndarray) -> np.ndarray:
    """The measured-response oracle: rank by the graded utility itself."""
    return np.asarray(utility, dtype=np.float64)


def paired(better: np.ndarray, worse: np.ndarray) -> dict:
    difference = np.asarray(better, dtype=np.float64) - np.asarray(worse, dtype=np.float64)
    return {**summarize(difference), "bootstrap": cluster_bootstrap(difference),
            "fraction_query_improved": float((difference > 0).mean()),
            "queries": int(difference.size)}


def evaluate_arms(score_by_arm: dict, utility_by_view: list, candidate_ids: np.ndarray,
                  self_positions: np.ndarray, mask: np.ndarray) -> dict:
    """Every arm on one candidate stratum, both query views pooled per query.

    Score and utility matrices are ``[queries, pool]`` rather than square: the query
    axis is the held context's eligible queries and the candidate axis is its whole
    usable pool.  ``self_positions[i]`` is where query ``i`` sits in that pool, or
    ``-1`` if it is not a candidate.  ``mask`` selects the stratum inside the pool.
    """
    columns = np.flatnonzero(np.asarray(mask, dtype=bool))
    if columns.size <= MAX_BUDGET:
        return {"skipped": f"stratum holds {int(columns.size)} candidates, "
                           f"not more than the largest budget {MAX_BUDGET}"}
    stratum_ids = np.asarray(candidate_ids, dtype=str)[columns]
    remap = {int(value): index for index, value in enumerate(columns.tolist())}
    inside = np.asarray([remap.get(int(p), -1) for p in np.asarray(self_positions).tolist()],
                        dtype=np.int64)

    per_arm, vectors = {}, {}
    for name, by_view in score_by_arm.items():
        pooled: dict = {}
        for view in (0, 1):
            block = stratum_metrics(np.asarray(by_view[view])[:, columns],
                                    np.asarray(utility_by_view[view])[:, columns],
                                    stratum_ids, inside)
            for key, values in block.items():
                pooled.setdefault(key, []).append(values)
        vectors[name] = {key: np.concatenate(values) for key, values in pooled.items()}
        per_arm[name] = {key: summarize(values) for key, values in vectors[name].items()}
    return {"candidates": int(columns.size), "queries": int(len(inside) * 2),
            "arms": per_arm, "_vectors": vectors}
