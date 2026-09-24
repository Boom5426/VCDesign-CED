"""Best/Mean@B metrics with explicit origin self-exclusion and fixed tie handling."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class TopKSummary:
    best: float
    mean: float
    valid_queries: int
    budget: int


def stable_order(scores: np.ndarray, candidate_ids: np.ndarray, excluded_index: int | None = None, candidate_mask: np.ndarray | None = None) -> np.ndarray:
    """Descending score with fixed ties, self exclusion, and an optional static coverage mask."""
    rank = np.asarray(scores, dtype=np.float64).copy()
    if candidate_mask is not None:
        candidate_mask = np.asarray(candidate_mask, dtype=bool)
        if candidate_mask.shape != rank.shape:
            raise ValueError("candidate_mask must match one candidate-score row")
        rank[~candidate_mask] = -np.inf
    if excluded_index is not None:
        rank[int(excluded_index)] = -np.inf
    return np.lexsort((np.asarray(candidate_ids, dtype=str), -rank))


def topk_summary(scores: np.ndarray, utility: np.ndarray, candidate_ids: np.ndarray, *, budget: int, self_exclude: bool, candidate_mask: np.ndarray | None = None) -> TopKSummary:
    scores = np.asarray(scores, dtype=np.float64)
    utility = np.asarray(utility, dtype=np.float64)
    if scores.ndim != 2 or utility.shape != scores.shape:
        raise ValueError("scores and utility must be aligned matrices")
    if scores.shape[1] != len(candidate_ids) or budget <= 0:
        raise ValueError("candidate axis or budget is invalid")
    if candidate_mask is not None and np.asarray(candidate_mask, dtype=bool).shape != (scores.shape[1],):
        raise ValueError("candidate_mask must have one value per candidate")
    best: list[float] = []
    mean: list[float] = []
    for query in range(scores.shape[0]):
        excluded = query if self_exclude else None
        order = stable_order(scores[query], candidate_ids, excluded, candidate_mask)
        if self_exclude:
            order = order[order != query]
        if candidate_mask is not None:
            order = order[np.asarray(candidate_mask, dtype=bool)[order]]
        picked = order[:budget]
        values = utility[query, picked]
        if len(picked) != budget or not np.isfinite(values).all():
            continue
        best.append(float(values.max()))
        mean.append(float(values.mean()))
    return TopKSummary(best=float(np.mean(best)) if best else float("nan"), mean=float(np.mean(mean)) if mean else float("nan"), valid_queries=len(best), budget=budget)


def summary_record(label: str, summary: TopKSummary) -> dict:
    record = asdict(summary)
    record["label"] = label
    return record


def goal_delta(true_scores: np.ndarray, wrong_scores: np.ndarray, utility: np.ndarray, candidate_ids: np.ndarray, *, budget: int, candidate_mask: np.ndarray | None = None) -> dict:
    true = topk_summary(true_scores, utility, candidate_ids, budget=budget, self_exclude=True, candidate_mask=candidate_mask)
    wrong = topk_summary(wrong_scores, utility, candidate_ids, budget=budget, self_exclude=True, candidate_mask=candidate_mask)
    return {
        "true": summary_record("true_goal", true), "wrong": summary_record("wrong_goal", wrong),
        "delta_best": true.best - wrong.best, "delta_mean": true.mean - wrong.mean,
    }


def secondary_topk_diagnostics(scores: np.ndarray, utility: np.ndarray, candidate_ids: np.ndarray, *, budget: int, candidate_mask: np.ndarray | None = None) -> dict:
    """Self-excluded regret and oracle-top-B recall, never used for selection."""
    regret: list[float] = []
    recall: list[float] = []
    for query in range(scores.shape[0]):
        predicted = stable_order(scores[query], candidate_ids, query, candidate_mask)
        predicted = predicted[predicted != query][:budget]
        oracle = stable_order(utility[query], candidate_ids, query, candidate_mask)
        oracle = oracle[oracle != query][:budget]
        if candidate_mask is not None:
            mask = np.asarray(candidate_mask, dtype=bool)
            predicted, oracle = predicted[mask[predicted]], oracle[mask[oracle]]
        if len(predicted) != budget or len(oracle) != budget:
            continue
        regret.append(float(utility[query, oracle].max() - utility[query, predicted].max()))
        recall.append(float(len(set(predicted.tolist()) & set(oracle.tolist())) / budget))
    return {"regret_at_budget": float(np.mean(regret)) if regret else float("nan"),
            "recall_against_oracle_top_budget": float(np.mean(recall)) if recall else float("nan"),
            "valid_queries": len(regret), "budget": budget}


def oracle_ceiling(utility: np.ndarray, candidate_ids: np.ndarray, *, budget: int, candidate_mask: np.ndarray | None = None) -> TopKSummary:
    """Best/Mean@B when frozen GT utility itself orders the eligible pool."""
    return topk_summary(utility, utility, candidate_ids, budget=budget, self_exclude=True, candidate_mask=candidate_mask)
