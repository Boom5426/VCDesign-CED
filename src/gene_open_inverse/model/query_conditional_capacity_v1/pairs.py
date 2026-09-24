"""Pair sets built once, written to disk, and read unchanged by every probe.

The previous phase could not separate "the sampler chose badly" from "the model
cannot represent the distinction", because the pairs a model trained on were
chosen from that model's own current ranking and moved every step.  Here the pair
sets are constructed once from frozen inputs, saved, and consumed identically by
both scorers, so a difference between the scorers cannot come from having seen
different pairs.

Three kinds, all utility-consistent and all self-excluded by position.

``UNIFORM``
    Two candidates drawn uniformly from the legal pool, kept when their frozen
    utilities differ.  This is the population the 53% pairwise-accuracy figure was
    measured on, so it is the one that answers whether the scorer is better than a
    coin at all.

``TOP_B_RELEVANT``
    The preferred member is drawn from the GT top 50, the other uniformly.  These
    are the pairs whose ordering can move the budget prefix under the ground truth.

``DECISION_CRITICAL``
    The largest exact swap values ``dMBRU`` under the STARTING checkpoint's
    predicted ranking.  These are inversions that checkpoint has, so they are the
    pairs a decision-aligned sampler would spend its budget on.  They are defined
    by STARTING once and never re-derived from a probe's own ranking: a positive
    ``dMBRU`` is by definition a pair the ranking model has inverted, so selecting
    them from the model under test would make its accuracy on them exactly zero by
    construction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..decision_alignment_v1.contract import BUDGETS, DENOMINATOR_FLOOR, DISCOUNT, DISCOUNT_SUM, MAX_BUDGET
from ..open_vocab_dual_encoder_v1.metrics import stable_order


KINDS = ("UNIFORM", "TOP_B_RELEVANT", "DECISION_CRITICAL")
PAIRS_PER_KIND = 256
PAIR_SEED = 20260918
GAP_WEIGHT_RANGE = (0.25, 4.0)


@dataclass(frozen=True)
class PairSet:
    """One flat pair table over a role and view, plus the frozen loss weight."""

    query: np.ndarray
    better: np.ndarray
    worse: np.ndarray
    kind: np.ndarray
    weight: np.ndarray

    def __len__(self) -> int:
        return int(self.query.size)

    def of_kind(self, kind: str) -> "PairSet":
        keep = self.kind == KINDS.index(kind)
        return PairSet(self.query[keep], self.better[keep], self.worse[keep], self.kind[keep], self.weight[keep])


def rank_coefficients(utility_row: np.ndarray, order: np.ndarray) -> np.ndarray:
    """``a_q(r)`` over the predicted rank axis, from this query's own nRU denominator."""
    legal = utility_row[order]
    reference = float(legal.mean())
    ordered = np.sort(legal)[::-1]
    values = np.zeros(order.size, dtype=np.float64)
    for budget in BUDGETS:
        oracle = float(np.dot(DISCOUNT[budget], ordered[:budget]) / DISCOUNT_SUM[budget])
        safe = max(oracle - reference, DENOMINATOR_FLOOR)
        values[:budget] += DISCOUNT[budget] / (DISCOUNT_SUM[budget] * safe)
    return values / len(BUDGETS)


def _frozen_weight(utility_row: np.ndarray, legal: np.ndarray, better: np.ndarray, worse: np.ndarray,
                   gt_rank: np.ndarray) -> np.ndarray:
    """The frozen ``gap_weight * top_weight``, so the loss form is unchanged."""
    legal_values = utility_row[legal]
    iqr = float(np.quantile(legal_values, 0.75) - np.quantile(legal_values, 0.25))
    gap_weight = np.clip((utility_row[better] - utility_row[worse]) / max(iqr, 1e-12), *GAP_WEIGHT_RANGE)
    rank = gt_rank[better]
    top_weight = np.where(rank <= 20, 4.0, np.where(rank <= 64, 2.0, np.where(rank <= 128, 1.0, 0.5)))
    return gap_weight * top_weight


def build(utility: np.ndarray, scores: np.ndarray, ids: np.ndarray, queries: np.ndarray,
          *, pairs_per_kind: int = PAIRS_PER_KIND, seed: int = PAIR_SEED) -> PairSet:
    """Build all three kinds for every query in ``queries``.

    ``scores`` is the STARTING checkpoint's score matrix and is used only to define
    the ``DECISION_CRITICAL`` kind.
    """
    generator = np.random.default_rng(seed)
    query_out, better_out, worse_out, kind_out, weight_out = [], [], [], [], []
    for query in queries.tolist():
        legal = np.delete(np.arange(utility.shape[1]), query)
        row = utility[query]
        gt_order = stable_order(row, ids, excluded_index=query)
        gt_order = gt_order[gt_order != query]
        gt_rank = np.empty(utility.shape[1], dtype=np.int64)
        gt_rank.fill(utility.shape[1] + 1)
        gt_rank[gt_order] = np.arange(1, gt_order.size + 1)

        chosen: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        left = legal[generator.integers(0, legal.size, size=pairs_per_kind * 4)]
        right = legal[generator.integers(0, legal.size, size=pairs_per_kind * 4)]
        keep = row[left] != row[right]
        left, right = left[keep][:pairs_per_kind], right[keep][:pairs_per_kind]
        swap = row[left] < row[right]
        chosen["UNIFORM"] = (np.where(swap, right, left), np.where(swap, left, right))

        top = gt_order[:MAX_BUDGET]
        high = top[generator.integers(0, top.size, size=pairs_per_kind * 4)]
        low = legal[generator.integers(0, legal.size, size=pairs_per_kind * 4)]
        keep = row[high] > row[low]
        chosen["TOP_B_RELEVANT"] = (high[keep][:pairs_per_kind], low[keep][:pairs_per_kind])

        predicted = stable_order(scores[query], ids, excluded_index=query)
        predicted = predicted[predicted != query]
        coefficients = rank_coefficients(row, predicted)
        inside = predicted[:MAX_BUDGET]
        gap = row[predicted][None, :] - row[inside][:, None]
        value = np.clip(gap, 0.0, None) * np.clip(coefficients[:MAX_BUDGET][:, None] - coefficients[None, :], 0.0, None)
        value[:, :MAX_BUDGET] = np.triu(value[:, :MAX_BUDGET], 1)
        flat = value.reshape(-1)
        positive = int((flat > 0).sum())
        if positive:
            take = min(pairs_per_kind, positive)
            top_flat = np.argpartition(-flat, take - 1)[:take]
            chosen["DECISION_CRITICAL"] = (predicted[top_flat % predicted.size], inside[top_flat // predicted.size])
        else:
            chosen["DECISION_CRITICAL"] = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))

        for index, kind in enumerate(KINDS):
            better, worse = chosen[kind]
            if better.size == 0:
                continue
            if not bool((row[better] > row[worse]).all()):
                raise RuntimeError(f"{kind} produced a utility-inconsistent pair")
            if query in better.tolist() or query in worse.tolist():
                raise RuntimeError("self-exclusion failed in pair construction")
            query_out.append(np.full(better.size, query, dtype=np.int32))
            better_out.append(better.astype(np.int32))
            worse_out.append(worse.astype(np.int32))
            kind_out.append(np.full(better.size, index, dtype=np.int8))
            weight_out.append(_frozen_weight(row, legal, better, worse, gt_rank))
    return PairSet(np.concatenate(query_out), np.concatenate(better_out), np.concatenate(worse_out),
                   np.concatenate(kind_out), np.concatenate(weight_out))


def save(path, pair_set: PairSet) -> None:
    np.savez(path, query=pair_set.query, better=pair_set.better, worse=pair_set.worse,
             kind=pair_set.kind, weight=pair_set.weight)


def load(path) -> PairSet:
    with np.load(path) as archive:
        return PairSet(archive["query"], archive["better"], archive["worse"], archive["kind"], archive["weight"])
