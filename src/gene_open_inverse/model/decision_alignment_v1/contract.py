"""Ranked Utility, nRU and MBRU exactly as frozen in the alignment contract.

Every quantity here is defined by `docs/VCDesign_ALIGNMENT_CONTRACT_V1.md`.  The
contract hash below is recorded in every artifact this package writes; if the
file is edited the hash check fails and the run stops rather than reporting
numbers under a contract that no longer says what it said.

Self-exclusion, descending score, and the ascending candidate-ID tie rule are
delegated to the repository's frozen `stable_order`, so a method that emits a
constant score gets a deterministic ordering and not a favorable one.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..open_vocab_dual_encoder_v1.metrics import stable_order


BUDGETS = (10, 20, 50)
MAX_BUDGET = max(BUDGETS)
DENOMINATOR_FLOOR = 1e-8
MAX_DEGENERATE_FRACTION = 0.01
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260916
ALIGNMENT_CONTRACT_SHA256 = "2c2d00ba9e4295acd4ea41cf505377b9d259c5546e4020097dbba0b5e9763c1f"
CONTRACT_FILENAME = "VCDesign_ALIGNMENT_CONTRACT_V1.md"


def verify_contract(path: Path) -> str:
    """Fail loudly if the frozen contract file is not the one these numbers assume."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != ALIGNMENT_CONTRACT_SHA256:
        raise RuntimeError(
            f"alignment contract hash mismatch: {path} is {digest}, contract requires {ALIGNMENT_CONTRACT_SHA256}"
        )
    return digest


def discount(budget: int) -> np.ndarray:
    """``d_r = 1 / log2(r + 1)`` for ``r = 1 .. budget``.  Fixed, never tuned."""
    if budget <= 0:
        raise ValueError("budget must be positive")
    ranks = np.arange(1, budget + 1, dtype=np.float64)
    return 1.0 / np.log2(ranks + 1.0)


def discount_sum(budget: int) -> float:
    """``Z_B``."""
    return float(discount(budget).sum())


DISCOUNT = {budget: discount(budget) for budget in BUDGETS}
DISCOUNT_SUM = {budget: discount_sum(budget) for budget in BUDGETS}


def legal_order(scores_row: np.ndarray, ids: np.ndarray, self_index: int) -> np.ndarray:
    """Predicted ordering of the legal pool, origin candidate removed."""
    order = stable_order(scores_row, ids, excluded_index=self_index)
    return order[order != self_index]


def ranked_utility(ordered_utility: np.ndarray, budget: int) -> float:
    """``RU@B`` from the utilities already arranged in predicted rank order."""
    if len(ordered_utility) < budget:
        raise ValueError("ordering is shorter than the requested budget")
    weights = DISCOUNT[budget] if budget in DISCOUNT else discount(budget)
    total = DISCOUNT_SUM[budget] if budget in DISCOUNT_SUM else discount_sum(budget)
    return float(np.dot(weights, ordered_utility[:budget]) / total)


@dataclass(frozen=True)
class QueryReferences:
    """Per-query random and oracle references, self-exclusion already applied."""

    random_reference: float
    oracle: dict[int, float]
    denominator: dict[int, float]
    legal_count: int


def query_references(utility_row: np.ndarray, ids: np.ndarray, self_index: int) -> QueryReferences:
    """``RU_rand``, ``RU_oracle@B`` and the nRU denominator for one query.

    The random reference needs no Monte Carlo: under a uniform permutation the
    expected utility is the same at every rank, so the discount cancels and the
    reference is the plain mean over the legal pool.
    """
    utility_row = np.asarray(utility_row, dtype=np.float64)
    order = stable_order(utility_row, ids, excluded_index=self_index)
    order = order[order != self_index]
    if len(order) < MAX_BUDGET:
        raise RuntimeError("self-excluded legal pool is smaller than the largest deployment budget")
    legal = utility_row[order]
    if not np.isfinite(legal).all():
        raise FloatingPointError("frozen utility row contains a non-finite legal value")
    reference = float(legal.mean())
    oracle = {budget: ranked_utility(legal, budget) for budget in BUDGETS}
    denominator = {budget: oracle[budget] - reference for budget in BUDGETS}
    return QueryReferences(reference, oracle, denominator, int(len(order)))


def rank_coefficients(denominator: dict[int, float]) -> np.ndarray:
    """``a_q(r)`` for ``r = 1 .. 50``; zero beyond the largest deployment budget.

    This is the exact derivative of MBRU with respect to the utility placed at
    rank ``r``, so a pure two-element swap changes MBRU by exactly
    ``(a(r_j) - a(r_i)) * (U_i - U_j)``.  It is fully determined by the frozen
    metric and carries no tunable term.
    """
    values = np.zeros(MAX_BUDGET, dtype=np.float64)
    for budget in BUDGETS:
        safe = max(denominator[budget], DENOMINATOR_FLOOR)
        values[:budget] += DISCOUNT[budget] / (DISCOUNT_SUM[budget] * safe)
    return values / len(BUDGETS)


def normalized_ranked_utility(ordered_utility: np.ndarray, references: QueryReferences) -> dict[int, float]:
    """``nRU@B`` for every deployment budget.  Not clipped: worse than random is negative."""
    output: dict[int, float] = {}
    for budget in BUDGETS:
        raw = ranked_utility(ordered_utility, budget)
        safe = max(references.denominator[budget], DENOMINATOR_FLOOR)
        output[budget] = (raw - references.random_reference) / safe
    return output


def mbru_from_nru(values: dict[int, float]) -> float:
    """Equal-weight mean of ``nRU@10``, ``nRU@20`` and ``nRU@50``."""
    return float(sum(values[budget] for budget in BUDGETS) / len(BUDGETS))


def mbru_of_order(order: np.ndarray, utility_row: np.ndarray, references: QueryReferences) -> float:
    """MBRU of an explicit candidate ordering.  Used as the brute-force reference."""
    return mbru_from_nru(normalized_ranked_utility(np.asarray(utility_row, dtype=np.float64)[order], references))


def pair_swap_value(
    coefficients: np.ndarray, rank_high: np.ndarray, rank_low: np.ndarray,
    utility_high: np.ndarray, utility_low: np.ndarray,
) -> np.ndarray:
    """``max(0, (a(r_j) - a(r_i)) * (U_i - U_j))`` for utility-consistent pairs.

    ``rank_high`` holds the current 1-based predicted rank of the higher-utility
    member ``i`` and ``rank_low`` that of the lower-utility member ``j``.  Ranks
    past the largest deployment budget contribute coefficient zero, so a pair
    that is entirely outside every actionable budget has no decision value.
    """
    gap = np.asarray(utility_high, dtype=np.float64) - np.asarray(utility_low, dtype=np.float64)
    if (gap < 0).any():
        raise ValueError("pair_swap_value requires utility-consistent pairs with U_i >= U_j")
    padded = np.concatenate((coefficients, [0.0]))
    index_high = np.minimum(np.asarray(rank_high, dtype=np.int64) - 1, MAX_BUDGET)
    index_low = np.minimum(np.asarray(rank_low, dtype=np.int64) - 1, MAX_BUDGET)
    if (index_high < 0).any() or (index_low < 0).any():
        raise ValueError("predicted ranks are 1-based")
    return np.maximum(0.0, (padded[index_low] - padded[index_high]) * gap)


@dataclass(frozen=True)
class DirectionMetrics:
    """Per-query nRU and MBRU for one query direction of one model."""

    nru: dict[int, np.ndarray]
    mbru: np.ndarray
    valid_queries: int


def direction_metrics(scores: np.ndarray, utility: np.ndarray, ids: np.ndarray, references: list[QueryReferences]) -> DirectionMetrics:
    """Replay one model's full score matrix under the frozen contract."""
    scores = np.asarray(scores, dtype=np.float64)
    utility = np.asarray(utility, dtype=np.float64)
    if scores.shape != utility.shape or scores.shape[0] != len(references) or scores.shape[1] != len(ids):
        raise ValueError("scores, utility, references and ids must describe one aligned direction")
    nru = {budget: np.empty(len(references), dtype=np.float64) for budget in BUDGETS}
    mbru = np.empty(len(references), dtype=np.float64)
    for query in range(len(references)):
        order = legal_order(scores[query], ids, query)
        values = normalized_ranked_utility(utility[query, order], references[query])
        for budget in BUDGETS:
            nru[budget][query] = values[budget]
        mbru[query] = mbru_from_nru(values)
    if not np.isfinite(mbru).all():
        raise FloatingPointError("MBRU replay produced a non-finite value")
    return DirectionMetrics(nru, mbru, int(len(references)))


def identifiability(references: list[QueryReferences]) -> dict:
    """Section 7 gate: the nRU denominator must not be degenerate.

    A denominator near zero means the oracle ordering is indistinguishable from
    a random one for that query, so nRU is not identifiable there.  The contract
    stops the comparison rather than reporting a ratio with no scale.
    """
    record: dict = {"max_degenerate_fraction_allowed": MAX_DEGENERATE_FRACTION, "budgets": {}}
    degenerate_any = np.zeros(len(references), dtype=bool)
    for budget in BUDGETS:
        values = np.asarray([item.denominator[budget] for item in references], dtype=np.float64)
        degenerate = values < DENOMINATOR_FLOOR
        degenerate_any |= degenerate
        record["budgets"][str(budget)] = {
            "min": float(values.min()), "p01": float(np.quantile(values, 0.01)),
            "p10": float(np.quantile(values, 0.10)), "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.90)), "max": float(values.max()),
            "mean": float(values.mean()),
            "degenerate_count": int(degenerate.sum()),
            "degenerate_fraction": float(degenerate.mean()),
        }
    record["degenerate_any_budget_fraction"] = float(degenerate_any.mean())
    record["passed"] = bool(record["degenerate_any_budget_fraction"] <= MAX_DEGENERATE_FRACTION)
    return record
