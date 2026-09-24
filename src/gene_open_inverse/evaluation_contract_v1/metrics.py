"""Per-row BU@B, MU@B, HvHit@B, PHR@B and the two QWR forms, on the frozen primitives.

Every row is one query in one view, ranked over the self-excluded legal pool of one candidate
stratum, in the frozen stable tie order.  BU@B is the frozen ``nRU@B``; MU@B and HvHit@B are
the frozen ``Mean@B`` and ``HighValueCount@B``.  The tests assert all three identities
against the existing evaluator rather than assuming them.
"""
from __future__ import annotations

import numpy as np

from ..model.decision_alignment_v1.contract import normalized_ranked_utility, query_references
from ..model.open_vocab_dual_encoder_v1.metrics import stable_order
from . import contract as ec


def row_metrics(scores: np.ndarray, utility: np.ndarray, candidate_ids: np.ndarray,
                self_positions: np.ndarray, realized_j: np.ndarray | None = None) -> dict:
    """``scores``, ``utility`` and ``realized_j`` are ``[rows, candidates_in_stratum]``.

    ``self_positions[i]`` is where row ``i``'s query sits in the stratum, or -1.  PHR is
    returned only when a realized ``J`` matrix is supplied; it is never approximated.
    """
    scores = np.asarray(scores, dtype=np.float64)
    utility = np.asarray(utility, dtype=np.float64)
    if scores.shape != utility.shape or scores.shape[1] != len(candidate_ids):
        raise ValueError("scores, utility and the candidate axis must be aligned")
    if realized_j is not None and np.asarray(realized_j).shape != scores.shape:
        raise ValueError("the realized J matrix must match the score matrix")
    rows = scores.shape[0]
    out = {f"{family}@{b}": np.empty(rows) for family in ("BU", "MU", "HvHit") for b in ec.BUDGETS}
    if realized_j is not None:
        out.update({f"PHR@{b}": np.empty(rows) for b in ec.BUDGETS})
    for row in range(rows):
        position = int(self_positions[row])
        excluded = position if position >= 0 else None
        order = stable_order(scores[row], candidate_ids, excluded)
        if excluded is not None:
            order = order[order != position]
        references = query_references(utility[row], candidate_ids, excluded)
        normalized = normalized_ranked_utility(utility[row, order], references)
        legal = np.delete(utility[row], position) if position >= 0 else utility[row]
        threshold = float(np.quantile(legal, ec.HIGH_VALUE_QUANTILE))
        for budget in ec.BUDGETS:
            picked = utility[row, order[:budget]]
            out[f"BU@{budget}"][row] = normalized[budget]
            out[f"MU@{budget}"][row] = float(picked.mean())
            out[f"HvHit@{budget}"][row] = float((picked >= threshold).sum())
            if realized_j is not None:
                out[f"PHR@{budget}"][row] = float((np.asarray(realized_j)[row, order[:budget]] > 0).mean())
    return out


def qwr(method: np.ndarray, comparator: np.ndarray) -> dict:
    """Both QWR forms for one pair of per-row BU vectors.

    ``paper`` is wins over all rows (the paper-facing form, comparator BASE); ``gate`` is
    wins over decided rows (the adoption form, comparator the incumbent).
    """
    method = np.asarray(method, dtype=np.float64)
    comparator = np.asarray(comparator, dtype=np.float64)
    wins = int((method > comparator).sum())
    losses = int((method < comparator).sum())
    ties = int(method.size - wins - losses)
    return {"rows": int(method.size), "wins": wins, "losses": losses, "ties": ties,
            "tie_fraction": ties / max(method.size, 1),
            "paper": wins / max(method.size, 1),
            "gate": wins / max(wins + losses, 1) if wins + losses else float("nan")}
