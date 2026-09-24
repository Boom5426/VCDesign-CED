"""One grading utility, four orderings: the endpoint this phase is judged on.

Every policy proposes an ordering and nothing else.  The gain that enters RU is
the same cross-fitted realized ``J`` matrix for all of them, because the
attenuation is chosen per ``(query, candidate)`` pair on the proposing view and
never depends on which policy selected that pair.  So the random reference, the
oracle and the nRU denominator are identical across policies too, and the paired
differences below are differences of orderings under one physical quantity
rather than comparisons across two normalizations.

That is the whole point of the design: the previous phase could not compare a
cosine utility against an endpoint utility without each being graded by its own
oracle, and the comparison inherited every difference in row shape.  Here there
is one oracle.
"""
from __future__ import annotations

import numpy as np

from ..model.decision_alignment_v1.contract import (
    BUDGETS, DISCOUNT, DISCOUNT_SUM, mbru_from_nru, normalized_ranked_utility, query_references,
)
from ..utility_gt_v2.validate import cluster_bootstrap, legal_orders, summarize


def raw_ranked_utility(order: np.ndarray, gain: np.ndarray) -> dict[int, np.ndarray]:
    """``RU@B`` in the physical units of ``J``: discounted mean realized improvement."""
    rows = np.arange(order.shape[0])[:, None]
    output = {}
    for budget in BUDGETS:
        selected = gain[rows, order[:, :budget]]
        output[budget] = selected @ DISCOUNT[budget] / DISCOUNT_SUM[budget]
    return output


def normalized(order: np.ndarray, gain: np.ndarray, ids: np.ndarray,
               self_position: np.ndarray) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """``nRU@B`` and MBRU against the same per-query random and oracle references."""
    queries = order.shape[0]
    nru = {budget: np.empty(queries, dtype=np.float64) for budget in BUDGETS}
    mbru = np.empty(queries, dtype=np.float64)
    for row in range(queries):
        references = query_references(gain[row], ids, int(self_position[row]))
        values = normalized_ranked_utility(gain[row, order[row]], references)
        for budget in BUDGETS:
            nru[budget][row] = values[budget]
        mbru[row] = mbru_from_nru(values)
    return nru, mbru


def policy_report(order: np.ndarray, gain: np.ndarray, ids: np.ndarray,
                  self_position: np.ndarray) -> dict:
    raw = raw_ranked_utility(order, gain)
    nru, mbru = normalized(order, gain, ids, self_position)
    record = {f"RU@{budget}": summarize(raw[budget]) for budget in BUDGETS}
    record.update({f"nRU@{budget}": summarize(nru[budget]) for budget in BUDGETS})
    record["MBRU"] = summarize(mbru)
    record["_raw"] = raw
    record["_nru"] = nru
    record["_mbru"] = mbru
    return record


def paired(better: dict, worse: dict) -> dict:
    """Paired difference of two orderings under the one shared grading utility."""
    record = {}
    for budget in BUDGETS:
        for label, store in (("RU", "_raw"), ("nRU", "_nru")):
            difference = better[store][budget] - worse[store][budget]
            record[f"{label}@{budget}"] = {
                "summary": summarize(difference),
                "fraction_query_improved": float((difference > 0).mean()),
                "bootstrap": cluster_bootstrap(difference),
            }
    difference = better["_mbru"] - worse["_mbru"]
    record["MBRU"] = {"summary": summarize(difference),
                      "fraction_query_improved": float((difference > 0).mean()),
                      "bootstrap": cluster_bootstrap(difference)}
    return record


def _percentile(values: np.ndarray) -> np.ndarray:
    output = np.empty(values.size, dtype=np.float64)
    output[np.argsort(values, kind="stable")] = np.arange(values.size) / max(values.size - 1, 1)
    return output


def prefix_behaviour(order: np.ndarray, gain: np.ndarray, alignment: np.ndarray,
                     target_energy: np.ndarray, candidate_energy: np.ndarray,
                     cosine: np.ndarray, action: np.ndarray,
                     proposing_energy: np.ndarray) -> dict:
    """What each policy's budget prefix physically is, read on the grading view only.

    ``target_axis_progress`` is the fraction of the way to the goal along the
    goal's own axis, ``a <r_c, d_q> / ||d_q||^2``, so 1.0 would be a candidate
    whose attenuated response projects exactly onto the target.
    """
    rows = np.arange(order.shape[0])[:, None]
    percentile = _percentile(candidate_energy)
    proposing_percentile = _percentile(proposing_energy)
    usable = target_energy > 0.0
    record = {}
    for budget in BUDGETS:
        picked = order[:, :budget]
        realized = gain[rows, picked]
        progress = np.where(usable[:, None],
                            action[rows, picked] * alignment[rows, picked] / np.where(
                                usable[:, None], target_energy[:, None], 1.0), np.nan)
        record[str(budget)] = {
            "fraction_J_positive": float((realized > 0).mean()),
            "per_query_fraction_J_positive_median": float(np.median((realized > 0).mean(axis=1))),
            "realized_J_median": float(np.median(realized)),
            "realized_J_mean": float(realized.mean()),
            "cross_view_cosine_median": float(np.median(cosine[rows, picked])),
            "target_axis_progress_median": float(np.nanmedian(progress)),
            "candidate_response_energy_percentile_median": float(np.median(percentile[picked])),
            # The winner's-curse readout.  A selection driven by favourable noise in
            # the selecting view's energy estimate shows a low percentile here and a
            # middling one under an independent measurement of the same candidates.
            "response_energy_percentile_in_the_selecting_view_median": float(
                np.median(proposing_percentile[picked])),
            "response_energy_median_selecting_view": float(np.median(proposing_energy[picked])),
            "response_energy_median_grading_view": float(np.median(candidate_energy[picked])),
        }
    record["queries_with_non_positive_target_energy"] = float((~usable).mean())
    return record


def action_behaviour(order: np.ndarray, action: np.ndarray) -> dict:
    """Where the chosen attenuation actually lands inside the frozen interval."""
    rows = np.arange(order.shape[0])[:, None]
    record = {}
    for budget in BUDGETS:
        chosen = action[rows, order[:, :budget]]
        record[str(budget)] = {
            "fraction_at_zero": float((chosen == 0.0).mean()),
            "fraction_interior": float(((chosen > 0.0) & (chosen < 1.0)).mean()),
            "fraction_at_one": float((chosen == 1.0).mean()),
            "median": float(np.median(chosen)),
            "p25": float(np.quantile(chosen, 0.25)), "p75": float(np.quantile(chosen, 0.75)),
        }
    record["whole_matrix"] = {
        "fraction_at_zero": float((action == 0.0).mean()),
        "fraction_interior": float(((action > 0.0) & (action < 1.0)).mean()),
        "fraction_at_one": float((action == 1.0).mean()),
    }
    return record


def set_overlap(first: np.ndarray, second: np.ndarray) -> dict:
    """Diagnostic only; this phase does not judge on Top-B identity overlap."""
    record = {}
    for budget in BUDGETS:
        overlap = np.asarray([
            len(set(first[row, :budget].tolist()) & set(second[row, :budget].tolist())) / budget
            for row in range(first.shape[0])], dtype=np.float64)
        record[str(budget)] = {"median": float(np.median(overlap)), "mean": float(overlap.mean())}
    return record
