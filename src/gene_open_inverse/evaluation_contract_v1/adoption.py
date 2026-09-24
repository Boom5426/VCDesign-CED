"""The frozen adoption gate (contract Section 5): a pure function of pooled numbers."""
from __future__ import annotations

from . import contract as ec


def decide(values: dict) -> dict:
    """``values`` holds pooled MASKED paired contrasts of the new method against the incumbent.

    Required keys: ``bu20`` {median, state}, ``bu20_by_context`` {name: {median, ci95}},
    ``mu20_state``, ``hvhit20_state``, ``phr20_state`` (None when unavailable),
    ``qwr_gate20`` (float).
    """
    by_context = values["bu20_by_context"]
    checks = {
        "G1_BU20_positive": values["bu20"]["state"] == ec.POSITIVE,
        "G2_BU20_at_least_minimum_worthwhile_gain": values["bu20"]["median"] >= ec.MIN_WORTHWHILE_GAIN,
        "G3_two_of_three_contexts_positive": sum(v["median"] > 0 for v in by_context.values()) >= ec.CONTEXT_MAJORITY,
        "G4_no_context_wholly_negative": not any(v["ci95"][1] < 0 for v in by_context.values()),
        "G5_MU20_not_reversed": values["mu20_state"] != ec.NEGATIVE,
        "G6_HvHit20_and_PHR20_not_worse": values["hvhit20_state"] != ec.NEGATIVE
                                          and values.get("phr20_state") != ec.NEGATIVE,
        "G7_QWR_gate20_above_half": values["qwr_gate20"] > ec.QWR_GATE,
    }
    failed = [name for name, ok in checks.items() if not ok]
    return {"outcome": ec.ADOPT if not failed else ec.KEEP, "checks": checks, "failed": failed,
            "phr_available": values.get("phr20_state") is not None}
