"""The frozen PPM decision (protocol Section 8), a pure function of pooled numbers."""
from __future__ import annotations

from . import contract as pc


def decision_input(pooled: dict, candidates: dict, per_fold: dict, records: dict, arm: str) -> dict:
    """Collect exactly the numbers the decision reads, for one PPM arm against RIDGE_UNIT."""
    label = f"{arm}_minus_{pc.RIDGE}"
    masked, seen = pooled["MASKED"], pooled["SEEN"]
    by_context = {}
    for name in pc.PRIMARY_CONTEXTS:
        entry = records[name]["MASKED"]["fused"]["contrasts"][label]["MBRU"]
        by_context[name] = {"median": entry["median"], "ci95": entry["query_identity"]["median_ci95"]}
    return {
        "arm": arm,
        "masked_fused": {k: masked["fused"][label]["MBRU"][k] for k in ("median", "median_ci95", "state")},
        "seen_fused_median": seen["fused"][label]["MBRU"]["median"],
        "masked_by_context": by_context,
        "masked_mean_at": {m: masked["fused"][label][m]["mean"] for m in ("Mean@10", "Mean@20", "Mean@50")},
        "full_cosine_state": candidates["MASKED"][f"{arm}|full_cosine"]["state"],
        "specific_cosine_state": candidates["MASKED"][f"{arm}|specific_cosine"]["state"],
        "common_removed_fused_median": masked["fused"][f"COMMON_REMOVED_{label}"]["MBRU"]["median"],
        "effect_only_median": masked["effect_only"][label]["MBRU"]["median"],
        "homogeneity_passes_all": all(records[n]["homogeneity"][arm]["passes"] for n in pc.PRIMARY_CONTEXTS),
        "per_fold_fused_median": per_fold[arm]["median"],
    }


def primary_passes(values: dict) -> bool:
    entry = values["masked_fused"]
    return bool(entry["median"] >= pc.MIN_GAIN and entry["state"] == pc.POSITIVE)


def checks(values: dict) -> dict:
    medians = [v["median"] for v in values["masked_by_context"].values()]
    seen = values["seen_fused_median"]
    masked = values["masked_fused"]["median"]
    return {
        "P1_pooled_masked_gain_at_least_0.010_and_positive": primary_passes(values),
        "R1_two_of_three_contexts_positive": sum(m > 0 for m in medians) >= pc.CONTEXT_MAJORITY,
        "R2_no_catastrophic_context_reversal": not any(
            v["median"] <= pc.REVERSAL_FLOOR and v["ci95"][1] < 0 for v in values["masked_by_context"].values()),
        "R3_mean_at_budget_direction_consistent": all(v > 0 for v in values["masked_mean_at"].values()),
        "R4_masked_not_much_weaker_than_seen": bool(seen <= 0 or masked >= pc.MASKED_TO_SEEN_RATIO * seen),
        "Q1_masked_full_direction_prediction_positive": values["full_cosine_state"] == pc.POSITIVE,
        "Q2_masked_candidate_specific_prediction_positive": values["specific_cosine_state"] == pc.POSITIVE,
        "Q3_gain_survives_common_removal_and_exposure": bool(values["common_removed_fused_median"] > 0
                                                            and values["effect_only_median"] > 0),
        "H_per_fold_holds_if_mixing_gate_fails": bool(values["homogeneity_passes_all"]
                                                      or values["per_fold_fused_median"] > 0),
    }


def decide(values: dict, replicate_primary: bool | None) -> dict:
    """``replicate_primary`` is the seed replicate's P1, or None when it was not run."""
    table = checks(values)
    failed = [name for name, ok in table.items() if not ok]
    if not failed and replicate_primary is True:
        outcome = pc.SUPPORTED
    elif values["masked_fused"]["state"] == pc.POSITIVE and (
            table["Q1_masked_full_direction_prediction_positive"]
            or table["Q2_masked_candidate_specific_prediction_positive"]):
        outcome = pc.PARTIAL
    else:
        outcome = pc.NOT_SUPPORTED
    if not failed and replicate_primary is not True:
        failed.append("seed replicate P1 " + ("not run" if replicate_primary is None else "failed"))
    return {"outcome": outcome, "checks": table, "failed_conditions": failed,
            "replicate_primary": replicate_primary}
