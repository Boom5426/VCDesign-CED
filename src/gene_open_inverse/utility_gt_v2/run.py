#!/usr/bin/env python3
"""Phase 1 and Phase 2 of Utility-GT V2, in one run that writes one record.

Phase 1 freezes query eligibility from the prescribed rule and applies it to
both roles unchanged.  Phase 2 asks whether a budget prefix ranked by one
measurement view keeps its experimental utility under an independent view, for
the latent value and for the incumbent cosine, on identical queries, candidates
and cells.

Nothing here trains, and nothing here reads a model score.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..model.decision_alignment_v1.assets import MODALITIES, AuditConfig
from ..model.decision_alignment_v1.contract import BUDGETS
from ..model.global_objective_knowledge_attribution_v1.assets import GlobalAttributionAssets
from . import contract as gt
from . import validate as val


SIDE_QUARTERS = {0: (0, 1), 1: (2, 3)}
DIRECTIONS = (("first_ranks_second_grades", 0, 1), ("second_ranks_first_grades", 1, 0))


def _distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(values)), "mean": float(values.mean()),
        "min": float(values.min()), "max": float(values.max()),
        "p05": float(np.quantile(values, 0.05)), "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)), "p95": float(np.quantile(values, 0.95)),
        "fraction_non_positive": float((values <= 0).mean()),
    }


def phase_one(transitions: np.ndarray) -> tuple[gt.Eligibility, dict]:
    """Freeze-rule eligibility at production A/B depth, with its full audit trail."""
    gram = gt.cross_view_gram(transitions[0], transitions[1])
    primary = gt.eligibility(gram, gt.PRIMARY_FDR)
    record = {
        "rule": {
            "metric_space": gt.METRIC_SPACE, "statistic": gt.ELIGIBILITY_STATISTIC,
            "null": gt.ELIGIBILITY_NULL, "primary_fdr": gt.PRIMARY_FDR,
            "require_positive_energy": gt.REQUIRE_POSITIVE_ENERGY,
            "gate_selected_from_G_select": gt.GATE_SELECTED_FROM_G_SELECT,
        },
        "nominal_queries": primary.count,
        "eligible_queries": primary.eligible_count,
        "eligibility_fraction": primary.fraction,
        "passes_fdr_only": int(primary.passes_fdr.sum()),
        "positive_energy_only": int(primary.positive_energy.sum()),
        "energy_distribution": _distribution(primary.energy),
        "energy_distribution_eligible": _distribution(primary.energy[primary.eligible]) if primary.eligible.any() else None,
        "p_value_distribution": {
            "median": float(np.median(primary.p_value)), "min": float(primary.p_value.min()),
            "fraction_below_0.05": float((primary.p_value < 0.05).mean()),
            "fraction_at_resolution_floor": float((primary.p_value <= 1.0 / primary.count + 1e-12).mean()),
            "resolution_floor": 1.0 / primary.count,
        },
        "q_value_distribution": {
            "median": float(np.median(primary.q_value)), "min": float(primary.q_value.min()),
        },
        "bh_p_cutoff_at_primary_fdr": gt.bh_p_cutoff(primary.p_value, gt.PRIMARY_FDR),
        "sensitivity": {
            str(level): {
                "eligible_queries": int(gt.eligibility(gram, level).eligible_count),
                "bh_p_cutoff": gt.bh_p_cutoff(primary.p_value, level),
            }
            for level in gt.SENSITIVITY_FDR
        },
    }
    return primary, record


def phase_two(quarter: np.ndarray, transitions: np.ndarray, ids: np.ndarray,
              candidate_rows: np.ndarray, query_in_candidate: np.ndarray) -> dict:
    """Cross-view decision reproducibility of the latent value and of the cosine."""
    candidate_ids = ids[candidate_rows]
    quarters = [np.asarray(quarter[index], dtype=np.float64)[candidate_rows] for index in range(4)]
    sides = [np.asarray(transitions[side], dtype=np.float64)[candidate_rows] for side in (0, 1)]

    latent = [gt.latent_value(quarters[SIDE_QUARTERS[side][0]], quarters[SIDE_QUARTERS[side][1]]) for side in (0, 1)]
    cosine = [gt.cosine_value(sides[side]) for side in (0, 1)]
    energy = [np.ascontiguousarray(np.diag(gt.cross_view_gram(
        quarters[SIDE_QUARTERS[side][0]], quarters[SIDE_QUARTERS[side][1]]))).copy() for side in (0, 1)]

    self_position = np.asarray(query_in_candidate, dtype=np.int64)
    estimators = {"latent_value": latent, "incumbent_cosine": cosine}
    rows = self_position

    record: dict = {
        "queries": int(self_position.size), "candidates": int(candidate_rows.size),
        "note": "same queries, same candidates, same cells; A/B for the cosine and (Q0,Q1)/(Q2,Q3) for the latent value",
        "estimators": {}, "paired_latent_minus_cosine": {}, "secondary": {},
    }

    graded: dict[str, dict[str, val.GradedRun]] = {}
    orders: dict[str, list[np.ndarray]] = {}
    for name, pair in estimators.items():
        orders[name] = [val.legal_orders(pair[side][rows], candidate_ids, self_position) for side in (0, 1)]
        graded[name] = {}
        entry: dict = {}
        for label, proposing, grading in DIRECTIONS:
            run = val.grade(orders[name][proposing], pair[grading][rows], candidate_ids, self_position)
            graded[name][label] = run
            entry[label] = val.graded_summary(run)
        both = np.mean([graded[name][label].mbru for label, _, _ in DIRECTIONS], axis=0)
        entry["MBRU_both_directions"] = val.summarize(both)
        record["estimators"][name] = entry
    record["per_query"] = {
        f"{name}__{label}": graded[name][label].mbru for name in estimators for label, _, _ in DIRECTIONS
    }
    record["per_query"]["order_first_latent_value"] = orders["latent_value"][0]
    record["per_query"]["order_first_incumbent_cosine"] = orders["incumbent_cosine"][0]

    for label, _, _ in DIRECTIONS:
        difference = graded["latent_value"][label].mbru - graded["incumbent_cosine"][label].mbru
        record["paired_latent_minus_cosine"][label] = {
            "MBRU": val.summarize(difference), "bootstrap": val.cluster_bootstrap(difference),
            "nRU": {
                f"@{budget}": val.summarize(
                    graded["latent_value"][label].nru[budget] - graded["incumbent_cosine"][label].nru[budget]
                ) for budget in BUDGETS
            },
        }
    averaged = np.mean([
        graded["latent_value"][label].mbru - graded["incumbent_cosine"][label].mbru for label, _, _ in DIRECTIONS
    ], axis=0)
    record["paired_latent_minus_cosine"]["both_directions"] = {
        "MBRU": val.summarize(averaged), "bootstrap": val.cluster_bootstrap(averaged),
    }

    for name, pair in estimators.items():
        secondary = {
            "set_overlap": val.set_overlap(orders[name][0], orders[name][1]),
            "boundary_flip": {
                label: val.boundary_flip(orders[name][proposing], pair[proposing][rows], pair[grading][rows])
                for label, proposing, grading in DIRECTIONS
            },
            "spearman": val.spearman(pair[0][rows], pair[1][rows], self_position),
        }
        prior_order = val.goal_independent_order(pair[0][rows], candidate_ids, self_position)
        prior_run = val.grade(prior_order, pair[1][rows], candidate_ids, self_position)
        secondary["goal_independent_prior"] = val.graded_summary(prior_run)
        secondary["goal_conditioned_advantage_MBRU"] = val.summarize(
            graded[name]["first_ranks_second_grades"].mbru - prior_run.mbru
        )
        record["secondary"][name] = secondary

    record["secondary"]["latent_value"]["positive_value_candidates"] = {
        "median_per_query": float(np.median((latent[0][rows] > 0).sum(axis=1))),
        "mean_fraction": float((latent[0][rows] > 0).mean()),
    }
    record["secondary"]["latent_value"]["physical_bound_violation_fraction"] = float(
        gt.physical_bound_violation(latent[0][rows], energy[0][rows]).mean()
    )
    record["secondary"]["latent_value"]["quarter_energy_non_positive_fraction"] = {
        "side_A": float((energy[0][rows] <= 0).mean()), "side_B": float((energy[1][rows] <= 0).mean()),
    }
    return record


def run(args: argparse.Namespace) -> dict:
    asset = Path(args.asset)
    quarter_all = np.load(asset / "quarter_responses.npy", mmap_mode="r")
    identity_axis = np.asarray([str(value) for value in np.load(asset / "identity_axis.npy", allow_pickle=True)])
    four_way = np.load(asset / "four_way_eligible.npy")
    config = AuditConfig.load(Path(args.config))
    assets = GlobalAttributionAssets(modalities=MODALITIES, **config.asset_arguments())
    axis_index = {name: index for index, name in enumerate(identity_axis.tolist())}

    result: dict = {
        "schema": "VCDESIGN_UTILITY_GT_V2_PHASE12_V1",
        "trained_anything": False, "read_G_check": False,
        "gate_selected_from_G_select": gt.GATE_SELECTED_FROM_G_SELECT,
        "divided_by_no_op_energy": False, "clipped_utility": False,
        "phase_1": {}, "phase_2": {},
    }

    eligible_by_role: dict[str, gt.Eligibility] = {}
    for role_name in ("G_fit", "G_select"):
        role = assets.role_arrays(role_name)
        transitions = np.asarray(role.base.transitions, dtype=np.float64)
        primary, record = phase_one(transitions)
        eligible_by_role[role_name] = primary
        result["phase_1"][role_name] = record

    cutoff = result["phase_1"]["G_fit"]["bh_p_cutoff_at_primary_fdr"]
    select = eligible_by_role["G_select"]
    result["phase_1"]["G_select"]["sensitivity_fixed_p_cutoff_frozen_on_G_fit"] = {
        "cutoff": cutoff,
        "eligible_queries": int(((select.p_value <= cutoff) & (select.energy > 0.0)).sum()),
        "note": "reported only; the primary rule is BH-FDR applied inside each role",
    }

    role = assets.role_arrays("G_select")
    ids = np.asarray([str(value) for value in role.base.ids])
    position = np.asarray([axis_index[name] for name in ids.tolist()], dtype=np.int64)
    duplicates = {name: int(count) for name, count in zip(*np.unique(ids, return_counts=True)) if count > 1}
    survivor = four_way[position]
    candidate_rows = np.flatnonzero(survivor)
    query_in_candidate = np.flatnonzero(select.eligible[candidate_rows])

    quarter = np.asarray(quarter_all[:, position, :], dtype=np.float32)
    transitions = np.asarray(role.base.transitions, dtype=np.float64)
    phase_two_record = phase_two(quarter, transitions, ids, candidate_rows, query_in_candidate)
    per_query = phase_two_record.pop("per_query")
    result["phase_2"] = phase_two_record
    result["phase_2"]["pool"] = {
        "nominal": int(len(ids)), "four_way_survivors": int(survivor.sum()),
        "eligible_of_nominal": int(select.eligible.sum()),
        "eligible_and_four_way": int(query_in_candidate.size),
        "duplicate_identity_ids": duplicates,
    }

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    result["runtime"] = {"timestamp": datetime.now(timezone.utc).isoformat(),
                         "python": platform.python_version(), "numpy": np.__version__}
    result["provenance"] = config.provenance()
    (output / "utility_gt_v2_phase12.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    np.save(output / "G_select_eligible.npy", select.eligible)
    np.save(output / "G_select_p_value.npy", select.p_value)
    np.save(output / "G_select_q_value.npy", select.q_value)
    np.save(output / "G_select_energy.npy", select.energy)
    np.save(output / "G_select_ids.npy", ids)
    for name, values in per_query.items():
        np.save(output / f"per_query__{name}.npy", np.asarray(values))
    np.save(output / "phase2_candidate_rows.npy", candidate_rows)
    np.save(output / "phase2_query_in_candidate.npy", query_in_candidate)
    np.save(output / "G_fit_eligible.npy", eligible_by_role["G_fit"].eligible)
    np.save(output / "G_fit_p_value.npy", eligible_by_role["G_fit"].p_value)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Utility-GT V2 Phase 1 and Phase 2")
    parser.add_argument("--asset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args)
    for role_name, record in result["phase_1"].items():
        print(f"PHASE1 {role_name}: eligible={record['eligible_queries']}/{record['nominal_queries']} "
              f"({record['eligibility_fraction']:.4f})")
    pool = result["phase_2"]["pool"]
    print(f"PHASE2 pool: queries={pool['eligible_and_four_way']} candidates={result['phase_2']['candidates']}")
    for name, entry in result["phase_2"]["estimators"].items():
        print(f"  {name}: MBRU both directions median={entry['MBRU_both_directions']['median']:.4f} "
              f"mean={entry['MBRU_both_directions']['mean']:.4f}")
    both = result["phase_2"]["paired_latent_minus_cosine"]["both_directions"]
    print(f"  paired latent-minus-cosine MBRU median={both['MBRU']['median']:.4f} "
          f"mean CI95={both['bootstrap']['mean']['ci95']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
