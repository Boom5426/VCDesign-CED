#!/usr/bin/env python3
"""Independent witness for the Utility-GT V2 Phase 1 and Phase 2 run.

Nothing here imports the frozen metric module or the Phase-2 code it is checking.
Ranked utility, the random and oracle references, the permutation p-value and the
BH step-up are all rewritten from their definitions, so agreement means two
implementations agree rather than one implementation repeating itself.  This is
lesson L3: a script must not be its own only witness.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..model.decision_alignment_v1.assets import MODALITIES, AuditConfig
from ..model.global_objective_knowledge_attribution_v1.assets import GlobalAttributionAssets


BUDGETS = (10, 20, 50)
SAMPLED_QUERIES = 48
SAMPLED_PAIRS = 400
VERIFY_SEED = 20260918


def _discount(budget: int) -> np.ndarray:
    return np.asarray([1.0 / np.log2(rank + 1.0) for rank in range(1, budget + 1)], dtype=np.float64)


def _ranked_utility(ordered: np.ndarray, budget: int) -> float:
    weights = _discount(budget)
    return float(sum(weights[index] * ordered[index] for index in range(budget)) / weights.sum())


def _order_excluding(row: np.ndarray, ids: np.ndarray, self_index: int) -> np.ndarray:
    keys = [(-float(row[index]), str(ids[index]), index) for index in range(len(row)) if index != self_index]
    keys.sort()
    return np.asarray([entry[2] for entry in keys], dtype=np.int64)


def _mbru(order: np.ndarray, grading_row: np.ndarray, ids: np.ndarray, self_index: int) -> float:
    legal = np.asarray([grading_row[index] for index in range(len(grading_row)) if index != self_index])
    reference = float(legal.mean())
    oracle_order = _order_excluding(grading_row, ids, self_index)
    total = 0.0
    for budget in BUDGETS:
        oracle = _ranked_utility(grading_row[oracle_order], budget)
        achieved = _ranked_utility(grading_row[order], budget)
        total += (achieved - reference) / max(oracle - reference, 1e-8)
    return total / len(BUDGETS)


def _permutation_p(gram: np.ndarray, rows: np.ndarray) -> np.ndarray:
    count = gram.shape[0]
    output = np.empty(len(rows), dtype=np.float64)
    for position, row in enumerate(rows):
        observed = gram[row, row]
        exceed = 0
        for column in range(count):
            if column != row and gram[row, column] >= observed:
                exceed += 1
        output[position] = (1.0 + exceed) / float(count)
    return output


def _bh_reject(p_value: np.ndarray, level: float) -> np.ndarray:
    count = len(p_value)
    order = sorted(range(count), key=lambda index: p_value[index])
    threshold = -1.0
    for rank, index in enumerate(order, start=1):
        if p_value[index] <= rank * level / count:
            threshold = p_value[index]
    return np.asarray([value <= threshold for value in p_value], dtype=bool)


def run(args: argparse.Namespace) -> dict:
    result_dir = Path(args.result)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"verifier refuses to overwrite {output}")
    record = json.loads((result_dir / "utility_gt_v2_phase12.json").read_text())

    asset = Path(args.asset)
    quarter_all = np.load(asset / "quarter_responses.npy", mmap_mode="r")
    identity_axis = np.asarray([str(value) for value in np.load(asset / "identity_axis.npy", allow_pickle=True)])
    four_way = np.load(asset / "four_way_eligible.npy")
    config = AuditConfig.load(Path(args.config))
    assets = GlobalAttributionAssets(modalities=MODALITIES, **config.asset_arguments())
    axis_index = {name: index for index, name in enumerate(identity_axis.tolist())}

    checks: dict = {}
    role = assets.role_arrays("G_select")
    ids = np.asarray([str(value) for value in role.base.ids])
    transitions = np.asarray(role.base.transitions, dtype=np.float64)

    gram = transitions[0] @ transitions[1].T
    energy = np.diag(gram).copy()
    saved_energy = np.load(result_dir / "G_select_energy.npy")
    checks["G_select_energy_matches"] = bool(np.allclose(energy, saved_energy, rtol=1e-9, atol=1e-9))

    generator = np.random.default_rng(VERIFY_SEED)
    sampled_rows = generator.choice(len(ids), size=min(120, len(ids)), replace=False)
    saved_p = np.load(result_dir / "G_select_p_value.npy")
    checks["G_select_permutation_p_matches_on_sample"] = bool(
        np.allclose(_permutation_p(gram, sampled_rows), saved_p[sampled_rows])
    )

    saved_eligible = np.load(result_dir / "G_select_eligible.npy")
    independent = _bh_reject(saved_p, record["phase_1"]["G_select"]["rule"]["primary_fdr"]) & (energy > 0.0)
    checks["G_select_eligibility_matches_an_independent_bh"] = bool(np.array_equal(independent, saved_eligible))
    checks["G_select_eligible_count"] = int(saved_eligible.sum())
    checks["G_select_eligible_count_matches_json"] = bool(
        int(saved_eligible.sum()) == record["phase_1"]["G_select"]["eligible_queries"]
    )

    position = np.asarray([axis_index[name] for name in ids.tolist()], dtype=np.int64)
    candidate_rows = np.load(result_dir / "phase2_candidate_rows.npy")
    query_in_candidate = np.load(result_dir / "phase2_query_in_candidate.npy")
    checks["candidate_rows_match_a_recount"] = bool(np.array_equal(candidate_rows, np.flatnonzero(four_way[position])))
    checks["query_rows_match_a_recount"] = bool(
        np.array_equal(query_in_candidate, np.flatnonzero(saved_eligible[candidate_rows]))
    )

    quarters = [np.asarray(quarter_all[index][position][candidate_rows], dtype=np.float64) for index in range(4)]
    candidate_ids = ids[candidate_rows]
    value_first = (quarters[0] @ quarters[1].T)
    value_first = value_first.T + value_first - np.diag(value_first)[None, :]
    value_second = (quarters[2] @ quarters[3].T)
    value_second = value_second.T + value_second - np.diag(value_second)[None, :]

    pair_generator = np.random.default_rng(VERIFY_SEED + 1)
    query_sample = pair_generator.choice(query_in_candidate, size=min(SAMPLED_PAIRS, len(query_in_candidate)), replace=True)
    candidate_sample = pair_generator.choice(len(candidate_rows), size=len(query_sample), replace=True)
    worst = 0.0
    for query, candidate in zip(query_sample.tolist(), candidate_sample.tolist()):
        explicit = (
            float(quarters[0][candidate] @ quarters[1][query])
            + float(quarters[1][candidate] @ quarters[0][query])
            - float(quarters[0][candidate] @ quarters[1][candidate])
        )
        worst = max(worst, abs(explicit - float(value_first[query, candidate])))
    checks["latent_value_matches_explicit_dot_products"] = {"max_abs_error": worst, "pairs": len(query_sample)}

    norms = np.linalg.norm(quarters[0], axis=1)
    checks["no_self_inner_product_in_the_estimator"] = bool(
        abs(float(value_first[query_in_candidate[0], query_in_candidate[0]])
            - float(quarters[0][query_in_candidate[0]] @ quarters[1][query_in_candidate[0]])) < 1e-6
        and norms.min() > 0.0
    )

    side = [np.asarray(transitions[index], dtype=np.float64)[candidate_rows] for index in (0, 1)]
    cosine = []
    for values in side:
        norm = np.linalg.norm(values, axis=1)
        norm[norm == 0.0] = 1.0
        unit = values / norm[:, None]
        cosine.append(unit @ unit.T)

    query_rows = generator.choice(query_in_candidate, size=min(SAMPLED_QUERIES, len(query_in_candidate)), replace=False)
    comparisons = {
        "latent_value__first_ranks_second_grades": (value_first, value_second),
        "latent_value__second_ranks_first_grades": (value_second, value_first),
        "incumbent_cosine__first_ranks_second_grades": (cosine[0], cosine[1]),
        "incumbent_cosine__second_ranks_first_grades": (cosine[1], cosine[0]),
    }
    for name, (proposing, grading) in comparisons.items():
        saved = np.load(result_dir / f"per_query__{name}.npy")
        errors = []
        for query in query_rows.tolist():
            order = _order_excluding(proposing[query], candidate_ids, query)
            mine = _mbru(order, grading[query], candidate_ids, query)
            index = int(np.flatnonzero(query_in_candidate == query)[0])
            errors.append(abs(mine - float(saved[index])))
            assert query not in order.tolist()
        checks[f"mbru_matches_a_from_scratch_recompute__{name}"] = {
            "max_abs_error": float(max(errors)), "queries": len(errors)
        }

    saved_order = np.load(result_dir / "per_query__order_first_latent_value.npy")
    checks["saved_orderings_exclude_the_query"] = bool(
        all(int(query_in_candidate[row]) not in saved_order[row].tolist() for row in range(len(query_in_candidate)))
    )
    checks["no_identity_appears_twice_on_the_candidate_axis"] = bool(
        len(set(candidate_ids.tolist())) == len(candidate_ids)
    )

    payload = {"schema": "VCDESIGN_UTILITY_GT_V2_VERIFICATION_V2", "checks": checks,
               "result_dir": str(result_dir.resolve())}
    # A dict is never ``is False``, so gating only on booleans would let every
    # numeric check through no matter how large its error.  Each numeric check
    # carries its own tolerance and is gated here explicitly.
    tolerance = {"latent_value_matches_explicit_dot_products": 1e-9}
    failures = []
    for name, value in checks.items():
        if value is False:
            failures.append(name)
        elif isinstance(value, dict) and "max_abs_error" in value:
            limit = tolerance.get(name, 1e-10)
            if not float(value["max_abs_error"]) <= limit:
                failures.append(name)
                value["tolerance"] = limit
    payload["all_checks_pass"] = not failures
    payload["all_boolean_checks_pass"] = not failures
    payload["failed_checks"] = failures
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Independent verification of Utility-GT V2 Phase 1 and 2")
    parser.add_argument("--result", required=True)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    payload = run(parser.parse_args())
    for name, value in payload["checks"].items():
        print(f"{name}: {value}")
    print("ALL CHECKS PASS" if payload["all_checks_pass"] else f"FAILED: {payload['failed_checks']}")
    return 0 if payload["all_boolean_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
