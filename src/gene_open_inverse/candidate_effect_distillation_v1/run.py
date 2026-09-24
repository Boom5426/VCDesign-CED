#!/usr/bin/env python3
"""Phases 1 to 6: build the effect predictor and ask whether it ranks.

No measured response of a scored candidate is ever an input.  The predictor sees
``G_fit`` static knowledge and ``G_fit`` measured responses at training time, and a
deployment candidate's static knowledge alone at inference.  ``G_check`` is sealed.

The ladder contains one row that is not deployable and is labelled so:
``MEASURED_EFFECT_COSINE`` applies the same scoring rule to the true response
instead of the predicted one.  It exists to split the oracle gap into the part that
belongs to the scoring rule and the part that belongs to prediction quality, which a
single oracle number cannot do.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ..model.decision_aligned_sampler_v1.evaluate import HIGH_VALUE_QUANTILE, _direction_metrics
from ..model.decision_alignment_v1.assets import AuditConfig
from ..model.decision_alignment_v1.contract import BUDGETS
from ..model.global_objective_knowledge_attribution_v1.assets import GlobalAttributionAssets
from ..model.open_vocab_dual_encoder_v1.data import FrozenTransitionAssets
from ..model.open_vocab_dual_encoder_v1.metrics import stable_order
from ..model.query_conditional_capacity_v1.run import _guardrail
from ..utility_gt_v2.validate import cluster_bootstrap, summarize
from . import basis as bs
from . import contract as ced
from . import predictor as pr


LEGALITY = {
    "RANDOM": "no information",
    "PREDICTED_EFFECT_MAGNITUDE": "strict legal, goal blind",
    "PREDICTED_EFFECT_COSINE": "strict legal, the method",
    "MEASURED_EFFECT_COSINE": "DIAGNOSTIC_ORACLE_NOT_DEPLOYABLE, same rule with the true response",
}


def _unit(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-30)


def _percentile(values: np.ndarray) -> np.ndarray:
    output = np.empty(values.size, dtype=np.float64)
    output[np.argsort(values, kind="stable")] = np.arange(values.size) / max(values.size - 1, 1)
    return output


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    config = AuditConfig.load(Path(args.config))
    assets = GlobalAttributionAssets(modalities=ced.LEGACY_MODALITIES, **config.asset_arguments())
    fit_role, select_role = assets.role_arrays("G_fit"), assets.role_arrays("G_select")
    select_ids = np.asarray([str(v) for v in select_role.base.ids])
    rows = np.flatnonzero(np.load(Path(args.eligibility) / "G_select_eligible.npy"))

    # --- Phase 1: response target, G_fit only ---------------------------------
    fit_transitions = np.asarray(fit_role.base.transitions, dtype=np.float64)
    supervision = np.arange(fit_transitions.shape[1])
    if args.eligible_targets:
        # Declared variant, motivated by a measurement rather than by a result: the
        # protocol's supervision target is the measured G_fit response, and the median
        # G_fit response has a cross-view cosine of about 0.05, so for most identities
        # the target is noise.  QUERY_ELIGIBILITY_V1 is this programme's existing frozen
        # instrument for "this transition is measurable", it is G_fit-only here, and it
        # is applied to the supervision population only.  Nothing about the evaluation,
        # the metric or the gate changes.
        supervision = np.flatnonzero(np.load(Path(args.eligibility) / "G_fit_eligible.npy"))
    consensus_all = bs.consensus_response(fit_transitions)
    consensus = consensus_all[supervision]
    record: dict = {
        "schema": "VCDESIGN_CANDIDATE_EFFECT_DISTILLATION_V1",
        "read_G_check": False,
        "primary_modalities": list(ced.PRIMARY_MODALITIES),
        "text_excluded": ced.PRIMARY_EXCLUDES_TEXT, "text_exclusion_reason": ced.TEXT_EXCLUSION_REASON,
        "inference_legality": ced.PREDICTED_EFFECT_IS_NOT_AN_INFERENCE_INPUT,
        "response_target": {
            "definition": ced.CONSENSUS, "candidates": int(consensus.shape[0]),
            "genes": int(consensus.shape[1]),
            "supervision_population": ("measurement-eligible G_fit identities"
                                       if args.eligible_targets else "every G_fit identity"),
            "supervision_is_a_declared_variant": bool(args.eligible_targets),
            **bs.view_reliability(fit_transitions[:, supervision]),
            "reliability_over_every_G_fit_identity": bs.view_reliability(fit_transitions)},
        "common_response": bs.common_direction_share(consensus),
    }

    # --- Phase 2: response basis, G_fit only ----------------------------------
    basis = bs.randomized_basis(consensus)
    targets = basis.project(consensus)
    record["basis"] = {"rank": int(ced.BASIS_RANK), "centered": ced.BASIS_CENTERED,
                       "seed": ced.BASIS_SEED, "selected_on": "G_fit only, no grid",
                       "explained_energy_fraction": basis.explained,
                       "singular_value_first": float(basis.singular_values[0]),
                       "singular_value_last": float(basis.singular_values[-1])}

    # --- Phase 3: static to effect, chosen inside G_fit ------------------------
    fit_features_all = pr.static_features(fit_role, ced.PRIMARY_MODALITIES)
    fit_features = pr.StaticFeatures(values=fit_features_all.values[supervision],
                                     present_any=fit_features_all.present_any[supervision],
                                     names=fit_features_all.names)
    select_features = pr.static_features(select_role, ced.PRIMARY_MODALITIES)
    keep = fit_features.fit_rows()
    record["predictor"] = {
        "population": ced.RIDGE_FIT_POPULATION, "fit_candidates": int(keep.size),
        "feature_dimension": int(fit_features.values.shape[1]),
        "criterion": ced.RIDGE_CRITERION, "sensitivity_criterion": ced.RIDGE_SENSITIVITY_CRITERION,
        "cross_validation": pr.cross_validate(fit_features.values[keep], targets[keep],
                                              consensus[keep], basis),
    }
    penalty = record["predictor"]["cross_validation"]["selected_penalty"]
    path = pr.RidgePath(fit_features.values[keep], targets[keep])

    # --- Phase 4: reconstruct the signed effect -------------------------------
    predicted_coefficients = path.predict(select_features.values, penalty)
    predicted_effect = basis.reconstruct(predicted_coefficients)
    all_missing = ~select_features.present_any
    predicted_effect[all_missing] = 0.0
    record["all_missing"] = {
        "G_fit": int((~fit_features_all.present_any).sum()), "G_select": int(all_missing.sum()),
        "G_select_fraction": float(all_missing.mean()),
        "rule": f"score fixed at {ced.ALL_MISSING_SCORE}, no learnable token"}

    # --- Phase 5: direct design probe -----------------------------------------
    select_transitions = np.asarray(select_role.base.transitions, dtype=np.float64)
    utility = [np.asarray(FrozenTransitionAssets.cross_view_utility(select_role.base, view),
                          dtype=np.float64) for view in (0, 1)]
    predicted_unit = _unit(predicted_effect)
    generator = np.random.default_rng(ced.BASIS_SEED)

    scores: dict[str, list[np.ndarray]] = {}
    scores["RANDOM"] = [generator.standard_normal((select_ids.size, select_ids.size)) for _ in (0, 1)]
    magnitude = np.linalg.norm(predicted_effect, axis=1)
    scores["PREDICTED_EFFECT_MAGNITUDE"] = [
        np.broadcast_to(magnitude, (select_ids.size, select_ids.size)).copy() for _ in (0, 1)]
    cosine_rows = []
    measured_rows = []
    for view in (0, 1):
        query = _unit(select_transitions[view])
        block = query @ predicted_unit.T
        block[:, all_missing] = ced.ALL_MISSING_SCORE
        cosine_rows.append(block)
        # the oracle reads the view the grading utility does not use
        measured_rows.append(query @ _unit(select_transitions[view]).T)
    scores["PREDICTED_EFFECT_COSINE"] = cosine_rows
    scores["MEASURED_EFFECT_COSINE"] = measured_rows

    asset = Path(args.asset)
    quarter_all = np.load(asset / "quarter_responses.npy", mmap_mode="r")
    identity_axis = np.asarray([str(v) for v in np.load(asset / "identity_axis.npy", allow_pickle=True)])
    four_way = np.load(asset / "four_way_eligible.npy")
    axis_index = {name: index for index, name in enumerate(identity_axis.tolist())}
    position = np.asarray([axis_index[name] for name in select_ids.tolist()], dtype=np.int64)
    candidate_rows = np.flatnonzero(four_way[position])
    guard_rows = np.flatnonzero(np.load(Path(args.eligibility) / "G_select_eligible.npy")[candidate_rows])
    quarters = [np.asarray(quarter_all[index][position][candidate_rows], dtype=np.float64)
                for index in range(4)]

    record["rows"] = {}
    stored = {}
    for name, matrices in scores.items():
        per_view = [_direction_metrics(matrices[view], utility[view], select_ids, rows) for view in (0, 1)]
        direction = {key: summarize(np.concatenate([entry[key] for entry in per_view]))
                     for key in per_view[0]}
        stored[name] = np.concatenate([entry["MBRU"] for entry in per_view])
        prefix = {str(budget): [] for budget in BUDGETS}
        for view in (0, 1):
            for query in rows.tolist():
                order = stable_order(matrices[view][query], select_ids, excluded_index=query)
                order = order[order != query]
                for budget in BUDGETS:
                    prefix[str(budget)].append(float(all_missing[order[:budget]].mean()))
        record["rows"][name] = {
            "legality": LEGALITY[name], "direction": direction,
            "guardrail": {k: v for k, v in _guardrail(matrices, quarters, candidate_rows, guard_rows,
                                                      select_ids[candidate_rows]).items()
                          if not k.startswith("_")},
            "all_missing_fraction_of_prefix": {key: float(np.mean(value)) for key, value in prefix.items()},
        }

    # --- Phase 5 diagnostics: prediction quality on G_select -------------------
    measured = bs.consensus_response(select_transitions)
    present = ~all_missing
    cosine = np.einsum("ij,ij->i", _unit(predicted_effect[present]), _unit(measured[present]))
    coefficient_truth = basis.project(measured)[present]
    coefficient_correlation = np.asarray([
        float(np.corrcoef(predicted_coefficients[present][:, k], coefficient_truth[:, k])[0, 1])
        for k in range(min(32, ced.BASIS_RANK))])
    # a constant predicted component makes the correlation undefined rather than zero
    finite = coefficient_correlation[np.isfinite(coefficient_correlation)]
    if finite.size == 0:
        finite = np.zeros(1)
    record["prediction_quality_diagnostic_only"] = {
        "G_select_gene_space_cosine": {
            "median": float(np.median(cosine)), "mean": float(cosine.mean()),
            "p05": float(np.quantile(cosine, 0.05)), "p95": float(np.quantile(cosine, 0.95)),
            "fraction_non_positive": float((cosine <= 0).mean())},
        "leading_program_coefficient_correlation": {
            "first": float(coefficient_correlation[0]),
            "median_of_first_32": float(np.median(finite)),
            "max_of_first_32": float(finite.max()),
            "undefined_components": int(coefficient_correlation.size - finite.size)},
        "predicted_effect_cosine_to_the_G_fit_mean_response": float(np.median(
            _unit(predicted_effect[present]) @ _unit(consensus_all.mean(axis=0)))),
    }
    block = scores["PREDICTED_EFFECT_COSINE"][0][rows]
    column = block.mean(axis=0)
    total = float(((block - block.mean()) ** 2).mean())
    record["prediction_quality_diagnostic_only"]["query_independent_share_of_the_score"] = (
        float(((column - block.mean()) ** 2).mean() / total) if total > 0 else float("nan"))

    # --- Phase 6: headroom accounting -----------------------------------------
    external = json.loads(Path(args.main_table).read_text())["rows"]
    baseline_name = args.baseline
    baseline = float(external[baseline_name]["direction"]["MBRU"]["median"])
    oracle = float(external[args.oracle]["direction"]["MBRU"]["median"])
    record["headroom"] = {
        "legal_baseline_row": baseline_name, "oracle_row": args.oracle,
        "definitions": "H = M(row) - M(legal baseline), on the same population and metric",
    }
    for metric, getter in (("MBRU", lambda entry: entry["direction"]["MBRU"]["median"]),
                           ("Mean@20", lambda entry: entry["direction"]["Mean@20"]["median"]),
                           ("J_RU@10", lambda entry: entry["guardrail"]["both_views"]["RU@10"]["median"])):
        base = float(getter(external[baseline_name]))
        top = float(getter(external[args.oracle]))
        got = float(getter(record["rows"]["PREDICTED_EFFECT_COSINE"]))
        same_rule = float(getter(record["rows"]["MEASURED_EFFECT_COSINE"]))
        record["headroom"][metric] = {
            "legal_baseline": base, "oracle": top, "predicted_effect": got,
            "same_rule_measured_effect": same_rule,
            "H_oracle": top - base, "H_recovered": got - base,
            "H_same_rule_ceiling": same_rule - base,
            "recovery_fraction": (got - base) / (top - base) if top != base else float("nan"),
            "recovery_fraction_of_the_same_rule_ceiling":
                (got - base) / (same_rule - base) if same_rule != base else float("nan"),
        }
    reference = stored["RANDOM"]
    for name, entry in record["rows"].items():
        difference = stored[name] - reference
        entry["MBRU_advantage_over_random"] = {
            **summarize(difference), "bootstrap": cluster_bootstrap(difference),
            "fraction_query_improved": float((difference > 0).mean())}
    difference = stored["PREDICTED_EFFECT_COSINE"] - stored["PREDICTED_EFFECT_MAGNITUDE"]
    record["goal_conditioning_of_the_method"] = {
        **summarize(difference), "bootstrap": cluster_bootstrap(difference),
        "fraction_query_improved": float((difference > 0).mean())}

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "response_basis.npy", basis.basis.astype(np.float32))
    np.save(output / "predicted_effect_G_select.npy", predicted_effect.astype(np.float32))
    record["runtime"] = {"timestamp": datetime.now(timezone.utc).isoformat(),
                         "python": platform.python_version(), "numpy": np.__version__}
    record["provenance"] = config.provenance()
    (output / "candidate_effect_distillation.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Candidate effect distillation V1")
    for name in ("config", "eligibility", "asset", "main_table", "output"):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, required=True)
    parser.add_argument("--baseline", default="RB_Q_CURRENT__C_LEGAL")
    parser.add_argument("--oracle", default="RB_Q_RAW__C_ORACLE")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eligible-targets", dest="eligible_targets", action="store_true",
                        help="restrict the supervision population to measurement-eligible G_fit identities")
    record = run(parser.parse_args())
    cv = record["predictor"]["cross_validation"]
    print(f"basis rank {record['basis']['rank']} explains {record['basis']['explained_energy_fraction']:.4f} "
          f"of G_fit response energy; mean direction carries "
          f"{record['common_response']['share_of_response_energy_on_the_mean_direction']:.4f}")
    print(f"supervision population: {record['response_target']['supervision_population']}, "
          f"{record['response_target']['candidates']} identities; cross-view response reliability "
          f"median cosine {record['response_target']['cross_view_cosine']['median']:.4f} "
          f"(all G_fit identities: "
          f"{record['response_target']['reliability_over_every_G_fit_identity']['cross_view_cosine']['median']:.4f})")
    print(f"ridge penalty {cv['selected_penalty']:g} chosen by {record['predictor']['criterion']}; "
          f"held-out gene-space cosine {cv['held_out_cosine_at_selected']:.4f}")
    diagnostic = record["prediction_quality_diagnostic_only"]
    print(f"G_select predicted-vs-measured cosine median "
          f"{diagnostic['G_select_gene_space_cosine']['median']:.4f}; "
          f"query-independent share of the score "
          f"{diagnostic['query_independent_share_of_the_score']:.4f}")
    print(f"\n{'row':32s} {'MBRU':>8s} {'Mean@10':>9s} {'Mean@20':>9s} {'HVC@10':>7s} {'HVC@20':>7s} "
          f"{'J RU@10':>9s} {'J>0@10':>7s} {'miss@10':>8s}")
    for name, entry in record["rows"].items():
        d, g = entry["direction"], entry["guardrail"]
        print(f"{name:32s} {d['MBRU']['median']:+8.4f} {d['Mean@10']['median']:+9.4f} "
              f"{d['Mean@20']['median']:+9.4f} {d['HighValueCount@10']['median']:7.2f} "
              f"{d['HighValueCount@20']['median']:7.2f} {g['both_views']['RU@10']['median']:+9.5f} "
              f"{g['view0']['fraction_J_positive@10']:7.4f} "
              f"{entry['all_missing_fraction_of_prefix']['10']:8.4f}")
    print("\nheadroom recovery")
    for metric in ("MBRU", "Mean@20", "J_RU@10"):
        item = record["headroom"][metric]
        print(f"  {metric:8s} baseline={item['legal_baseline']:+.4f} predicted={item['predicted_effect']:+.4f} "
              f"same-rule-oracle={item['same_rule_measured_effect']:+.4f} oracle={item['oracle']:+.4f}  "
              f"H_rec={item['H_recovered']:+.4f} / H_or={item['H_oracle']:+.4f} = "
              f"{item['recovery_fraction']:+.3f}")
    goal = record["goal_conditioning_of_the_method"]
    low, high = goal["bootstrap"]["median"]["ci95"]
    print(f"\ncosine minus magnitude (is the method goal conditioned): median {goal['median']:+.4f} "
          f"CI95=[{low:+.4f},{high:+.4f}] improved={goal['fraction_query_improved']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
