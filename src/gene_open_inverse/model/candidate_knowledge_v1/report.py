#!/usr/bin/env python3
"""Open G_check relevance after the lock and issue the one final report."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from .common import (ARM_NAMES, bootstrap, bootstrap_ratio, cosine_rows, dump_json_exclusive,
                     evaluator_rows, exact_softndcg10_rows, load_json, load_module, pearson_columns,
                     resolve_config, save_npz_exclusive, sha256_file, stable_int)


def finite_summary(value):
    x = np.asarray(value, np.float64)
    x = x[np.isfinite(x)]
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)),
            "q10": float(np.quantile(x, 0.1)), "q90": float(np.quantile(x, 0.9))}


def main():
    if not __debug__:
        raise RuntimeError("assertions are part of the protocol")
    p = argparse.ArgumentParser()
    for name in ("config", "design", "freeze", "lock", "features", "feature-manifest", "models",
                 "selection", "scores", "score-manifest", "out", "per-query-out", "csv-out"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args()
    cfg, freeze, lock = resolve_config(args.config), load_json(args.freeze), load_json(args.lock)
    score_manifest = load_json(args.score_manifest)
    assert lock["status"] == "LOCKED_BEFORE_G_CHECK_OPEN"
    assert score_manifest["status"] == "COMPLETE_LABEL_UNOPENED"
    assert score_manifest["governance"]["G_check_relevance_opened"] is False
    assert score_manifest["asset"]["sha256"] == sha256_file(args.scores)
    here = Path(__file__).resolve().parent
    for name in ("common.py", "report.py"):
        assert lock["code_sha256"][name] == sha256_file(here / name), "post-lock code drift " + name

    selection = load_json(args.selection)
    feature_manifest = load_json(args.feature_manifest)
    score = np.load(args.scores, allow_pickle=False)
    label = np.load(cfg["paths"]["gcp_check_label"], allow_pickle=False)
    genes = label["genes"].astype(str)
    relevance = label["R"].astype(np.float32)
    assert np.array_equal(genes, score["genes"].astype(str)) and np.all(np.diag(relevance) == 0)
    evaluator = load_module(cfg["paths"]["evaluator"], "candidate_knowledge_check_evaluator")
    nboot, bseed = int(cfg["diagnostics"]["bootstrap_n"]), int(cfg["diagnostics"]["bootstrap_seed"])

    names = list(ARM_NAMES) + ["ORACLE8"]
    rows, trows, summaries = {}, {}, {}
    audit_indices = np.asarray(sorted(range(len(genes)),
                                      key=lambda i: (stable_int("ck-metric-audit", genes[i]), i))
                               [:int(cfg["diagnostics"]["independent_metric_rows"])], np.int64)
    metric_audit = []
    for arm in names:
        true_scores = score[arm + "__score"].astype(np.float32)
        official = evaluator_rows(evaluator, genes, relevance, true_scores, "candidate_knowledge_G_check")
        independent = exact_softndcg10_rows(relevance, true_scores)
        difference = np.abs(official[audit_indices] - independent[audit_indices])
        difference = difference[np.isfinite(difference)]
        maximum = float(difference.max(initial=0.0))
        assert maximum <= cfg["diagnostics"]["metric_tolerance"]
        perm = score[arm + "__perm"]
        perm_rows = np.stack([exact_softndcg10_rows(relevance, perm[i]) for i in range(len(perm))])
        rows[arm] = official
        trows[arm] = official - np.nanmean(perm_rows, axis=0)
        summaries[arm] = {"M": float(np.nanmean(official)), "T": float(np.nanmean(trows[arm])),
                          "permutation_mean_M": float(np.nanmean(perm_rows)),
                          "permutation_M": np.nanmean(perm_rows, axis=1).tolist(),
                          "n_estimable": int(np.isfinite(official).sum()),
                          "M_bootstrap": bootstrap(official, bseed + stable_int(arm, "M") % 100000, nboot),
                          "T_bootstrap": bootstrap(trows[arm], bseed + stable_int(arm, "T") % 100000, nboot)}
        metric_audit.append({"arm": arm, "n_rows": int(len(audit_indices)), "max_abs_difference": maximum})

    contrasts = {}
    for arm in ARM_NAMES:
        contrasts[arm + "_minus_STRING"] = bootstrap(
            rows[arm] - rows["STRING"], bseed + stable_int(arm, "minus-string") % 100000, nboot)
    contrasts["ALL_minus_ALL_shuffled"] = bootstrap(
        rows["ALL"] - rows["ALL_shuffled"], bseed + 701, nboot)
    contrasts["T_ALL_minus_T_STRING"] = bootstrap(
        trows["ALL"] - trows["STRING"], bseed + 702, nboot)
    contrasts["T_ALL_minus_T_ALL_shuffled"] = bootstrap(
        trows["ALL"] - trows["ALL_shuffled"], bseed + 703, nboot)

    denominator = rows["ORACLE8"] - rows["STRING"]
    gap_closed = {}
    for arm in ARM_NAMES:
        gap_closed[arm] = bootstrap_ratio(rows[arm] - rows["STRING"], denominator,
                                          bseed + stable_int(arm, "gap-closed") % 100000, nboot)
    oracle_gap = bootstrap(denominator, bseed + 704, nboot)

    measured = score["measured"].astype(np.float32)
    measured0 = score["measured_R0"].astype(np.float32)
    measured1 = score["measured_R1"].astype(np.float32)
    cross_reliability = cosine_rows(measured0, measured1)
    operator = {"measured_R0_vs_R1_cosine": {
                    "summary": finite_summary(cross_reliability),
                    "bootstrap": bootstrap(cross_reliability, bseed + 800, nboot)},
                "note": "Measured R0/R1 operators are noisy assay views, not error-free ground truth.",
                "arms": {}}
    operator_cross_rows = {}
    estimable = label["relc"].astype(np.float32) > 0
    for arm in ARM_NAMES:
        pred = score[arm + "__pred"].astype(np.float32)
        average_cos = cosine_rows(pred, measured)
        half_cos = 0.5 * (cosine_rows(pred, measured0) + cosine_rows(pred, measured1))
        operator_cross_rows[arm] = half_cos
        corr, r2 = pearson_columns(pred, measured)
        operator["arms"][arm] = {
            "candidate_cosine_to_average_operator": finite_summary(average_cos),
            "cross_half_aware_candidate_cosine": {"summary": finite_summary(half_cos),
                "bootstrap": bootstrap(half_cos, bseed + stable_int(arm, "operator") % 100000, nboot)},
            "estimable_candidate_cross_half_cosine": finite_summary(half_cos[estimable]),
            "per_axis_pearson": corr, "per_axis_R2": r2,
            "mean_axis_R2": float(np.mean([x for x in r2 if x is not None]))}
    operator["contrasts"] = {
        "ALL_minus_STRING_cross_half_cosine": bootstrap(
            operator_cross_rows["ALL"] - operator_cross_rows["STRING"], bseed + 901, nboot),
        "ALL_minus_ALL_shuffled_cross_half_cosine": bootstrap(
            operator_cross_rows["ALL"] - operator_cross_rows["ALL_shuffled"], bseed + 902, nboot)}
    operator["ALL_minus_STRING_axis_R2"] = [
        a - b for a, b in zip(operator["arms"]["ALL"]["per_axis_R2"],
                              operator["arms"]["STRING"]["per_axis_R2"])]

    rank_vs_string = contrasts["ALL_minus_STRING"]["ci95"][0] > 0
    rank_vs_shuffle = contrasts["ALL_minus_ALL_shuffled"]["ci95"][0] > 0
    op_vs_string = operator["contrasts"]["ALL_minus_STRING_cross_half_cosine"]["ci95"][0] > 0
    op_vs_shuffle = operator["contrasts"]["ALL_minus_ALL_shuffled_cross_half_cosine"]["ci95"][0] > 0
    knowledge_supported = bool(rank_vs_string and rank_vs_shuffle)
    practical_value = bool(knowledge_supported and op_vs_string)
    nonlinear_eligible = bool(practical_value and op_vs_shuffle and gap_closed["ALL"]["ci95"][0] > 0)
    if knowledge_supported:
        direction = "RICHER_STATIC_CANDIDATE_KNOWLEDGE_SUPPORTED"
    else:
        direction = "CLOSE_STATIC_CANDIDATE_KNOWLEDGE_ENRICHMENT"

    arrays = {"genes": genes, "estimable": estimable, "measured_R0_R1_cosine": cross_reliability,
              "oracle_gap_row": denominator}
    for arm in names:
        arrays[arm + "__M"] = rows[arm]
        arrays[arm + "__T"] = trows[arm]
    for arm in ARM_NAMES:
        arrays[arm + "__operator_crosshalf_cosine"] = operator_cross_rows[arm]
    per_hash = save_npz_exclusive(args.per_query_out, **arrays)
    with open(args.csv_out, "x", newline="", encoding="utf-8") as handle:
        fields = ["gene", "estimable", "measured_R0_R1_cosine"]
        fields += [arm + "_M" for arm in names] + [arm + "_T" for arm in names]
        fields += [arm + "_operator_crosshalf_cosine" for arm in ARM_NAMES]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, gene in enumerate(genes):
            item = {"gene": gene, "estimable": bool(estimable[i]),
                    "measured_R0_R1_cosine": cross_reliability[i]}
            for arm in names:
                item[arm + "_M"] = rows[arm][i]
                item[arm + "_T"] = trows[arm][i]
            for arm in ARM_NAMES:
                item[arm + "_operator_crosshalf_cosine"] = operator_cross_rows[arm][i]
            writer.writerow(item)

    result = {"schema": "CANDIDATE_KNOWLEDGE_V1_REPORT", "status": "COMPLETE_AND_CLOSED",
              "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "evidence_class": "previously exposed official-TRAIN internal G_check development evidence",
              "model_summary": summaries, "ranking_contrasts": contrasts,
              "operator_prediction": operator,
              "oracle_gap": {"definition": "M(ORACLE8)-M(STRING) on identical half-pooled G_check rows",
                             "value": oracle_gap, "gap_closed": gap_closed,
                             "warning": "ORACLE8 is deployment-illegal and partly label-coupled; do not combine this denominator with the historical strict cross-view +0.092382 gap."},
              "features": {"provenance": feature_manifest["sha256"]["inputs"],
                           "details": feature_manifest["feature_details"],
                           "coverage": feature_manifest["coverage"]},
              "selection": {arm: selection["arms"][arm] for arm in ARM_NAMES},
              "interpretation": {"ALL_beats_STRING": rank_vs_string,
                                 "ALL_beats_ALL_shuffled": rank_vs_shuffle,
                                 "ALL_operator_beats_STRING": op_vs_string,
                                 "ALL_operator_beats_ALL_shuffled": op_vs_shuffle,
                                 "gene_specific_knowledge_supported": knowledge_supported,
                                 "operator_and_ranking_practical_value_supported": practical_value,
                                 "finite_nonlinear_operator_experiment_eligible_but_not_started": nonlinear_eligible,
                                 "direction": direction,
                                 "scope": "seen HEK293T context plus globally unseen candidates only"},
              "classification": {"verified_fact": ["all transforms fitted on G_fit static features only",
                                                       "all arms use one deterministic multi-output ridge and the same XSVD8 scorer"],
                                 "diagnostic_oracle": ["ORACLE8", "Gap_closed"],
                                 "inference": [direction],
                                 "unresolved_hypothesis": ["official DEV/TEST generalisation", "unseen-context generalisation", "prospective biological utility"]},
              "metric_audit": {"method": "frozen evaluator versus independent lexsort implementation",
                               "records": metric_audit, "PASS": True},
              "uncertainty": {"method": "paired nonparametric bootstrap over origin genes",
                              "bootstrap_n": nboot, "candidate_pairs_treated_as_independent": False},
              "governance": {"G_check_opened_after_lock_and_blind_scores": True,
                             "official_DEV_TEST_response_or_relevance_opened": False,
                             "benchmark_modified": False, "additional_rescue_started": False},
              "outputs": {"per_query_npz": {"path": str(Path(args.per_query_out).resolve()), "sha256": per_hash},
                          "per_query_csv": {"path": str(Path(args.csv_out).resolve()), "sha256": sha256_file(args.csv_out)}},
              "sha256": {"config": sha256_file(args.config), "design": sha256_file(args.design),
                         "freeze": sha256_file(args.freeze), "lock": sha256_file(args.lock),
                         "features": sha256_file(args.features), "feature_manifest": sha256_file(args.feature_manifest),
                         "models": sha256_file(args.models), "selection": sha256_file(args.selection),
                         "scores": sha256_file(args.scores), "score_manifest": sha256_file(args.score_manifest),
                         "check_label": sha256_file(cfg["paths"]["gcp_check_label"]),
                         "report_code": sha256_file(__file__)}}
    digest = dump_json_exclusive(args.out, result)
    print(json.dumps({"report": str(Path(args.out).resolve()), "sha256": digest,
                      "interpretation": result["interpretation"],
                      "M": {arm: summaries[arm]["M"] for arm in names}}, indent=2))


if __name__ == "__main__":
    main()
