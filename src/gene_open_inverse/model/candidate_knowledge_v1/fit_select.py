#!/usr/bin/env python3
"""Fit all ridge paths on G_fit and select alpha once on G_select."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .common import (ARM_NAMES, array_sha256, dump_json_exclusive, evaluator_rows,
                     exact_softndcg10_rows, load_json, load_module, metric_rows, resolve_config,
                     response_operators, ridge_path, save_npz_exclusive, sha256_file,
                     shuffled_donor_index, target_donor_maps)


def log(message, started=[time.time()]):
    print("[%7.1fs] %s" % (time.time() - started[0], message), flush=True)


def arm_matrix(features, arm):
    if arm in ("STRING", "SEQUENCE", "FUNCTION", "TEXT"):
        return features[arm].astype(np.float32)
    return np.concatenate([features[name].astype(np.float32)
                           for name in ("STRING", "SEQUENCE", "FUNCTION", "TEXT")], axis=1)


def main():
    if not __debug__:
        raise RuntimeError("assertions are part of the protocol")
    p = argparse.ArgumentParser()
    for name in ("config", "design", "freeze", "features", "feature-manifest", "models-out", "selection-out"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args()
    cfg, freeze = resolve_config(args.config), load_json(args.freeze)
    assert freeze["status"] == "FROZEN_BEFORE_RESPONSE_OR_G_SELECT_READ"
    here = Path(__file__).resolve().parent
    for name in ("common.py", "fit_select.py", "score.py", "report.py", "lock.py"):
        assert freeze["code_sha256"][name] == sha256_file(here / name), "code drift " + name
    assert freeze["sha256"]["features"] == sha256_file(args.features)
    assert freeze["sha256"]["feature_manifest"] == sha256_file(args.feature_manifest)

    split = load_json(cfg["paths"]["split"])
    features = np.load(args.features, allow_pickle=False)
    fit = np.load(cfg["paths"]["gcp_fit_asset"], allow_pickle=False)
    select = np.load(cfg["paths"]["gcp_select_asset"], allow_pickle=False)
    universe = features["universe"].astype(str)
    assert np.array_equal(universe, fit["universe"].astype(str))
    assert list(fit["G_fit"].astype(str)) == split["G_fit"]
    assert list(select["genes"].astype(str)) == split["G_select"]
    uindex = {g: i for i, g in enumerate(universe)}
    fit_pos = np.asarray([uindex[g] for g in split["G_fit"]], np.int64)
    select_pos = np.asarray([uindex[g] for g in split["G_select"]], np.int64)
    assert np.array_equal(features["fit_pos"], fit_pos)

    measured_fit, a0_fit, a1_fit, u_fit = response_operators(
        cfg["paths"]["hek_shadow"], split["G_fit"], fit)
    log("G_fit operator target reconstructed")
    donors = target_donor_maps(len(select_pos), len(fit_pos), cfg["diagnostics"]["target_derangement_salts"])
    donor_query = u_fit[donors]
    relevance = select["R"].astype(np.float32)
    query = select["U"].astype(np.float32)
    evaluator = load_module(cfg["paths"]["evaluator"], "candidate_knowledge_select_evaluator")
    alpha_grid = list(map(float, cfg["ridge"]["alphas"]))
    shuffled = shuffled_donor_index(universe, split["G_fit"], cfg["shuffle"]["seed"])

    model_arrays = {"universe": universe, "G_fit": np.asarray(split["G_fit"]),
                    "U_fit": u_fit.astype(np.float32), "measured_fit": measured_fit.astype(np.float32),
                    "ALL_shuffled_donor": shuffled}
    selection = {"schema": "CANDIDATE_KNOWLEDGE_V1_SELECTION", "status": "SELECTED_ON_G_SELECT_ONLY",
                 "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "primary": cfg["ridge"]["selection"],
                 "arms": {}, "governance": {"G_check_response_or_relevance_opened": False,
                                               "official_DEV_TEST_response_or_relevance_opened": False}}
    all_matrix = None
    for arm in ARM_NAMES:
        base_arm = "ALL" if arm == "ALL_shuffled" else arm
        if base_arm == "ALL":
            if all_matrix is None:
                all_matrix = arm_matrix(features, "ALL")
            x = all_matrix
        else:
            x = arm_matrix(features, base_arm)
        train_x = x[shuffled[fit_pos]] if arm == "ALL_shuffled" else x[fit_pos]
        candidate_x = x[shuffled[select_pos]] if arm == "ALL_shuffled" else x[select_pos]
        paths, audit = ridge_path(train_x, measured_fit, alpha_grid)
        grid = []
        for alpha in alpha_grid:
            w, b = paths[alpha]
            pred = (candidate_x.astype(np.float64) @ w + b).astype(np.float32)
            score = query @ pred.T
            official = evaluator_rows(evaluator, select["genes"], relevance, score,
                                      "candidate_knowledge_G_select")
            independent = exact_softndcg10_rows(relevance, score)
            difference = np.abs(official - independent)
            difference = difference[np.isfinite(difference)]
            assert float(difference.max(initial=0.0)) <= cfg["diagnostics"]["metric_tolerance"]
            record, _, _ = metric_rows(relevance, query, pred, donor_query)
            assert abs(record["M"] - float(np.nanmean(official))) <= cfg["diagnostics"]["metric_tolerance"]
            grid.append({"alpha": alpha, **record})
        chosen = max(grid, key=lambda item: (item["M"], -item["alpha"]))
        w, b = paths[chosen["alpha"]]
        model_arrays[arm + "__W"] = w.astype(np.float64)
        model_arrays[arm + "__b"] = b.astype(np.float64)
        model_arrays[arm + "__alpha"] = np.asarray(chosen["alpha"], np.float64)
        selection["arms"][arm] = {"grid": grid, "selected": chosen, "ridge_audit": audit,
                                   "feature_dim": int(x.shape[1])}
        log("%s selected alpha %g M %.6f" % (arm, chosen["alpha"], chosen["M"]))

    # The inherited alpha-100 STRING computation is an end-to-end continuity witness, not a selector.
    string_x = arm_matrix(features, "STRING")
    ref_path, _ = ridge_path(string_x[fit_pos], measured_fit, [100.0])
    rw, rb = ref_path[100.0]
    rebuilt = (string_x.astype(np.float64) @ rw + rb).astype(np.float32)
    historical = fit["A_all"].astype(np.float32)
    scores_new = select["U"].astype(np.float32) @ rebuilt[select_pos].T
    scores_old = select["U"].astype(np.float32) @ historical[select_pos].T
    maximum = float(np.max(np.abs(scores_new - scores_old)))
    assert maximum <= float(cfg["diagnostics"]["string_alpha100_score_tolerance"]), maximum
    selection["continuity"] = {"STRING_alpha": 100.0, "max_abs_G_select_score_difference": maximum,
                               "new_M": float(np.nanmean(exact_softndcg10_rows(relevance, scores_new))),
                               "historical_M": float(np.nanmean(exact_softndcg10_rows(relevance, scores_old))),
                               "tolerance": cfg["diagnostics"]["string_alpha100_score_tolerance"]}
    selection["sha256"] = {"config": sha256_file(args.config), "design": sha256_file(args.design),
                           "freeze": sha256_file(args.freeze), "features": sha256_file(args.features),
                           "feature_manifest": sha256_file(args.feature_manifest),
                           "gcp_fit_asset": sha256_file(cfg["paths"]["gcp_fit_asset"]),
                           "gcp_select_asset": sha256_file(cfg["paths"]["gcp_select_asset"]),
                           "operator_target": array_sha256(measured_fit)}
    model_arrays["meta"] = np.asarray(json.dumps(selection, sort_keys=True))
    model_hash = save_npz_exclusive(args.models_out, **model_arrays)
    selection["models"] = {"path": str(Path(args.models_out).resolve()), "sha256": model_hash}
    digest = dump_json_exclusive(args.selection_out, selection)
    print(json.dumps({"selection": str(Path(args.selection_out).resolve()), "sha256": digest,
                      "selected": {a: selection["arms"][a]["selected"] for a in ARM_NAMES}}, indent=2))


if __name__ == "__main__":
    main()
