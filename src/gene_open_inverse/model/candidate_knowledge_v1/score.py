#!/usr/bin/env python3
"""Blind G_check scoring; the relevance asset is neither accepted nor opened."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .common import (ARM_NAMES, array_sha256, dump_json_exclusive, load_json, resolve_config,
                     response_operators, save_npz_exclusive, sha256_file, target_donor_maps)


def feature_matrix(features, arm):
    if arm in ("STRING", "SEQUENCE", "FUNCTION", "TEXT"):
        return features[arm].astype(np.float32)
    return np.concatenate([features[name].astype(np.float32)
                           for name in ("STRING", "SEQUENCE", "FUNCTION", "TEXT")], axis=1)


def main():
    if not __debug__:
        raise RuntimeError("assertions are part of the protocol")
    p = argparse.ArgumentParser()
    for name in ("config", "design", "freeze", "lock", "features", "models", "selection",
                 "implementation-correction", "out", "manifest"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args()
    cfg, freeze, lock = resolve_config(args.config), load_json(args.freeze), load_json(args.lock)
    selection = load_json(args.selection)
    assert lock["status"] == "LOCKED_BEFORE_G_CHECK_OPEN"
    for key in ("freeze", "features", "models", "selection"):
        assert lock["sha256"][key] == sha256_file(getattr(args, key))
    here = Path(__file__).resolve().parent
    assert lock["code_sha256"]["common.py"] == sha256_file(here / "common.py")
    assert lock["code_sha256"]["report.py"] == sha256_file(here / "report.py")
    correction = load_json(args.implementation_correction)
    assert correction["status"] == "FROZEN_NUMERICAL_ISOLATION_AUDIT_CORRECTION_BEFORE_VALID_SCORE_OUTPUT"
    assert correction["freeze_sha256"] == sha256_file(args.freeze)
    assert correction["lock_sha256"] == sha256_file(args.lock)
    assert correction["old_score_sha256"] == lock["code_sha256"]["score.py"]
    assert correction["new_score_sha256"] == sha256_file(here / "score.py")
    isolation_tolerance = float(correction["max_abs_tolerance"])

    split = load_json(cfg["paths"]["split"])
    features = np.load(args.features, allow_pickle=False)
    models = np.load(args.models, allow_pickle=False)
    fit = np.load(cfg["paths"]["gcp_fit_asset"], allow_pickle=False)
    query_asset = np.load(cfg["paths"]["gcp_check_query"], allow_pickle=False)
    assert "R" not in query_asset.files
    genes = query_asset["genes"].astype(str)
    assert list(genes) == split["G_check"]
    universe = features["universe"].astype(str)
    uindex = {g: i for i, g in enumerate(universe)}
    check_pos = np.asarray([uindex[g] for g in genes], np.int64)
    query = query_asset["U"].astype(np.float32)
    measured, a0, a1, reconstructed_u = response_operators(cfg["paths"]["hek_shadow"], genes, fit)
    assert float(np.max(np.abs(query - reconstructed_u))) < 2e-5
    donors = target_donor_maps(len(genes), len(models["G_fit"]), cfg["diagnostics"]["target_derangement_salts"])
    donor_query = models["U_fit"].astype(np.float32)[donors]
    shuffled = models["ALL_shuffled_donor"].astype(np.int64)

    arrays = {"genes": genes, "query": query, "measured": measured.astype(np.float32),
              "measured_R0": a0.astype(np.float32), "measured_R1": a1.astype(np.float32)}
    score_hash = {}
    all_matrix = None
    for arm in ARM_NAMES:
        base = "ALL" if arm == "ALL_shuffled" else arm
        if base == "ALL":
            if all_matrix is None:
                all_matrix = feature_matrix(features, "ALL")
            x = all_matrix
        else:
            x = feature_matrix(features, base)
        cx = x[shuffled[check_pos]] if arm == "ALL_shuffled" else x[check_pos]
        w, b = models[arm + "__W"].astype(np.float64), models[arm + "__b"].astype(np.float64)
        pred = (cx.astype(np.float64) @ w + b).astype(np.float32)
        score = (query @ pred.T).astype(np.float32)
        perm = np.stack([(q @ pred.T).astype(np.float32) for q in donor_query])
        arrays[arm + "__pred"] = pred
        arrays[arm + "__score"] = score
        arrays[arm + "__perm"] = perm
        isolated = query[0] @ pred.T
        isolation_max = float(np.max(np.abs(score[0].astype(np.float64) - isolated.astype(np.float64))))
        score_hash[arm] = {"pred": array_sha256(pred), "score": array_sha256(score),
                           "perm": array_sha256(perm), "one_query_isolation_max_abs": isolation_max,
                           "one_query_isolation_pass": isolation_max <= isolation_tolerance}
        assert score_hash[arm]["one_query_isolation_pass"]
    oracle_score = (query @ measured.T).astype(np.float32)
    oracle_perm = np.stack([(q @ measured.T).astype(np.float32) for q in donor_query])
    arrays["ORACLE8__pred"] = measured.astype(np.float32)
    arrays["ORACLE8__score"] = oracle_score
    arrays["ORACLE8__perm"] = oracle_perm
    arrays["meta"] = np.asarray(json.dumps({"score_hash": score_hash}, sort_keys=True))
    digest = save_npz_exclusive(args.out, **arrays)
    manifest = {"schema": "CANDIDATE_KNOWLEDGE_V1_BLIND_SCORES", "status": "COMPLETE_LABEL_UNOPENED",
                "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "governance": {"G_check_query_opened": True, "G_check_response_opened_for_operator_diagnostic": True,
                               "G_check_relevance_opened": False, "official_DEV_TEST_response_or_relevance_opened": False,
                               "cross_query_state_update": False},
                "asset": {"path": str(Path(args.out).resolve()), "sha256": digest},
                "sha256": {"config": sha256_file(args.config), "design": sha256_file(args.design),
                           "freeze": sha256_file(args.freeze), "lock": sha256_file(args.lock),
                           "features": sha256_file(args.features), "models": sha256_file(args.models),
                           "selection": sha256_file(args.selection),
                           "check_query": sha256_file(cfg["paths"]["gcp_check_query"]),
                           "score_code": sha256_file(__file__),
                           "implementation_correction": sha256_file(args.implementation_correction)},
                "arrays": score_hash}
    print(dump_json_exclusive(args.manifest, manifest))


if __name__ == "__main__":
    main()
