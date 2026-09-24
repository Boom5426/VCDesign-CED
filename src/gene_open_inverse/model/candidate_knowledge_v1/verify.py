#!/usr/bin/env python3
"""Post-hoc independent verifier for the immutable CANDIDATE_KNOWLEDGE_V1 outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ARMS = ("STRING", "SEQUENCE", "FUNCTION", "TEXT", "ALL", "ALL_shuffled", "ORACLE8")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def metric(relevance, scores):
    n = len(relevance)
    discount = 1.0 / np.log2(np.arange(2, 12, dtype=np.float64))
    out = np.full(n, np.nan, np.float64)
    for row in range(n):
        pool = np.arange(n) != row
        r = relevance[row, pool].astype(np.float64)
        s = scores[row, pool].astype(np.float64)
        order = np.lexsort((r, -s))[:10]
        ideal = np.sort(r)[::-1][:10]
        denominator = float(ideal @ discount)
        if denominator > 0:
            out[row] = float(r[order] @ discount / denominator)
    return out


def cosine(a, b):
    x, y = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.sum(x * y, 1) / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-12)


def main():
    p = argparse.ArgumentParser()
    for name in ("report", "scores", "score-manifest", "label", "features", "feature-manifest",
                 "models", "selection", "freeze", "lock", "per-query", "csv", "out"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args()
    report = json.load(open(args.report, encoding="utf-8"))
    score_manifest = json.load(open(args.score_manifest, encoding="utf-8"))
    assert report["status"] == "COMPLETE_AND_CLOSED"
    assert score_manifest["status"] == "COMPLETE_LABEL_UNOPENED"
    assert score_manifest["asset"]["sha256"] == digest(args.scores)
    bound = {"scores": args.scores, "score_manifest": args.score_manifest, "features": args.features,
             "feature_manifest": args.feature_manifest, "models": args.models, "selection": args.selection,
             "freeze": args.freeze, "lock": args.lock, "check_label": args.label}
    for key, path in bound.items():
        assert report["sha256"][key] == digest(path), key
    assert report["outputs"]["per_query_npz"]["sha256"] == digest(args.per_query)
    assert report["outputs"]["per_query_csv"]["sha256"] == digest(args.csv)

    scores = np.load(args.scores, allow_pickle=False)
    label = np.load(args.label, allow_pickle=False)
    relevance = label["R"].astype(np.float32)
    assert np.array_equal(scores["genes"].astype(str), label["genes"].astype(str))
    rows, trows, checks = {}, {}, {}
    for arm in ARMS:
        rows[arm] = metric(relevance, scores[arm + "__score"])
        perm = np.stack([metric(relevance, scores[arm + "__perm"][i])
                         for i in range(len(scores[arm + "__perm"]))])
        trows[arm] = rows[arm] - np.nanmean(perm, axis=0)
        dm = abs(float(np.nanmean(rows[arm])) - report["model_summary"][arm]["M"])
        dt = abs(float(np.nanmean(trows[arm])) - report["model_summary"][arm]["T"])
        assert dm < 1e-12 and dt < 1e-12
        checks[arm] = {"M_abs_difference": dm, "T_abs_difference": dt}
    gap = rows["ORACLE8"] - rows["STRING"]
    assert abs(float(np.nanmean(gap)) - report["oracle_gap"]["value"]["mean"]) < 1e-12
    for arm in ARMS[:-1]:
        ratio = float(np.nanmean(rows[arm] - rows["STRING"]) / np.nanmean(gap))
        assert abs(ratio - report["oracle_gap"]["gap_closed"][arm]["ratio_of_means"]) < 1e-12

    measured0, measured1 = scores["measured_R0"], scores["measured_R1"]
    for arm in ARMS[:-1]:
        expected = 0.5 * (cosine(scores[arm + "__pred"], measured0) +
                          cosine(scores[arm + "__pred"], measured1))
        got = report["operator_prediction"]["arms"][arm]["cross_half_aware_candidate_cosine"]["summary"]["mean"]
        assert abs(float(expected.mean()) - got) < 1e-12

    result = {"schema": "CANDIDATE_KNOWLEDGE_V1_VERIFICATION", "status": "PASS",
              "independent_implementation": True, "checks": checks,
              "oracle_gap_abs_difference": abs(float(np.nanmean(gap)) - report["oracle_gap"]["value"]["mean"]),
              "sha256": {key: digest(path) for key, path in {**bound, "report": args.report,
                                                               "per_query": args.per_query, "csv": args.csv}.items()}}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"verification": str(out.resolve()), "sha256": digest(out), "status": "PASS"}, indent=2))


if __name__ == "__main__":
    main()
