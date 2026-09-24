#!/usr/bin/env python3
"""Bind selected models before G_check query/response access."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .common import ARM_NAMES, dump_json_exclusive, load_json, resolve_config, sha256_file


def main():
    p = argparse.ArgumentParser()
    for name in ("config", "design", "freeze", "features", "feature-manifest", "models", "selection", "out"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args()
    cfg, freeze, selection = resolve_config(args.config), load_json(args.freeze), load_json(args.selection)
    assert freeze["status"] == "FROZEN_BEFORE_RESPONSE_OR_G_SELECT_READ"
    assert selection["status"] == "SELECTED_ON_G_SELECT_ONLY"
    assert selection["governance"]["G_check_response_or_relevance_opened"] is False
    assert set(selection["arms"]) == set(ARM_NAMES)
    assert selection["models"]["sha256"] == sha256_file(args.models)
    here = Path(__file__).resolve().parent
    for name, digest in freeze["code_sha256"].items():
        assert digest == sha256_file(here / name), "post-freeze code drift " + name
    payload = {"schema": "CANDIDATE_KNOWLEDGE_V1_CHECK_LOCK",
               "status": "LOCKED_BEFORE_G_CHECK_OPEN", "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "selected": {arm: selection["arms"][arm]["selected"]["alpha"] for arm in ARM_NAMES},
               "governance": {"G_check_response_or_relevance_opened_before_lock": False,
                              "official_DEV_TEST_response_or_relevance_opened": False,
                              "additional_model_or_permutation_after_lock_allowed": False},
               "sha256": {"config": sha256_file(args.config), "design": sha256_file(args.design),
                          "freeze": sha256_file(args.freeze), "features": sha256_file(args.features),
                          "feature_manifest": sha256_file(args.feature_manifest),
                          "models": sha256_file(args.models), "selection": sha256_file(args.selection)},
               "code_sha256": {name: sha256_file(here / name) for name in ("common.py", "score.py", "report.py")}}
    print(dump_json_exclusive(args.out, payload))


if __name__ == "__main__":
    main()
