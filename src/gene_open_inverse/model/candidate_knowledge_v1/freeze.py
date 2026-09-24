#!/usr/bin/env python3
"""Bind the prospective design, static assets and code before any response-dependent fit."""
from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
import torch

from .common import dump_json_exclusive, load_json, resolve_config, sha256_file, sha256_tree


def main():
    p = argparse.ArgumentParser()
    for name in ("config", "design", "features", "feature-manifest", "out"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args()
    cfg = resolve_config(args.config)
    manifest = load_json(args.feature_manifest)
    assert manifest["status"] == "STATIC_FEATURES_COMPLETE_NO_RESPONSE_READ"
    assert manifest["governance"]["perturbation_response_read"] is False
    assert manifest["asset"]["sha256"] == sha256_file(args.features)
    here = Path(__file__).resolve().parent
    code_names = ("common.py", "build_features.py", "fit_select.py", "lock.py", "score.py", "report.py",
                  "test_protocol.py", "freeze.py")
    payload = {"schema": "CANDIDATE_KNOWLEDGE_V1_PROSPECTIVE_FREEZE",
               "status": "FROZEN_BEFORE_RESPONSE_OR_G_SELECT_READ",
               "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "governance": {"response_or_relevance_read_before_freeze": False,
                              "feature_coverage_allowed_before_freeze": True,
                              "official_DEV_TEST_response_or_relevance_opened": False,
                              "benchmark_modified": False},
               "sha256": {"config": sha256_file(args.config), "design": sha256_file(args.design),
                          "split": sha256_file(cfg["paths"]["split"]),
                          "features": sha256_file(args.features),
                          "feature_manifest": sha256_file(args.feature_manifest),
                          "string": sha256_file(cfg["paths"]["string"]),
                          "sequence": sha256_file(cfg["paths"]["sequence"]),
                          "reactome_gmt_zip": sha256_file(cfg["paths"]["reactome_gmt_zip"]),
                          "ncbi_gene_info": sha256_file(cfg["paths"]["ncbi_gene_info"]),
                          "text_encoder_tree": sha256_tree(cfg["paths"]["text_encoder"])[0]},
               "code_sha256": {name: sha256_file(here / name) for name in code_names},
               "environment": {"python": platform.python_version(), "numpy": np.__version__,
                               "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
                               "torch": torch.__version__, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}}
    print(dump_json_exclusive(args.out, payload))


if __name__ == "__main__":
    main()
