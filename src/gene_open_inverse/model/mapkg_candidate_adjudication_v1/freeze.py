#!/usr/bin/env python3
"""Bind the complete prospective implementation before response fit."""
import argparse, platform, time
from pathlib import Path
import numpy as np, torch
from .common import dump_json_exclusive, load_json, resolve_config, sha256_file

def main():
    p=argparse.ArgumentParser()
    for n in ("config","design","provenance","features","feature-manifest","out"): p.add_argument("--"+n,required=True)
    a=p.parse_args(); cfg=resolve_config(a.config); fm=load_json(a.feature_manifest)
    assert fm["status"]=="STATIC_FEATURES_COMPLETE_NO_RESPONSE_READ" and fm["asset"]["sha256"]==sha256_file(a.features)
    here=Path(__file__).parent; names=("common.py","build_features.py","prepare.py","fit_select.py","lock.py","score.py","report.py","verify.py","test_protocol.py","freeze.py")
    out={"schema":"MAPKG_CANDIDATE_ADJUDICATION_V1_FREEZE","status":"FROZEN_BEFORE_RESPONSE_FIT","created":time.strftime("%Y-%m-%dT%H:%M:%S%z"),
      "governance":{"response_or_relevance_read_before_freeze":False,"official_DEV_TEST_opened":False,"benchmark_modified":False},
      "sha256":{"config":sha256_file(a.config),"design":sha256_file(a.design),"provenance":sha256_file(a.provenance),"split":sha256_file(cfg["paths"]["split"]),"features":sha256_file(a.features),"feature_manifest":sha256_file(a.feature_manifest),"checkpoint":sha256_file(cfg["paths"]["mapkg_checkpoint"]),"esm":sha256_file(cfg["paths"]["esm"]),"string":sha256_file(cfg["paths"]["string"]),"relevance_builder":sha256_file(cfg["paths"]["relevance_builder"]),"evaluator":sha256_file(cfg["paths"]["evaluator"])},
      "code_sha256":{n:sha256_file(here/n) for n in names},"environment":{"python":platform.python_version(),"numpy":np.__version__,"torch":torch.__version__,"device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}}
    print(dump_json_exclusive(a.out,out))
if __name__=="__main__": main()
