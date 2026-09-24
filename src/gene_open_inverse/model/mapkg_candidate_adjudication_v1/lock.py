#!/usr/bin/env python3
"""Lock the selected model and lambda before the one KG_check open."""
import argparse,time
from pathlib import Path
from .common import dump_json_exclusive,load_json,sha256_file
def main():
 p=argparse.ArgumentParser()
 for n in ("config","design","freeze","features","feature-manifest","fit-asset","models","selection","out"):p.add_argument("--"+n,required=True)
 a=p.parse_args(); fr=load_json(a.freeze); s=load_json(a.selection); assert fr["status"]=="FROZEN_BEFORE_RESPONSE_FIT" and s["status"]=="SELECTED_ON_KG_SELECT_ONLY" and not s["governance"]["KG_check_opened"]
 assert s["models"]["sha256"]==sha256_file(a.models); here=Path(__file__).parent
 for n,h in fr["code_sha256"].items():assert sha256_file(here/n)==h,"code drift "+n
 out={"schema":"MAPKG_V1_CHECK_LOCK","status":"LOCKED_BEFORE_KG_CHECK_OPEN","created":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"selected":{"alpha":s["fixed_alpha"],"lambda":s["arms"]["FUSION"]["selected"]["lambda"]},
 "governance":{"KG_check_opened_before_lock":False,"official_DEV_TEST_opened":False,"post_check_changes_allowed":False},"sha256":{k:sha256_file(getattr(a,k)) for k in ("config","design","freeze","features","feature_manifest","fit_asset","models","selection")},"code_sha256":{n:sha256_file(here/n) for n in ("common.py","score.py","report.py","verify.py")}}
 print(dump_json_exclusive(a.out,out))
if __name__=="__main__":main()
