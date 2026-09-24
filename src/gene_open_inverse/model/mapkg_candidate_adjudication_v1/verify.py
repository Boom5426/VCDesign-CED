#!/usr/bin/env python3
"""Independent final-artifact verification."""
import argparse,json
import numpy as np
from .common import ARMS,exact_rows,load_json,sha256_file
def main():
 p=argparse.ArgumentParser()
 for n in ("report","scores","label","per-query"):p.add_argument("--"+n,required=True)
 a=p.parse_args();r=load_json(a.report);s=np.load(a.scores);l=np.load(a.label);q=np.load(a.per_query)
 assert r["status"]=="COMPLETE_AND_CLOSED" and r["sha256"]["scores"]==sha256_file(a.scores) and r["sha256"]["label"]==sha256_file(a.label)
 assert r["outputs"]["per_query_npz"]["sha256"]==sha256_file(a.per_query) and np.array_equal(s["genes"],l["genes"])
 audit={}
 for arm in ARMS:
  rows=exact_rows(l["R"],s[arm+"__score"]);assert np.allclose(rows,q[arm+"__M"],equal_nan=True,atol=1e-12);audit[arm]={"M":float(np.nanmean(rows)),"report_M":r["ranking"][arm]["M"],"pass":abs(np.nanmean(rows)-r["ranking"][arm]["M"])<1e-12};assert audit[arm]["pass"]
 print(json.dumps({"PASS":True,"audit":audit},indent=2))
if __name__=="__main__":main()
