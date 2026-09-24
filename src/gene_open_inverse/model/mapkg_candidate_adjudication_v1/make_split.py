#!/usr/bin/env python3
"""Freeze the fresh MAPKG split solely from the old G_fit identity pool."""
import argparse, hashlib, json
from pathlib import Path
from .common import dump_json_exclusive, load_json, resolve_config, sha256_file

def ordered(xs, salt):
    return sorted(map(str,xs), key=lambda x:(hashlib.sha256((salt+x).encode()).hexdigest(),x))

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--out",required=True); a=p.parse_args()
    cfg=resolve_config(a.config); parent=load_json(cfg["paths"]["parent_split"]); pool=parent["G_fit"]
    assert len(pool)==12220 and len(set(pool))==len(pool)
    s=cfg["split"]; rank=ordered(pool,s["salt"]); check=sorted(rank[:s["n_check"]]); select=sorted(rank[s["n_check"]:s["n_check"]+s["n_select"]]); fit=sorted(rank[s["n_check"]+s["n_select"]:])
    qref=sorted(ordered(fit,s["qref_salt"])[:s["n_qref"]]); qrank=sorted(set(fit)-set(qref))
    assert [len(fit),len(select),len(check)]==[s["n_fit"],s["n_select"],s["n_check"]]
    assert set(fit)|set(select)|set(check)==set(pool) and not(set(fit)&set(select)|set(fit)&set(check)|set(select)&set(check))
    out={"schema":"MAPKG_CANDIDATE_ADJUDICATION_V1_SPLIT","status":"FROZEN_BEFORE_RESPONSE_FIT",
      "rule":{"primary":"SHA256(split salt + gene): first KG_check, next KG_select, remainder KG_fit", "qref":"first n_qref of KG_fit by independent hash", "response_or_model_quantity_used":False},
      "counts":{"parent_G_fit":len(pool),"KG_fit":len(fit),"KG_select":len(select),"KG_check":len(check),"Q_ref_train":len(qref),"Q_rank_train":len(qrank)},
      "salts":{"split":s["salt"],"qref":s["qref_salt"]},"parent_split":{"path":cfg["paths"]["parent_split"],"sha256":sha256_file(cfg["paths"]["parent_split"])},
      "config_sha256":sha256_file(a.config),"KG_fit":fit,"KG_select":select,"KG_check":check,"Q_ref_train":qref,"Q_rank_train":qrank}
    print(json.dumps({"out":str(Path(a.out).resolve()),"sha256":dump_json_exclusive(a.out,out),"counts":out["counts"]},indent=2))
if __name__=="__main__": main()
