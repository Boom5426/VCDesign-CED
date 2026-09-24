#!/usr/bin/env python3
"""Blind KG_check scoring. This script cannot accept or read relevance."""
import argparse,json,time
from pathlib import Path
import numpy as np
from .common import ARMS,array_sha256,donor_maps,dump_json_exclusive,load_json,resolve_config,save_npz_exclusive,sha256_file
def main():
 p=argparse.ArgumentParser()
 for n in ("config","design","freeze","lock","features","models","selection","query","out","manifest"):p.add_argument("--"+n,required=True)
 a=p.parse_args();cfg=resolve_config(a.config);lock=load_json(a.lock);assert lock["status"]=="LOCKED_BEFORE_KG_CHECK_OPEN"
 for k in ("freeze","features","models","selection"):assert lock["sha256"][k]==sha256_file(getattr(a,k))
 here=Path(__file__).parent
 for n in ("common.py","score.py","report.py","verify.py"):assert lock["code_sha256"][n]==sha256_file(here/n)
 split=load_json(cfg["paths"]["split"]);f=np.load(a.features);m=np.load(a.models);qz=np.load(a.query);assert "R" not in qz.files
 genes=qz["genes"].astype(str);assert list(genes)==split["KG_check"];uni=f["universe"].astype(str);ui={g:i for i,g in enumerate(uni)};pos=np.asarray([ui[g] for g in genes]);q=qz["U"].astype(np.float32)
 donor=donor_maps(len(genes),len(m["KG_fit"]),cfg["diagnostics"]["target_derangement_salts"]);dq=m["U_fit"].astype(np.float32)[donor];sh=m["shuffle_donor"]
 pred={}
 for arm in ("STRING","MAPKG","MAPKG_shuffled"):
  base="MAPKG" if arm=="MAPKG_shuffled" else arm;x=f[base].astype(np.float32);cx=x[sh[pos]] if arm=="MAPKG_shuffled" else x[pos];pred[arm]=(cx.astype(np.float64)@m[arm+"__W"]+m[arm+"__b"]).astype(np.float32)
 lam=float(m["fusion_lambda"]);pred["FUSION"]=(lam*pred["STRING"]+(1-lam)*pred["MAPKG"]).astype(np.float32)
 arrays={"genes":genes,"query":q,"measured":qz["measured"],"measured_R0":qz["measured_R0"],"measured_R1":qz["measured_R1"]};audit={}
 for arm in ARMS:
  score=(q@pred[arm].T).astype(np.float32);perm=np.stack([(x@pred[arm].T).astype(np.float32) for x in dq]);arrays[arm+"__pred"]=pred[arm];arrays[arm+"__score"]=score;arrays[arm+"__perm"]=perm;audit[arm]={"pred":array_sha256(pred[arm]),"score":array_sha256(score),"perm":array_sha256(perm)}
 digest=save_npz_exclusive(a.out,**arrays);man={"schema":"MAPKG_V1_BLIND_SCORES","status":"COMPLETE_LABEL_UNOPENED","created":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"governance":{"KG_check_response_opened":True,"KG_check_relevance_opened":False,"official_DEV_TEST_opened":False},"asset":{"path":str(Path(a.out).resolve()),"sha256":digest},"arrays":audit,"sha256":{"query":sha256_file(a.query),"lock":sha256_file(a.lock),"score_code":sha256_file(__file__)}}
 print(dump_json_exclusive(a.manifest,man))
if __name__=="__main__":main()
