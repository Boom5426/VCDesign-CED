#!/usr/bin/env python3
"""Fit fixed-alpha operator ridges on KG_fit and select one late-fusion lambda on KG_select."""
import argparse, json, time
from pathlib import Path
import numpy as np
from .common import (array_sha256, donor_maps, dump_json_exclusive, evaluator_rows, exact_rows, load_json, load_module,
                     resolve_config, ridge_fit, save_npz_exclusive, sha256_file, shuffled_donors)

def summarize(rel,q,pred,donor):
    true=exact_rows(rel,q@pred.T); perm=np.stack([exact_rows(rel,x@pred.T) for x in donor]); t=true-np.nanmean(perm,axis=0)
    return {"M":float(np.nanmean(true)),"T":float(np.nanmean(t)),"permutation_mean_M":float(np.nanmean(perm)),"n_estimable":int(np.isfinite(true).sum())}

def main():
    p=argparse.ArgumentParser()
    for n in ("config","design","freeze","features","feature-manifest","fit-asset","select-asset","models-out","selection-out"):p.add_argument("--"+n,required=True)
    a=p.parse_args(); cfg=resolve_config(a.config); fr=load_json(a.freeze); assert fr["status"]=="FROZEN_BEFORE_RESPONSE_FIT"
    here=Path(__file__).parent
    for n,h in fr["code_sha256"].items(): assert sha256_file(here/n)==h,"code drift "+n
    for n in ("features","feature_manifest"): assert fr["sha256"][n]==sha256_file(getattr(a,n.replace('_','-'),None) or getattr(a,n))
    split=load_json(cfg["paths"]["split"]); feat=np.load(a.features); fit=np.load(a.fit_asset); sel=np.load(a.select_asset)
    uni=feat["universe"].astype(str); ui={g:i for i,g in enumerate(uni)}; fp=np.asarray([ui[g] for g in split["KG_fit"]]); sp=np.asarray([ui[g] for g in split["KG_select"]])
    assert list(sel["genes"].astype(str))==split["KG_select"] and list(fit["KG_fit"].astype(str))==split["KG_fit"]
    y=fit["measured_fit"].astype(np.float32); q=sel["U"].astype(np.float32); rel=sel["R"].astype(np.float32)
    donors=donor_maps(len(sp),len(fp),cfg["diagnostics"]["target_derangement_salts"]); donor_q=fit["U_fit"].astype(np.float32)[donors]
    evaluator=load_module(cfg["paths"]["evaluator"],"mapkg_select_eval"); shuffled=shuffled_donors(uni,split["KG_fit"],cfg["shuffle"]["seed"])
    arrays={"universe":uni,"KG_fit":np.asarray(split["KG_fit"]),"U_fit":fit["U_fit"].astype(np.float32),"shuffle_donor":shuffled}; records={}
    predictions={}
    for arm in ("STRING","MAPKG","MAPKG_shuffled"):
        base="MAPKG" if arm=="MAPKG_shuffled" else arm; x=feat[base].astype(np.float32)
        tx=x[shuffled[fp]] if arm=="MAPKG_shuffled" else x[fp]; cx=x[shuffled[sp]] if arm=="MAPKG_shuffled" else x[sp]
        w,b=ridge_fit(tx,y,cfg["ridge"]["alpha"]); pred=(cx.astype(np.float64)@w+b).astype(np.float32); predictions[arm]=pred
        rec=summarize(rel,q,pred,donor_q); official=evaluator_rows(evaluator,sel["genes"],rel,q@pred.T,"mapkg_KG_select")
        assert np.nanmax(np.abs(official-exact_rows(rel,q@pred.T)))<=cfg["diagnostics"]["metric_tolerance"] and abs(np.nanmean(official)-rec["M"])<=cfg["diagnostics"]["metric_tolerance"]
        records[arm]=rec; arrays[arm+"__W"]=w; arrays[arm+"__b"]=b
    grid=[]
    for lam in cfg["fusion"]["lambda_grid"]:
        pred=lam*predictions["STRING"]+(1-lam)*predictions["MAPKG"]; grid.append({"lambda":lam,**summarize(rel,q,pred,donor_q)})
    chosen=max(grid,key=lambda x:(x["M"],x["lambda"])); records["FUSION"]={"selected":chosen,"grid":grid}; arrays["fusion_lambda"]=np.asarray(chosen["lambda"])
    selection={"schema":"MAPKG_V1_SELECTION","status":"SELECTED_ON_KG_SELECT_ONLY","created":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"fixed_alpha":cfg["ridge"]["alpha"],"arms":records,
      "governance":{"KG_check_opened":False,"official_DEV_TEST_opened":False},"sha256":{"config":sha256_file(a.config),"design":sha256_file(a.design),"freeze":sha256_file(a.freeze),"features":sha256_file(a.features),"feature_manifest":sha256_file(a.feature_manifest),"fit_asset":sha256_file(a.fit_asset),"select_asset":sha256_file(a.select_asset),"fit_target":array_sha256(y)}}
    arrays["meta"]=np.asarray(json.dumps(selection,sort_keys=True)); mh=save_npz_exclusive(a.models_out,**arrays); selection["models"]={"path":str(Path(a.models_out).resolve()),"sha256":mh}
    print(json.dumps({"sha256":dump_json_exclusive(a.selection_out,selection),"lambda":chosen,"arms":records},indent=2))
if __name__=="__main__":main()
