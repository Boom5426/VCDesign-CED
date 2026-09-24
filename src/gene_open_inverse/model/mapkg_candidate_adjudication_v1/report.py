#!/usr/bin/env python3
"""Open KG_check relevance once, report ranking/operator evidence, and close the design."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
from .common import ARMS,axis_metrics,bootstrap,cosine_rows,dump_json_exclusive,evaluator_rows,exact_rows,load_json,load_module,resolve_config,save_npz_exclusive,sha256_file,stable_int
def summary(x):
 x=np.asarray(x);x=x[np.isfinite(x)];return {"n":len(x),"mean":float(x.mean()),"median":float(np.median(x)),"q10":float(np.quantile(x,.1)),"q90":float(np.quantile(x,.9))}
def main():
 p=argparse.ArgumentParser()
 for n in ("config","design","provenance","freeze","lock","feature-manifest","selection","scores","score-manifest","label","out","per-query-out","csv-out"):p.add_argument("--"+n,required=True)
 a=p.parse_args();cfg=resolve_config(a.config);lock=load_json(a.lock);sm=load_json(a.score_manifest);assert lock["status"]=="LOCKED_BEFORE_KG_CHECK_OPEN" and sm["status"]=="COMPLETE_LABEL_UNOPENED" and not sm["governance"]["KG_check_relevance_opened"]
 here=Path(__file__).parent
 for n in ("common.py","report.py","verify.py"):assert lock["code_sha256"][n]==sha256_file(here/n)
 s=np.load(a.scores);l=np.load(a.label);genes=l["genes"].astype(str);r=l["R"].astype(np.float32);assert np.array_equal(genes,s["genes"].astype(str)) and np.all(np.diag(r)==0)
 ev=load_module(cfg["paths"]["evaluator"],"mapkg_check_eval");nboot=cfg["diagnostics"]["bootstrap_n"];seed=cfg["diagnostics"]["bootstrap_seed"];rows={};trows={};summaries={};metric_audit=[]
 for arm in ARMS:
  off=evaluator_rows(ev,genes,r,s[arm+"__score"],"mapkg_KG_check");ind=exact_rows(r,s[arm+"__score"]);delta=np.abs(off-ind);mx=float(np.nanmax(delta));assert mx<=cfg["diagnostics"]["metric_tolerance"]
  pr=np.stack([exact_rows(r,x) for x in s[arm+"__perm"]]);tr=off-np.nanmean(pr,axis=0);rows[arm]=off;trows[arm]=tr
  summaries[arm]={"M":float(np.nanmean(off)),"T":float(np.nanmean(tr)),"permutation_mean_M":float(np.nanmean(pr)),"n_estimable":int(np.isfinite(off).sum()),"M_bootstrap":bootstrap(off,seed+stable_int(arm,"M")%100000,nboot),"T_bootstrap":bootstrap(tr,seed+stable_int(arm,"T")%100000,nboot)};metric_audit.append({"arm":arm,"max_abs":mx})
 pairs=(("MAPKG","STRING"),("MAPKG","MAPKG_shuffled"),("FUSION","STRING"),("FUSION","MAPKG"));contrasts={f"{x}_minus_{y}":bootstrap(rows[x]-rows[y],seed+stable_int(x,y)%100000,nboot) for x,y in pairs}
 contrasts.update({f"T_{x}_minus_T_{y}":bootstrap(trows[x]-trows[y],seed+stable_int("T",x,y)%100000,nboot) for x,y in pairs})
 measured=s["measured"].astype(np.float32);m0=s["measured_R0"].astype(np.float32);m1=s["measured_R1"].astype(np.float32);oprows={};operator={"measured_R0_R1_cosine":summary(cosine_rows(m0,m1)),"warning":"R0/R1 are noisy assay views, not error-free truth.","arms":{}}
 for arm in ARMS:
  pred=s[arm+"__pred"].astype(np.float32);half=.5*(cosine_rows(pred,m0)+cosine_rows(pred,m1));oprows[arm]=half;corr,r2=axis_metrics(pred,measured);operator["arms"][arm]={"cross_half_cosine":{"summary":summary(half),"bootstrap":bootstrap(half,seed+stable_int(arm,"op")%100000,nboot)},"per_axis_pearson":corr,"per_axis_R2":r2,"mean_axis_R2":float(np.mean(r2))}
 operator["contrasts"]={f"{x}_minus_{y}":bootstrap(oprows[x]-oprows[y],seed+stable_int("op",x,y)%100000,nboot) for x,y in pairs}
 rank_ms=contrasts["MAPKG_minus_STRING"]["ci95"][0]>0;rank_msh=contrasts["MAPKG_minus_MAPKG_shuffled"]["ci95"][0]>0;op_ms=operator["contrasts"]["MAPKG_minus_STRING"]["ci95"][0]>0;op_msh=operator["contrasts"]["MAPKG_minus_MAPKG_shuffled"]["ci95"][0]>0;fusion=contrasts["FUSION_minus_STRING"]["ci95"][0]>0
 if rank_ms and rank_msh and op_ms and op_msh:decision="A_MECHANISM_AWARE_STATIC_KNOWLEDGE_SUPPORTED"
 elif not rank_ms and fusion:decision="B_COMPLEMENTARY_LATE_FUSION_SUPPORTED"
 else:decision="C_CLOSE_GENERIC_STATIC_CANDIDATE_REPRESENTATION_RESEARCH"
 arrays={"genes":genes,"measured_R0_R1_cosine":cosine_rows(m0,m1)}
 for arm in ARMS:arrays[arm+"__M"]=rows[arm];arrays[arm+"__T"]=trows[arm];arrays[arm+"__operator_crosshalf_cosine"]=oprows[arm]
 ph=save_npz_exclusive(a.per_query_out,**arrays)
 with open(a.csv_out,"x",newline="",encoding="utf-8") as f:
  fields=["gene","measured_R0_R1_cosine"]+[f"{x}_{m}" for x in ARMS for m in ("M","T","operator_crosshalf_cosine")];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for i,g in enumerate(genes):
   z={"gene":g,"measured_R0_R1_cosine":arrays["measured_R0_R1_cosine"][i]}
   for arm in ARMS:
    for m in ("M","T","operator_crosshalf_cosine"):z[f"{arm}_{m}"]=arrays[f"{arm}__{m}"][i]
   w.writerow(z)
 out={"schema":"MAPKG_CANDIDATE_ADJUDICATION_V1_REPORT","status":"COMPLETE_AND_CLOSED","created":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"evidence_class":"fresh split within previously exposed old G_fit; development evidence",
 "ranking":summaries,"ranking_contrasts":contrasts,"operator_prediction":operator,"selection":load_json(a.selection),"feature_provenance":load_json(a.feature_manifest),"decision":{"rule":decision,"success_A":bool(rank_ms and rank_msh and op_ms and op_msh),"success_B":bool((not rank_ms) and fusion),"next_series_started":False},
 "historical_oracle_context":{"strict_crossview_candidate_gap":0.092382,"source":"TASK_VALIDITY_AND_BOTTLENECK_V1","used_for_gap_closure":False,"partly_label_coupled_ORACLE8_used":False},"classification":{"verified_fact":["fresh hash split","all response-derived transforms fit on KG_fit only","same fixed-alpha ridge and XSVD8 scorer"],"diagnostic":["operator metrics against noisy measured views"],"inference":[decision],"unresolved":["official DEV/TEST","prospective biological confirmation"]},
 "governance":{"KG_check_opened_once_after_lock":True,"official_DEV_TEST_opened":False,"benchmark_modified":False,"rescue_started":False},"metric_audit":metric_audit,"uncertainty":{"method":"paired bootstrap over origin gene","n":nboot,"candidate_pairs_independent":False},"outputs":{"per_query_npz":{"path":str(Path(a.per_query_out).resolve()),"sha256":ph},"per_query_csv":{"path":str(Path(a.csv_out).resolve()),"sha256":sha256_file(a.csv_out)}},"sha256":{n:sha256_file(getattr(a,n)) for n in ("config","design","provenance","freeze","lock","feature_manifest","selection","scores","score_manifest","label")}}
 print(json.dumps({"sha256":dump_json_exclusive(a.out,out),"decision":decision,"M":{x:summaries[x]["M"] for x in ARMS},"contrasts":contrasts},indent=2))
if __name__=="__main__":main()
