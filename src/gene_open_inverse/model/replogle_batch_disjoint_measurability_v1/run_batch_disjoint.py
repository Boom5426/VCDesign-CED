#!/usr/bin/env python3
"""Frozen model-free batch-disjoint Replogle measurement run."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import anndata, numpy as np, pandas as pd, torch
from scipy.stats import rankdata, spearmanr

SEED=20260914; MIN_VIEW=25; K=10; K50=50; CONTROL='control'
def side(batch): return int.from_bytes(hashlib.sha256(f'REPLOGLE_BATCH_DISJOINT_V1|{SEED}|{batch}'.encode()).digest()[:8],'big')&1
def ci(x,nboot=5000,seed=SEED):
 x=np.asarray(x,dtype=float);x=x[np.isfinite(x)]
 if not len(x):return [None,None,None]
 rng=np.random.default_rng(seed);v=np.empty(nboot)
 for i in range(nboot):v[i]=np.mean(x[rng.integers(0,len(x),len(x))])
 return [float(np.mean(x)),float(np.quantile(v,.025)),float(np.quantile(v,.975))]
def topk(sim,k):
 idx=np.argpartition(-sim,kth=k-1,axis=1)[:,:k];rows=np.arange(len(sim))[:,None]
 return np.take_along_axis(idx,np.argsort(-sim[rows,idx],axis=1),axis=1)
def gpu_mm(a,b):
 with torch.no_grad(): return (torch.from_numpy(a).cuda()@torch.from_numpy(b).cuda()).cpu().numpy()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 ad=anndata.read_h5ad(a.input,backed='r');obs=ad.obs;pert=obs['perturbation'].astype(str).to_numpy();batch=obs['batch'].astype(str).to_numpy();cells=np.asarray(ad.obs_names.astype(str))
 labels=np.asarray(np.unique(pert[pert!=CONTROL]),dtype=str);batches=np.asarray(np.unique(batch),dtype=str);li={x:i for i,x in enumerate(labels)};bi={x:i for i,x in enumerate(batches)}
 pidx=pd.Categorical(pert,categories=list(labels)+[CONTROL]).codes.astype(np.int32);bidx=pd.Categorical(batch,categories=batches).codes.astype(np.int16)
 inc=np.zeros((len(labels),len(batches)),dtype=np.int32);m=pidx<len(labels);np.add.at(inc,(pidx[m],bidx[m]),1)
 bs=np.array([side(x) for x in batches],dtype=np.int8);ca=inc[:,bs==0].sum(1);cb=inc[:,bs==1].sum(1);keep=(ca>=MIN_VIEW)&(cb>=MIN_VIEW);elabels=np.asarray(labels[keep],dtype=str);global_to_e=np.full(len(labels)+1,-1,dtype=np.int32);global_to_e[np.flatnonzero(keep)]=np.arange(keep.sum());n=len(elabels);d=ad.n_vars
 sums=np.zeros((2,n,d),dtype=np.float32);cnt=np.zeros((2,n),dtype=np.int64);ctrl_sum=np.zeros((len(batches),d),dtype=np.float32);ctrl_n=np.zeros(len(batches),dtype=np.int64)
 chunk=5000
 for start in range(0,len(pert),chunk):
  stop=min(start+chunk,len(pert));x=np.asarray(ad.X[start:stop],dtype=np.float32);rs=x.sum(1);good=np.isfinite(rs)&(rs>0);x[good]=np.log1p(x[good]/rs[good,None]*1e4);x[~good]=np.nan
  pp=pidx[start:stop];bb=bidx[start:stop];vv=bs[bb]
  rows=np.flatnonzero(good&(pp==len(labels)))
  for b in np.unique(bb[rows]):
   q=rows[bb[rows]==b];ctrl_sum[b]+=np.nansum(x[q],axis=0);ctrl_n[b]+=len(q)
  rows=np.flatnonzero(good&(pp<len(labels))&(global_to_e[np.minimum(pp,len(labels))]>=0))
  ee=global_to_e[pp[rows]]
  for h in (0,1):
   rh=rows[vv[rows]==h];eh=global_to_e[pp[rh]]
   for e in np.unique(eh):
    q=rh[eh==e];sums[h,e]+=np.nansum(x[q],axis=0);cnt[h,e]+=len(q)
 ad.file.close()
 ctrl_mean=ctrl_sum/np.maximum(ctrl_n[:,None],1);inc_e=inc[keep].astype(np.float32)
 resp=np.empty_like(sums)
 for h in (0,1):
  w=inc_e.copy();w[:,bs!=h]=0;w/=np.maximum(w.sum(1,keepdims=True),1)
  cref=gpu_mm(w,ctrl_mean.astype(np.float32));resp[h]=sums[h]/cnt[h,:,None]-cref
 norm=np.linalg.norm(resp,axis=2);rn=resp/np.maximum(norm[:,:,None],1e-12)
 sim0=gpu_mm(rn[0],rn[0].T).astype(np.float32);sim1=gpu_mm(rn[1],rn[1].T).astype(np.float32);cross=gpu_mm(rn[0],rn[1].T).astype(np.float32)
 same=np.diag(cross).astype(float);cross_no=cross.copy();np.fill_diagonal(cross_no,-np.inf);nearest=np.maximum(np.max(cross_no,1),np.max(cross_no.T,1))
 np.fill_diagonal(sim0,-np.inf);np.fill_diagonal(sim1,-np.inf);t0=topk(sim0,K);t1=topk(sim1,K);t050=topk(sim0,K50);t150=topk(sim1,K50)
 held01=np.mean(np.take_along_axis(sim1,t0,axis=1),1);held10=np.mean(np.take_along_axis(sim0,t1,axis=1),1);over=np.array([len(set(t0[i])&set(t1[i]))/K for i in range(n)]);over50=np.array([len(set(t050[i])&set(t150[i]))/K50 for i in range(n)])
 win0=t0[:,0];win1=t1[:,0];wexact=(win0==win1).astype(float);ws01=sim1[np.arange(n),win0];ws10=sim0[np.arange(n),win1]
 rho=np.empty(n)
 for i in range(n):rho[i]=np.corrcoef(rankdata(sim0[i]),rankdata(sim1[i]))[0,1]
 rank01=1+np.sum(cross>same[:,None],axis=1);rank10=1+np.sum(cross.T>same[:,None],axis=1)
 edges=np.quantile(same,[1/3,2/3]);stratum=np.digitize(same,edges,right=True)
 batch_count=(inc_e>0).sum(1);batch_a=(inc_e[:,bs==0]>0).sum(1);batch_b=(inc_e[:,bs==1]>0).sum(1)
 np.savez_compressed(out/'per_query.npz',perturbation=elabels,n_cells_a=cnt[0],n_cells_b=cnt[1],batch_count=batch_count,batch_count_a=batch_a,batch_count_b=batch_b,target_reliability=same,same_target_similarity=same,nearest_alternative_similarity=nearest,reliability_tercile=stratum,heldout_s_a_to_b=held01,heldout_s_b_to_a=held10,query_spearman=rho,k10_overlap=over,k50_overlap=over50,winner_identity_a=elabels[win0],winner_identity_b=elabels[win1],winner_replication=wexact,winner_heldout_s_a_to_b=ws01,winner_heldout_s_b_to_a=ws10,target_rank_a_to_b=rank01,target_rank_b_to_a=rank10)
 strata={}
 for s,name in enumerate(['low','middle','high']):
  q=stratum==s;strata[name]={'n':int(q.sum()),'reliability':ci(same[q]),'heldout_s_a_to_b':ci(held01[q]),'heldout_s_b_to_a':ci(held10[q]),'query_spearman':ci(rho[q]),'k10_overlap':ci(over[q]),'winner_replication':ci(wexact[q]),'same_above_nearest_fraction':ci((same[q]>nearest[q]).astype(float))}
 summary={'schema':'REPLOGLE_BATCH_DISJOINT_MEASURABILITY_V1_RESULT','status':'EXECUTED_MODEL_FREE','source':a.input,'source_sha256':'d3269d9c863d96555eba768b42ab330cc1b57493cd33b4ac8fe7fbab8278f104','n_eligible':n,'n_batches':len(batches),'batches_a':int(np.sum(bs==0)),'batches_b':int(np.sum(bs==1)),'min_cells_per_view':MIN_VIEW,'control_matching':'same-view batch-specific controls weighted by perturbation cells per batch','view_claim':'technical-batch-disjoint, not biological replication','global':{'target_reliability_median':float(np.median(same)),'target_reliability_q1':float(np.quantile(same,.25)),'target_reliability_q3':float(np.quantile(same,.75)),'heldout_s_a_to_b':ci(held01),'heldout_s_b_to_a':ci(held10),'query_spearman':ci(rho),'k10_overlap':ci(over),'k10_random_expected':K/(n-1),'k10_overlap_enrichment':float(np.mean(over)/(K/(n-1))),'k50_overlap':ci(over50),'winner_replication':ci(wexact),'winner_positive_a_to_b':ci((ws01>0).astype(float)),'winner_positive_b_to_a':ci((ws10>0).astype(float)),'same_above_nearest_fraction':ci((same>nearest).astype(float)),'target_top10_a_to_b':ci((rank01<=10).astype(float)),'target_top10_b_to_a':ci((rank10<=10).astype(float))},'strata':strata,'cell_count_relationship':{'reliability':float(spearmanr(cnt.sum(0),same).statistic),'heldout_a_to_b':float(spearmanr(cnt.sum(0),held01).statistic),'heldout_b_to_a':float(spearmanr(cnt.sum(0),held10).statistic)},'governance':{'model_training':False,'beta_delta':False,'official_dev_test':False,'mapkg':False,'k2':False,'evaluation_v3':False},'created':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
 (out/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__':main()
