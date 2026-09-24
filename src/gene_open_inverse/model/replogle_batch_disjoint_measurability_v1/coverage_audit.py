#!/usr/bin/env python3
"""Metadata-only perturbation-by-batch coverage audit."""
import argparse, hashlib, json
from pathlib import Path
import anndata, numpy as np

SEED=20260914
def side(batch):
 return int.from_bytes(hashlib.sha256(f'REPLOGLE_BATCH_DISJOINT_V1|{SEED}|{batch}'.encode()).digest()[:8],'big')&1
def q(x,p): return float(np.quantile(x,p))
def desc(x):
 x=np.asarray(x); return {'min':int(x.min()),'q1':q(x,.25),'median':q(x,.5),'q3':q(x,.75),'max':int(x.max())}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
 ad=anndata.read_h5ad(a.input,backed='r'); pert=ad.obs['perturbation'].astype(str).to_numpy(); batch=ad.obs['batch'].astype(str).to_numpy()
 labels=np.unique(pert[pert!='control']); batches=np.unique(batch); bmap={b:side(b) for b in batches}
 lab_i={x:i for i,x in enumerate(labels)}; bat_i={x:i for i,x in enumerate(batches)}
 inc=np.zeros((len(labels),len(batches)),dtype=np.int32); ctrl=np.zeros(len(batches),dtype=np.int64)
 for p,b in zip(pert,batch):
  j=bat_i[b]
  if p=='control': ctrl[j]+=1
  else: inc[lab_i[p],j]+=1
 bc=(inc>0).sum(1); pair=inc[inc>0]; va=np.array([bmap[b]==0 for b in batches]); vb=~va; ca=inc[:,va].sum(1); cb=inc[:,vb].sum(1)
 out={'schema':'REPLOGLE_BATCH_DISJOINT_MEASURABILITY_V1_COVERAGE','source':a.input,'source_sha256':'d3269d9c863d96555eba768b42ab330cc1b57493cd33b4ac8fe7fbab8278f104','outcomes_read':False,'seed':SEED,'n_batches':len(batches),'batches_A':int(va.sum()),'batches_B':int(vb.sum()),'n_perturbations':len(labels),'batch_count_distribution':desc(bc),'perturbation_batch_cell_count_distribution':desc(pair),'fractions':{'ge2':float(np.mean(bc>=2)),'ge3':float(np.mean(bc>=3)),'ge4':float(np.mean(bc>=4))},'controls':{'batches_covered':int(np.sum(ctrl>0)),'all_batches_covered':bool(np.all(ctrl>0)),'cell_count_distribution':desc(ctrl)},'view_cell_counts':{'A':desc(ca),'B':desc(cb)},'eligible_by_min_cells_per_view':{str(m):int(np.sum((ca>=m)&(cb>=m))) for m in [5,10,15,20,25,30,40,50]},'fixed_min_cells_per_view':25,'fixed_eligible_count':int(np.sum((ca>=25)&(cb>=25))),'batch_assignment':bmap}
 Path(a.out).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out,sort_keys=True))
 ad.file.close()
if __name__=='__main__':main()
