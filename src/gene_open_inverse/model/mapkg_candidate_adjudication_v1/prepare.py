#!/usr/bin/env python3
"""Rebuild all response-derived assets on KG_fit and open later roles in stages."""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from .common import dump_json_exclusive, load_json, load_module, resolve_config, save_npz_exclusive, sha256_file

def whitening(raw0,raw1):
    delta=raw1.astype(np.float64)-raw0.astype(np.float64);sig2=np.mean(delta*delta,axis=0)/2
    positive=sig2[sig2>0];assert len(positive);sig2=np.maximum(sig2,float(np.quantile(positive,.01)))
    aw=np.sqrt(1/(2*sig2));return (raw0.astype(np.float64)*aw).astype(np.float32),(raw1.astype(np.float64)*aw).astype(np.float32),sig2.astype(np.float32)

def apply_whitening(raw0,raw1,sig2):
    aw=np.sqrt(1/(2*sig2.astype(np.float64)));return (raw0.astype(np.float64)*aw).astype(np.float32),(raw1.astype(np.float64)*aw).astype(np.float32)

def matrix_free_xsvd(z0,z1,cfg,device):
    rc=cfg["representation"];k=int(rc["operator_dim"]);width=k+int(rc["xs_oversample"]);dev=torch.device(device)
    t0=torch.from_numpy(z0).to(dev);t1=torch.from_numpy(z1).to(dev);gen=torch.Generator(device="cpu").manual_seed(int(rc["xs_iteration_seed"]));q=torch.randn((z0.shape[1],width),generator=gen).to(dev);q,_=torch.linalg.qr(q,mode="reduced")
    def mul(x):return .5*(t0.T@(t1@x)+t1.T@(t0@x))
    for _ in range(int(rc["xs_iterations"])):q,_=torch.linalg.qr(mul(q),mode="reduced")
    mq=mul(q);small=.5*(q.T@mq+mq.T@q);values,vectors=torch.linalg.eigh(small);order=torch.argsort(values,descending=True)[:k];eig=values[order];basis=q@vectors[:,order];res=(mul(basis)-basis*eig[None]).norm(dim=0)/eig.abs().clamp(min=1)
    b=basis.cpu().numpy().astype(np.float32)
    for j in range(k):
        if b[np.argmax(np.abs(b[:,j])),j]<0:b[:,j]*=-1
    audit={"method":"deterministic matrix-free block subspace iteration","width":width,"iterations":int(rc["xs_iterations"]),"seed":int(rc["xs_iteration_seed"]),"eigenvalues":eig.cpu().numpy().tolist(),"max_relative_residual":float(res.max().cpu())};assert audit["max_relative_residual"]<5e-3,audit
    return b,audit

def coefficient(z,basis):
    z=z.astype(np.float64);return ((z@basis.astype(np.float64))/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)).astype(np.float32)

def response_stats(z0,z1):
    num=np.sum(z0.astype(np.float64)*z1.astype(np.float64),axis=1);den=np.linalg.norm(z0.astype(np.float64),axis=1)*np.linalg.norm(z1.astype(np.float64),axis=1)
    return (num/np.maximum(den,1e-12)).astype(np.float32),(.5*np.log(np.maximum(num,1e-8))).astype(np.float32)

def assign_strata(rel,mag,relcut,magcut):return (np.searchsorted(magcut,mag,side="left")*10+np.searchsorted(relcut,rel,side="left")).astype(np.int64)

def similarity(q0,q1,qpos,c0,c1,cpos,candidate_rel,device,block=128):
    dev=torch.device(device);tc0=torch.from_numpy(c0).to(dev);tc1=torch.from_numpy(c1).to(dev);cp=torch.from_numpy(cpos.astype(np.int64)).to(dev);cn0=(tc0*tc0).sum(1);cn1=(tc1*tc1).sum(1);relq,_=response_stats(q0,q1);out=np.empty((len(q0),len(c0)),np.float32);local={int(p):i for i,p in enumerate(cpos)}
    def one(a,b,bp,an,bn,apos):
        inner=a@b.T;a_at_b=a[:,bp];b_at_a=b[:,apos].T;inner=inner-a_at_b*b[torch.arange(len(b),device=dev),bp][None];na=an[:,None]-a_at_b.square();nb=bn[None]-b_at_a.square();return inner/(na.clamp(min=1e-12).sqrt()*nb.clamp(min=1e-12).sqrt())
    for start in range(0,len(q0),block):
        stop=min(start+block,len(q0));tq0=torch.from_numpy(q0[start:stop]).to(dev);tq1=torch.from_numpy(q1[start:stop]).to(dev);qp=torch.from_numpy(qpos[start:stop].astype(np.int64)).to(dev);n0=(tq0*tq0).sum(1);n1=(tq1*tq1).sum(1);value=(.5*(one(tq1,tc0,cp,n1,cn0,qp)+one(tq0,tc1,cp,n0,cn1,qp))).cpu().numpy().astype(np.float32);value[(relq[start:stop,None]<=0)|(candidate_rel[None,:]<=0)]=np.nan
        for row,gpos in enumerate(qpos[start:stop]):
            col=local.get(int(gpos));
            if col is not None:value[row,col]=np.nan
        out[start:stop]=value
    return out,relq

def make_label(fn,scores,refs,strata):
    value=fn(torch.from_numpy(scores),torch.from_numpy(refs),torch.from_numpy(strata),exclude_self=False).numpy().astype(np.float16);assert np.isfinite(value).all() and value.min()>=0 and value.max()<=1;return value

def gather(shadow, genes):
    p=Path(shadow); allg=np.load(p/"genes.npy").astype(str); universe=np.load(p/"universe.npy").astype(str)
    ri={g:i for i,g in enumerate(allg)}; ui={g:i for i,g in enumerate(universe)}
    rows=np.asarray([ri[g] for g in genes]); own=np.asarray([ui[g] for g in genes])
    out=[]
    for name in ("half_pooled_R0.npy","half_pooled_R1.npy"):
        a=np.load(p/name,mmap_mode="r"); x=np.asarray(a[rows],np.float32); x[np.arange(len(x)),own]=0; out.append(x)
    return universe,out[0],out[1],own

def check_freeze(freeze_path, args):
    f=load_json(freeze_path); assert f["status"]=="FROZEN_BEFORE_RESPONSE_FIT"
    assert f["sha256"]["config"]==sha256_file(args.config) and f["sha256"]["split"]==sha256_file(args.split)
    return f

def main():
    p=argparse.ArgumentParser(); p.add_argument("--stage",choices=("fit","select","check"),required=True)
    for n in ("config","split","design","freeze"): p.add_argument("--"+n,required=True)
    p.add_argument("--fit-asset"); p.add_argument("--lock"); p.add_argument("--out"); p.add_argument("--query-out"); p.add_argument("--label-out")
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu"); a=p.parse_args()
    cfg=resolve_config(a.config); split=load_json(a.split); freeze=check_freeze(a.freeze,a)
    builder=load_module(cfg["paths"]["relevance_builder"],"mapkg_relevance")
    if a.stage=="fit":
        assert a.out and not a.fit_asset
        genes=split["KG_fit"]; universe,r0,r1,pos=gather(cfg["paths"]["shadow"],genes)
        z0,z1,sigma2=whitening(r0,r1); del r0,r1
        basis,audit=matrix_free_xsvd(z0,z1,cfg,a.device)
        measured=.5*(coefficient(z0,basis)+coefficient(z1,basis)); u=coefficient(.5*(z0+z1),basis)
        gi={g:i for i,g in enumerate(genes)}; qidx=np.asarray([gi[g] for g in split["Q_ref_train"]])
        rel,mag=response_stats(z0,z1); cuts=np.linspace(.1,.9,9); relcut=np.quantile(rel,cuts).astype(np.float32); magcut=np.quantile(mag,cuts).astype(np.float32)
        meta={"schema":"MAPKG_V1_FIT_ASSET","status":"KG_FIT_ONLY","created":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"roles":{"response_rows":"KG_fit only","KG_select":False,"KG_check":False,"official_DEV_TEST":False},"basis":audit,
              "sha256":{"config":sha256_file(a.config),"split":sha256_file(a.split),"design":sha256_file(a.design),"freeze":sha256_file(a.freeze)}}
        digest=save_npz_exclusive(a.out,universe=universe,KG_fit=np.asarray(genes),Q_ref_train=np.asarray(split["Q_ref_train"]),V8=basis,sigma2=sigma2,
          Z_ref0=z0[qidx],Z_ref1=z1[qidx],ref_pos=pos[qidx],U_fit=u,measured_fit=measured,measured_R0_fit=coefficient(z0,basis),measured_R1_fit=coefficient(z1,basis),rel_cut=relcut,mag_cut=magcut,meta=np.asarray(json.dumps(meta)))
        meta["asset"]={"path":str(Path(a.out).resolve()),"sha256":digest}; dump_json_exclusive(str(Path(a.out).with_suffix(".json")),meta); print(digest); return
    assert a.fit_asset
    fit=np.load(a.fit_asset,allow_pickle=False); assert list(fit["KG_fit"].astype(str))==split["KG_fit"]
    role="KG_select" if a.stage=="select" else "KG_check"
    if role=="KG_check":
        assert a.lock and a.query_out and a.label_out; lock=load_json(a.lock); assert lock["status"]=="LOCKED_BEFORE_KG_CHECK_OPEN" and lock["sha256"]["fit_asset"]==sha256_file(a.fit_asset)
    else: assert a.out and not a.lock
    genes=split[role]; universe,r0,r1,pos=gather(cfg["paths"]["shadow"],genes); assert np.array_equal(universe,fit["universe"].astype(str))
    z0,z1=apply_whitening(r0,r1,fit["sigma2"]); del r0,r1
    rel,mag=response_stats(z0,z1); strata=assign_strata(rel,mag,fit["rel_cut"],fit["mag_cut"])
    sref,_=similarity(fit["Z_ref0"],fit["Z_ref1"],fit["ref_pos"],z0,z1,pos,rel,a.device)
    scores,relq=similarity(z0,z1,pos,z0,z1,pos,rel,a.device); relevance=make_label(builder.relevance_v2b,scores,sref,strata); assert np.all(np.diag(relevance)==0)
    a0=coefficient(z0,fit["V8"]); a1=coefficient(z1,fit["V8"]); u=coefficient(.5*(z0+z1),fit["V8"])
    meta={"schema":"MAPKG_V1_ROLE_ASSET","role":role,"created":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"official_DEV_TEST":False,"fit_asset_sha256":sha256_file(a.fit_asset)}
    if role=="KG_select":
        digest=save_npz_exclusive(a.out,genes=np.asarray(genes),U=u,R=relevance,relq=relq,relc=rel,measured=.5*(a0+a1),measured_R0=a0,measured_R1=a1,meta=np.asarray(json.dumps(meta)))
        meta["asset"]={"path":str(Path(a.out).resolve()),"sha256":digest}; dump_json_exclusive(str(Path(a.out).with_suffix(".json")),meta); print(digest)
    else:
        qd=save_npz_exclusive(a.query_out,genes=np.asarray(genes),U=u,measured=.5*(a0+a1),measured_R0=a0,measured_R1=a1,meta=np.asarray(json.dumps(meta)))
        ld=save_npz_exclusive(a.label_out,genes=np.asarray(genes),R=relevance,relq=relq,relc=rel,meta=np.asarray(json.dumps(meta)))
        print(json.dumps({"query":qd,"label":ld}))
if __name__=="__main__": main()
