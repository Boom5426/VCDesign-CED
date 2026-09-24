#!/usr/bin/env python3
"""Extract static STRING and frozen official MAP-KG gene features; never reads response."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from .common import array_sha256, dump_json_exclusive, load_json, resolve_config, save_npz_exclusive, sha256_file, standardise_present

class ResidualProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(5120,1536),nn.LayerNorm(1536),nn.GELU(),nn.Dropout(.1),nn.Linear(1536,1024))
        self.residual_proj=nn.Linear(5120,1024); self.gate=nn.Sequential(nn.Linear(5120,1024),nn.Sigmoid())
    def forward(self,x):
        t=self.mlp(x); r=self.residual_proj(x); g=self.gate(x); return g*t+(1-g)*r

def md5_file(path):
    h=hashlib.md5()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def coverage(present,pos):
    x=np.asarray(present)[np.asarray(pos)]; return {"present":int(x.sum()),"total":len(x),"fraction":float(x.mean())}

def extract_state(checkpoint):
    obj=torch.load(checkpoint,map_location="cpu",weights_only=False)
    assert isinstance(obj,dict) and "model_state_dict" in obj
    full=obj["model_state_dict"]; expected=set(ResidualProjector().state_dict())
    chosen={}
    for k,v in full.items():
        marker="gene_projector."
        if marker in k: chosen[k.split(marker,1)[1]]=v
    assert set(chosen)==expected, {"missing":sorted(expected-set(chosen)),"extra":sorted(set(chosen)-expected)}
    # MAP-KG's static MoleculeSTM submodule legitimately has a SMILES ``decoder``.
    # Only response-prediction-specific state is forbidden; we use none of the
    # molecule/text parameters in this experiment in any case.
    forbidden=[k for k in full if any(x in k.lower() for x in ("perturb", "scfm", "condition_encoder", "cell_projector", "gene_decoder", "project_out"))]
    assert not forbidden, "downstream perturbation keys in alleged MAP-KG checkpoint"
    return chosen, sorted(full), {k:obj.get(k) for k in ("epoch","step","loss")}

def main():
    p=argparse.ArgumentParser()
    for n in ("config","design","provenance","out","manifest"): p.add_argument("--"+n,required=True)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu"); a=p.parse_args()
    cfg=resolve_config(a.config); split=load_json(cfg["paths"]["split"]); prov=Path(a.provenance).read_text()
    assert "Gate status: **PASS**" in prov and "PENDING ACQUISITION" not in prov, "legality gate not fully passed"
    cp=cfg["paths"]["mapkg_checkpoint"]
    assert Path(cp).stat().st_size==cfg["mapkg"]["checkpoint_size"]
    assert md5_file(cp)==cfg["mapkg"]["checkpoint_md5"]
    assert sha256_file(cfg["paths"]["esm"])==cfg["mapkg"]["esm_sha256"]
    z=np.load(cfg["paths"]["string"],allow_pickle=False); universe=z["genes"].astype(str); raw_s=z["Z"].astype(np.float32)
    p_s=(z["degree"]>0)&np.isfinite(raw_s).all(1)&(np.linalg.norm(raw_s,axis=1)>0); z.close()
    ui={g:i for i,g in enumerate(universe)}; roles={r:np.asarray([ui[g] for g in split[r]]) for r in ("KG_fit","KG_select","KG_check")}; fp=roles["KG_fit"]
    string,smu,ssd,skeep=standardise_present(raw_s,p_s,fp,cfg["representation"]["sd_floor"])
    state,all_keys,training_meta=extract_state(cp); model=ResidualProjector(); model.load_state_dict(state,strict=True); model.to(a.device).eval()
    esm=torch.load(cfg["paths"]["esm"],map_location="cpu",weights_only=False); assert isinstance(esm,dict)
    raw_k=np.zeros((len(universe),1024),np.float32); p_k=np.asarray([g in esm for g in universe])
    idx=np.flatnonzero(p_k); batch=32
    with torch.inference_mode():
        for start in range(0,len(idx),batch):
            rows=idx[start:start+batch]; x=torch.stack([esm[universe[i]].float() for i in rows]).to(a.device)
            raw_k[rows]=model(x).float().cpu().numpy()
    mapkg,kmu,ksd,kkeep=standardise_present(raw_k,p_k,fp,cfg["representation"]["sd_floor"])
    arrays={"universe":universe,"fit_pos":fp,"STRING":string,"STRING_present":p_s,"MAPKG":mapkg,"MAPKG_present":p_k,
            "STRING_mu":smu,"STRING_sd":ssd,"MAPKG_mu":kmu,"MAPKG_sd":ksd}
    digest=save_npz_exclusive(a.out,**arrays)
    manifest={"schema":"MAPKG_CANDIDATE_ADJUDICATION_V1_STATIC_FEATURES","status":"STATIC_FEATURES_COMPLETE_NO_RESPONSE_READ",
      "created":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"governance":{"perturbation_response_read":False,"relevance_read":False,"downstream_MAP_checkpoint_used":False,"MAPKG_finetuned":False},
      "asset":{"path":str(Path(a.out).resolve()),"sha256":digest},"coverage":{name:{r:coverage(p,roles[r]) for r in roles}|{"universe":coverage(p,np.arange(len(p)))} for name,p in (("STRING",p_s),("MAPKG",p_k))},
      "details":{"STRING":{"dimension":512,"nondegenerate":int(skeep.sum())},"MAPKG":{"dimension":1024,"nondegenerate":int(kkeep.sum()),"definition":cfg["mapkg"]["embedding"]}},
      "checkpoint":{"path":cp,"size":Path(cp).stat().st_size,"md5":md5_file(cp),"sha256":sha256_file(cp),"state_key_count":len(all_keys),"state_keys_sha256":array_sha256(np.asarray(all_keys)),"training_meta":training_meta},
      "sha256":{"config":sha256_file(a.config),"design":sha256_file(a.design),"provenance":sha256_file(a.provenance),"split":sha256_file(cfg["paths"]["split"]),"string":sha256_file(cfg["paths"]["string"]),"esm":sha256_file(cfg["paths"]["esm"]),"code":sha256_file(__file__)}}
    print(json.dumps({"manifest":str(Path(a.manifest).resolve()),"sha256":dump_json_exclusive(a.manifest,manifest),"coverage":manifest["coverage"]},indent=2))
if __name__=="__main__": main()
