"""Small deterministic utilities for MAPKG_CANDIDATE_ADJUDICATION_V1."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

ARMS = ("STRING", "MAPKG", "MAPKG_shuffled", "FUSION")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_tree(path):
    root = Path(path)
    rows = [f"{sha256_file(p)}  {p.relative_to(root)}" for p in sorted(x for x in root.rglob("*") if x.is_file())]
    return hashlib.sha256(("\n".join(rows) + "\n").encode()).hexdigest(), rows


def array_sha256(x):
    a = np.ascontiguousarray(x)
    h = hashlib.sha256(a.dtype.str.encode())
    h.update(np.asarray(a.shape, np.int64).tobytes())
    h.update(a.tobytes())
    return h.hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dump_json_exclusive(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False, default=json_default); f.write("\n")
    return sha256_file(p)


def save_npz_exclusive(path, **arrays):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "xb") as f: np.savez(f, **arrays)
    return sha256_file(p)


def json_default(x):
    if isinstance(x, (np.bool_, bool)): return bool(x)
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, np.floating): return float(x)
    if isinstance(x, np.ndarray): return x.tolist()
    raise TypeError(type(x))


def resolve_config(path):
    cfg = load_json(path)
    assert cfg["schema"] == "MAPKG_CANDIDATE_ADJUDICATION_V1_CONFIG"
    cfg["paths"] = {k: os.path.expandvars(v) for k, v in cfg["paths"].items()}
    assert all("$" not in v for v in cfg["paths"].values())
    return cfg


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def stable_int(*parts):
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def standardise_present(raw, present, fit_pos, floor):
    raw = np.asarray(raw, np.float32); present = np.asarray(present, bool); fit_pos = np.asarray(fit_pos)
    fp = fit_pos[present[fit_pos]]
    assert len(fp) > 1 and np.isfinite(raw[fp]).all()
    mu = raw[fp].astype(np.float64).mean(0); sd = raw[fp].astype(np.float64).std(0)
    keep = sd > float(floor); safe = np.where(keep, sd, 1.0)
    out = ((raw.astype(np.float64) - mu) / safe).astype(np.float32)
    out[:, ~keep] = 0; out[~present] = 0
    assert np.isfinite(out).all()
    return out, mu, sd, keep


def ridge_fit(x, y, alpha):
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    xm, ym = x.mean(0), y.mean(0); xc, yc = x-xm, y-ym
    w = np.linalg.solve(xc.T @ xc + float(alpha)*np.eye(x.shape[1]), xc.T @ yc)
    return w, ym - xm @ w


def exact_rows(relevance, scores):
    r = np.asarray(relevance, np.float64); s = np.asarray(scores, np.float64).copy()
    assert r.shape == s.shape and r.shape[0] == r.shape[1]
    np.fill_diagonal(s, -np.inf); k = min(10, len(r))
    order = np.lexsort((r, -s), axis=1)[:, :k]
    d = 1 / np.log2(np.arange(2, k+2)); dcg = np.take_along_axis(r, order, axis=1) @ d
    idcg = (-np.sort(-r, axis=1)[:, :k]) @ d
    out = np.full(len(r), np.nan); ok = idcg > 0; out[ok] = dcg[ok]/idcg[ok]
    return out


def evaluator_rows(evaluator, genes, relevance, scores, label):
    genes = np.asarray(genes).astype(str); n = len(genes); ones = np.ones(n)
    b = evaluator.Benchmark(genes, genes, np.asarray(relevance), np.arange(n), ones, ones, ones,
                            np.zeros(n, bool), label_id=label, split="development")
    return np.asarray(evaluator._per_query(b, lambda i: np.asarray(scores)[i],
                      lambda i: np.arange(n) != i, ks=(10,))["ndcg@10"], np.float64)


def donor_maps(n_recipient, n_donor, salts):
    assert n_donor >= n_recipient
    return np.asarray([sorted(range(n_donor), key=lambda i: (stable_int("target", salt, i), i))[:n_recipient]
                       for salt in salts], np.int64)


def shuffled_donors(universe, fit_genes, seed):
    universe = np.asarray(universe).astype(str); fit_genes = np.asarray(fit_genes).astype(str)
    ui = {g:i for i,g in enumerate(universe)}; fp = np.asarray([ui[g] for g in fit_genes])
    rng = np.random.default_rng(seed); perm = rng.permutation(len(fp))
    while np.any(perm == np.arange(len(fp))): perm = rng.permutation(len(fp))
    donor = np.empty(len(universe), np.int64); donor[fp] = fp[perm]
    fset = set(fp.tolist()); other = sorted((i for i in range(len(universe)) if i not in fset),
        key=lambda i:(stable_int("shuffle", seed, universe[i]), i)); cycle = rng.permutation(fp)
    for j,i in enumerate(other): donor[i] = cycle[j % len(cycle)]
    assert np.all(donor[fp] != fp) and set(donor.tolist()) <= fset
    return donor


def bootstrap(rows, seed, n):
    x = np.asarray(rows, np.float64); x = x[np.isfinite(x)]; assert len(x)
    rng = np.random.default_rng(seed); means = np.asarray([x[rng.integers(0,len(x),len(x))].mean() for _ in range(n)])
    return {"mean":float(x.mean()), "ci95":[float(np.quantile(means,.025)),float(np.quantile(means,.975))],
            "n_origin_clusters":len(x), "unit":"paired origin gene"}


def cosine_rows(a, b):
    a=np.asarray(a,np.float64); b=np.asarray(b,np.float64)
    return np.sum(a*b,1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-12)


def axis_metrics(pred, truth):
    pred=np.asarray(pred,np.float64); truth=np.asarray(truth,np.float64); corr=[]; r2=[]
    for j in range(pred.shape[1]):
        x,y=pred[:,j],truth[:,j]
        corr.append(float(np.corrcoef(x,y)[0,1]) if x.std()>0 and y.std()>0 else None)
        den=np.sum((y-y.mean())**2); r2.append(float(1-np.sum((y-x)**2)/den) if den>0 else None)
    return corr,r2
