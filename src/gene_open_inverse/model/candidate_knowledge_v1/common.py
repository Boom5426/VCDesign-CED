"""Result-independent utilities for CANDIDATE_KNOWLEDGE_V1."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np


ARM_NAMES = ("STRING", "SEQUENCE", "FUNCTION", "TEXT", "ALL", "ALL_shuffled")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_tree(path):
    root = Path(path)
    records = []
    for item in sorted(p for p in root.rglob("*") if p.is_file()):
        records.append("%s  %s" % (sha256_file(item), item.relative_to(root)))
    return hashlib.sha256(("\n".join(records) + "\n").encode()).hexdigest(), records


def array_sha256(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode())
    h.update(np.asarray(a.shape, np.int64).tobytes())
    h.update(a.tobytes())
    return h.hexdigest()


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def json_default(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def dump_json_exclusive(path, value):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False, default=json_default)
        handle.write("\n")
    return sha256_file(out)


def save_npz_exclusive(path, **arrays):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "xb") as handle:
        np.savez(handle, **arrays)
    return sha256_file(out)


def resolve_config(path):
    cfg = load_json(path)
    assert cfg["schema"] == "CANDIDATE_KNOWLEDGE_V1_CONFIG"
    cfg["paths"] = {key: os.path.expandvars(value) for key, value in cfg["paths"].items()}
    assert all("$" not in value for value in cfg["paths"].values()), "unresolved path variable"
    return cfg


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def standardise_present(raw, present, fit_pos, sd_floor):
    value = np.asarray(raw, np.float32)
    present = np.asarray(present, bool)
    fit_present = np.asarray(fit_pos, np.int64)[present[np.asarray(fit_pos, np.int64)]]
    assert len(fit_present) > 1 and np.isfinite(value[fit_present]).all()
    mu = value[fit_present].astype(np.float64).mean(0)
    sd = value[fit_present].astype(np.float64).std(0)
    keep = sd > float(sd_floor)
    safe = np.where(keep, sd, 1.0)
    out = ((value.astype(np.float64) - mu) / safe).astype(np.float32)
    out[:, ~keep] = 0.0
    out[~present] = 0.0
    assert np.isfinite(out).all() and np.all(out[~present] == 0)
    return out, mu, sd, keep, fit_present


def ridge_path(x, y, alphas):
    """Fit an exact intercepted ridge path after one symmetric eigendecomposition."""
    xd = np.asarray(x, np.float64)
    yd = np.asarray(y, np.float64)
    xm, ym = xd.mean(0), yd.mean(0)
    xc, yc = xd - xm, yd - ym
    gram = xc.T @ xc
    rhs = xc.T @ yc
    values, vectors = np.linalg.eigh(gram)
    values = np.maximum(values, 0.0)
    rotated = vectors.T @ rhs
    result = {}
    for alpha in map(float, alphas):
        w = vectors @ (rotated / (values[:, None] + alpha))
        b = ym - xm @ w
        result[alpha] = (w, b)
    return result, {"gram_min_eigenvalue": float(values.min()),
                    "gram_max_eigenvalue": float(values.max()), "feature_dim": int(xd.shape[1])}


def exact_softndcg10_rows(relevance, scores):
    r = np.asarray(relevance, np.float64)
    s = np.asarray(scores, np.float64).copy()
    assert r.ndim == 2 and r.shape == s.shape and r.shape[0] == r.shape[1]
    np.fill_diagonal(s, -np.inf)
    k = min(10, r.shape[1])
    order = np.lexsort((r, -s), axis=1)[:, :k]
    gain = np.take_along_axis(r, order, axis=1)
    discount = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
    dcg = gain @ discount
    idcg = (-np.sort(-r, axis=1)[:, :k]) @ discount
    out = np.full(len(r), np.nan, np.float64)
    valid = idcg > 0
    out[valid] = dcg[valid] / idcg[valid]
    return out


def evaluator_rows(evaluator, genes, relevance, scores, label_id):
    genes = np.asarray(genes).astype(str)
    n = len(genes)
    dummy = np.ones(n, np.float64)
    bench = evaluator.Benchmark(genes, genes, np.asarray(relevance), np.arange(n, dtype=np.int64),
                                dummy, dummy, dummy, np.zeros(n, bool),
                                label_id=label_id, split="development")

    def pool(row):
        keep = np.ones(n, bool)
        keep[row] = False
        return keep

    per = evaluator._per_query(bench, lambda row: np.asarray(scores)[row], pool, ks=(10,))
    return np.asarray(per["ndcg@10"], np.float64)


def target_donor_maps(n_recipient, n_donor, salts):
    assert n_donor >= n_recipient
    out = np.empty((len(salts), n_recipient), np.int64)
    for row, salt in enumerate(salts):
        order = sorted(range(n_donor), key=lambda i: (stable_int("ck-target", salt, i), i))
        out[row] = order[:n_recipient]
    return out


def metric_rows(relevance, query, candidate, donor_query):
    true = exact_softndcg10_rows(relevance, np.asarray(query) @ np.asarray(candidate).T)
    perm = np.stack([exact_softndcg10_rows(relevance, q @ np.asarray(candidate).T)
                     for q in np.asarray(donor_query)])
    trow = true - np.nanmean(perm, axis=0)
    return ({"M": float(np.nanmean(true)), "T": float(np.nanmean(trow)),
             "permutation_mean_M": float(np.nanmean(perm)),
             "permutation_M": np.nanmean(perm, axis=1).tolist(),
             "n_estimable": int(np.isfinite(true).sum())}, true, trow)


def bootstrap(rows, seed, n_boot):
    value = np.asarray(rows, np.float64)
    value = value[np.isfinite(value)]
    assert len(value)
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(n_boot), np.float64)
    for i in range(len(means)):
        means[i] = value[rng.integers(0, len(value), len(value))].mean()
    return {"mean": float(value.mean()),
            "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
            "n_origin_clusters": int(len(value)), "unit": "origin gene"}


def bootstrap_ratio(numerator_rows, denominator_rows, seed, n_boot):
    n = np.asarray(numerator_rows, np.float64)
    d = np.asarray(denominator_rows, np.float64)
    valid = np.isfinite(n) & np.isfinite(d)
    n, d = n[valid], d[valid]
    assert len(n) and abs(d.mean()) > 1e-12
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(n_boot), np.float64)
    for i in range(len(values)):
        take = rng.integers(0, len(n), len(n))
        values[i] = n[take].mean() / d[take].mean()
    return {"ratio_of_means": float(n.mean() / d.mean()),
            "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
            "n_origin_clusters": int(len(n)), "unit": "paired origin gene bootstrap"}


def stable_int(*parts):
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:16], 16)


def shuffled_donor_index(universe, gfit, seed):
    """Balanced G_fit donor mapping; the G_fit restriction is an exact derangement."""
    universe = np.asarray(universe).astype(str)
    gfit = np.asarray(gfit).astype(str)
    uindex = {g: i for i, g in enumerate(universe)}
    fit_pos = np.asarray([uindex[g] for g in gfit], np.int64)
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(len(gfit))
    while np.any(perm == np.arange(len(gfit))):
        perm = rng.permutation(len(gfit))
    donor = np.empty(len(universe), np.int64)
    donor[fit_pos] = fit_pos[perm]
    fit_set = set(fit_pos.tolist())
    other = np.asarray([i for i in range(len(universe)) if i not in fit_set], np.int64)
    receivers = sorted(other.tolist(), key=lambda i: (stable_int("ck-shuffle-recipient", seed, universe[i]), i))
    cycle = rng.permutation(fit_pos)
    for j, row in enumerate(receivers):
        donor[row] = cycle[j % len(cycle)]
    assert np.all(donor[fit_pos] != fit_pos)
    assert set(donor.tolist()) <= set(fit_pos.tolist())
    return donor


def response_operators(shadow_dir, genes, fit_asset):
    """Return average-view, R0 and R1 operators on the inherited half-pooled basis."""
    shadow = Path(shadow_dir)
    source_genes = np.load(shadow / "genes.npy").astype(str)
    index = {g: i for i, g in enumerate(source_genes)}
    rows = np.asarray([index[g] for g in map(str, genes)], np.int64)
    sigma2 = fit_asset["sigma2"].astype(np.float64)
    basis = fit_asset["V8"].astype(np.float64)
    aw = np.sqrt(1.0 / (2.0 * sigma2))

    def one(name):
        raw = np.load(shadow / ("half_pooled_" + name + ".npy"), mmap_mode="r")
        z = np.asarray(raw[rows], np.float64) * aw
        norm = np.linalg.norm(z, axis=1, keepdims=True)
        a = (z @ basis) / np.maximum(norm, 1e-12)
        return z, a.astype(np.float32)

    z0, a0 = one("R0")
    z1, a1 = one("R1")
    zmean = 0.5 * (z0 + z1)
    u = ((zmean @ basis) / np.maximum(np.linalg.norm(zmean, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    del z0, z1, zmean
    return 0.5 * (a0 + a1), a0, a1, u


def pearson_columns(pred, measured):
    x = np.asarray(pred, np.float64)
    y = np.asarray(measured, np.float64)
    corr, r2 = [], []
    for axis in range(x.shape[1]):
        xx, yy = x[:, axis], y[:, axis]
        corr.append(float(np.corrcoef(xx, yy)[0, 1]) if xx.std() > 0 and yy.std() > 0 else None)
        denom = float(np.sum((yy - yy.mean()) ** 2))
        r2.append(float(1.0 - np.sum((yy - xx) ** 2) / denom) if denom > 0 else None)
    return corr, r2


def cosine_rows(a, b):
    x, y = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.sum(x * y, axis=1) / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1), 1e-12)
