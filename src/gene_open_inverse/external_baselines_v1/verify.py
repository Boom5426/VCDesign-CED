#!/usr/bin/env python3
"""Independent witness (lesson L3).  Imports no production scoring, metric or PHR code.

From the frozen packs, C-018's effects, the GEARS and CellNavi artifacts and the quarter builds, with
its own loops, it recomputes and compares with the run:
  1. VCDesign's fused scores (own cosine, own row-z) and GEARS's cosine scores (own cross-fit);
  2. CellNavi's score matrix from its saved per-(query, view) log-probabilities (own name mapping);
  3. per-row BU@20 for VCDesign, GEARS and CellNavi (own ranking loop) and their per-context medians;
  4. realized J (own closed-form attenuation) and PHR@20 for BASE and VCDesign in one context.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

C018 = "inputs/runs/decision_consistent_effect_v1"
FC = "inputs/runs/four_context_v1"


def _unit(x):
    return x / np.maximum(np.sqrt((x * x).sum(axis=-1, keepdims=True)), 1e-300)


def _cos(query, effect):
    out = _unit(query) @ _unit(effect).T
    out[:, np.sqrt((effect * effect).sum(1)) <= 0] = 0.0
    return out


def _z(m):
    s = m.std(axis=1, keepdims=True)
    s[s < 1e-12] = 1e-12
    return (m - m.mean(axis=1, keepdims=True)) / s


def _order(scores, ids):
    """Stable descending score, ties by candidate identity ascending (the frozen convention)."""
    return np.lexsort((ids, -scores))


def _bu(scores, utility, ids, selfpos, columns, budget=20, realized=None):
    d = 1.0 / np.log2(np.arange(1, budget + 1) + 1.0)
    bu, phr = [], []
    for row in range(scores.shape[0]):
        legal = np.asarray([c for c in columns.tolist() if c != int(selfpos[row])])
        u = utility[row, legal]
        order = _order(scores[row, legal], ids[legal])[:budget]
        top = (d * u[order]).sum() / d.sum()
        best = (d * np.sort(u)[::-1][:budget]).sum() / d.sum()
        bu.append((top - u.mean()) / (best - u.mean()))
        if realized is not None:
            phr.append(float((realized[row, legal][order] > 0).mean()))
    return np.asarray(bu), (np.asarray(phr) if realized is not None else None)


def _crossfit(seen, parts, eligible, folds):
    out = seen.copy()
    for f, part in enumerate(parts):
        out[eligible & (folds == f)] = part[eligible & (folds == f)]
    return out


def _j(q, view):
    """Own implementation of the attenuation-only deployment value, chosen on side v, realized on 1-v."""
    a, b = q[2 * view], q[2 * view + 1]
    c, d = q[2 - 2 * view], q[3 - 2 * view]
    s_p = 0.5 * (a @ b.T + (a @ b.T).T)
    r_p = np.einsum("ij,ij->i", a, b)
    s_g = 0.5 * (c @ d.T + (c @ d.T).T)
    r_g = np.einsum("ij,ij->i", c, d)
    best_a = np.zeros_like(s_p)
    best_v = np.zeros_like(s_p)
    for candidate in (np.ones_like(s_p), np.clip(np.where(r_p[None] != 0, s_p / np.where(r_p[None] != 0, r_p[None], 1), 0), 0, 1)):
        value = 2 * candidate * s_p - candidate ** 2 * r_p[None]
        better = value > best_v + 0.0
        best_a = np.where(better, candidate, best_a)
        best_v = np.where(better, value, best_v)
    return 2 * best_a * s_g - best_a ** 2 * r_g[None]


def main() -> int:
    parser = argparse.ArgumentParser(description="independent external-baseline witness")
    parser.add_argument("--run-dir", dest="run_dir", required=True)
    parser.add_argument("--contexts", nargs="+", default=["RPE1", "Jurkat"])
    parser.add_argument("--phr-context", dest="phr_context", default="Jurkat")
    args = parser.parse_args()
    run = Path(args.run_dir)
    target = run / "VERIFICATION.json"
    if target.exists():
        raise FileExistsError(target)
    result = json.loads((run / "external_baselines_v1.json").read_text())
    checks, facts = {}, {}
    for held in args.contexts:
        pack = dict(np.load(Path(FC) / "packs" / f"{held}.npz", allow_pickle=False))
        saved = np.load(run / "scores" / f"{held}.npz", allow_pickle=False)
        eff = np.load(Path(C018) / f"effects_{held}.npz", allow_pickle=False)
        eligible, folds = eff["eligible"], eff["folds"]
        ids = pack["ids"].astype(str)
        columns = np.flatnonzero(eligible)
        # 1. VCDesign and GEARS scores
        for regime, unit_key in (("SEEN", "unit_seen"), ("MASKED", "unit_cross")):
            for v in (0, 1):
                q = pack["query_responses"][v].astype(np.float64)
                ours = _z(pack[f"base_{v}"]) + _z(_cos(q, eff[unit_key].astype(np.float64)))
                facts[f"{held}|{regime}|VCDESIGN|v{v}|max_abs"] = float(np.abs(ours - saved[f"{regime}|VCDESIGN_BASE_PLUS_RIDGE_UNIT|{v}"]).max())
        genes = np.load(Path(FC) / "genes" / "G_4.npy", allow_pickle=True).astype(str)
        def gears(name):
            s = np.load(run / "gears" / name / "effects.npz", allow_pickle=False)
            index = {g: i for i, g in enumerate(s["identities"].astype(str).tolist())}
            e = np.zeros((ids.size, genes.size))
            for i, g in enumerate(ids.tolist()):
                if g in index:
                    e[i] = s["delta"][index[g]]
            return e
        g_seen = gears("SEEN")
        g_masked = _crossfit(g_seen, [gears(f"FOLD_{f}") for f in range(5)], eligible, folds)
        for regime, e in (("SEEN", g_seen), ("MASKED", g_masked)):
            for v in (0, 1):
                ours = _cos(pack["query_responses"][v].astype(np.float64), e)
                facts[f"{held}|{regime}|GEARS|v{v}|max_abs"] = float(np.abs(ours - saved[f"{regime}|GEARS_FORWARD_ENUMERATE|{v}"]).max())
        # 2. CellNavi mapping
        navi = np.load(run / "cellnavi" / held / "scores.npz", allow_pickle=False)
        classes = navi["classes"].astype(str).tolist()
        queries = ids[pack["self_positions"]]
        for v in (0, 1):
            rows = [next(i for i, (qq, vv) in enumerate(zip(navi["queries"].astype(str), navi["views"])) if qq == q and vv == v)
                    for q in queries.tolist()]
            ours = np.full((queries.size, ids.size), np.nan)
            for j, c in enumerate(ids.tolist()):
                if c in classes:
                    ours[:, j] = navi["mean_log_prob"][rows, classes.index(c)]
            s = saved[f"SEEN|CELLNAVI_DESIGN_COMPATIBLE|{v}"]
            ok = ~np.isnan(ours)
            facts[f"{held}|CELLNAVI|v{v}|max_abs_on_covered"] = float(np.abs(ours[ok] - s[ok]).max())
            facts[f"{held}|CELLNAVI|v{v}|stratum_covered"] = bool(ok[:, columns].all())
        # 3. BU@20 medians
        selfpos = pack["self_positions"]
        for regime, arm, key in (("MASKED", "VCDESIGN_BASE_PLUS_RIDGE_UNIT", "VCDESIGN"), ("MASKED", "GEARS_FORWARD_ENUMERATE", "GEARS"),
                                 ("SEEN", "CELLNAVI_DESIGN_COMPATIBLE", "CELLNAVI")):
            rows = np.concatenate([_bu(saved[f"{regime}|{arm}|{v}"], pack[f"utility_{v}"], ids, selfpos, columns)[0] for v in (0, 1)])
            run_median = result["main_table"][regime][arm][held]["BU@20"]["median"]
            facts[f"{held}|{regime}|{key}|BU20_median_ours"] = float(np.median(rows))
            facts[f"{held}|{regime}|{key}|BU20_median_run"] = run_median
            checks[f"{held}_{regime}_{key}_BU20_regraded"] = abs(float(np.median(rows)) - run_median) < 1e-12
        checks[f"{held}_scores_reproduced"] = all(v < 1e-9 for k, v in facts.items() if k.startswith(held) and k.endswith("max_abs"))
        checks[f"{held}_cellnavi_mapping"] = all(facts[f"{held}|CELLNAVI|v{v}|max_abs_on_covered"] == 0.0 and facts[f"{held}|CELLNAVI|v{v}|stratum_covered"] for v in (0, 1))
    # 4. PHR in one context, from the quarter build
    held = args.phr_context
    pack = dict(np.load(Path(FC) / "packs" / f"{held}.npz", allow_pickle=False))
    saved = np.load(run / "scores" / f"{held}.npz", allow_pickle=False)
    eligible = np.load(Path(C018) / f"effects_{held}.npz", allow_pickle=False)["eligible"]
    build = Path(FC) / held
    genes = np.load(build / "gene_axis.npy", allow_pickle=True).astype(str)
    g4 = np.load(Path(FC) / "genes" / "G_4.npy", allow_pickle=True).astype(str)
    cols = np.asarray([list(genes).index(g) for g in g4.tolist()]) if genes.size < 50 else np.asarray(
        [dict(zip(genes.tolist(), range(genes.size)))[g] for g in g4.tolist()])
    fw = np.load(build / "four_way_eligible.npy")[pack["pool_rows"]]
    quarters = np.load(build / "quarter_responses.npy", mmap_mode="r")
    members = np.flatnonzero(fw)
    q = [np.asarray(quarters[k][pack["pool_rows"][members]][:, cols], dtype=np.float64) for k in range(4)]
    local = {int(m): i for i, m in enumerate(members.tolist())}
    qrows = np.asarray([r for r in range(pack["self_positions"].size) if fw[pack["self_positions"][r]]])
    sub = np.flatnonzero(fw & eligible)
    ids = pack["ids"].astype(str)
    selfsub = np.asarray([int(np.flatnonzero(sub == pack["self_positions"][r])[0]) if pack["self_positions"][r] in set(sub.tolist()) else -1 for r in qrows])
    phr_run = result["phr"][held]["MASKED"]
    for arm in ("BASE", "VCDESIGN_BASE_PLUS_RIDGE_UNIT"):
        values = []
        for v in (0, 1):
            j = _j(q, v)[np.ix_([local[int(pack["self_positions"][r])] for r in qrows], [local[int(c)] for c in sub])]
            s = saved[f"MASKED|{arm}|{v}"][np.ix_(qrows, sub)]
            u = pack[f"utility_{v}"][np.ix_(qrows, sub)]
            values.append(_bu(s, u, ids[sub], selfsub, np.arange(sub.size), realized=j)[1])
        ours = float(np.mean(np.concatenate(values)))
        facts[f"{held}|PHR@20|{arm}|mean_ours"] = ours
        facts[f"{held}|PHR@20|{arm}|mean_run"] = phr_run[arm]["PHR@20"]["mean"]
        checks[f"{held}_PHR20_{arm}_regraded"] = abs(ours - phr_run[arm]["PHR@20"]["mean"]) < 1e-12
    out = {"checks": {k: bool(v) for k, v in checks.items()}, "facts": facts,
           "all_pass": bool(all(checks.values())), "read_anchor_G_check": False}
    target.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    for k, v in checks.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    print(f"\nALL PASS = {out['all_pass']}")
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
