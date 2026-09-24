#!/usr/bin/env python3
"""Unified contract V1 evaluation of every arm (runs in the ``boom`` env).

Every arm is graded on the same queries, the same C-016 stratum, the same self-exclusion, the same
utilities, budgets, tie rule and query-gene cluster bootstrap.  VCDesign is read from C-018's saved
effects and gated: its reconstructed per-query MBRU must equal C-018's saved vectors exactly.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from ..decision_consistent_effect_v1 import arms as dca
from ..decision_consistent_effect_v1 import stats as st
from ..evaluation_contract_v1 import metrics as mt
from ..evidence_v1 import contract as xc
from ..four_context_v1 import evaluate as ev
from ..four_context_v1 import harness as hn
from ..model.decision_alignment_v1.contract import mbru_from_nru
from ..ppm_v1.contract_readout import _rows
from . import contract as cc
from . import phr
from . import scores as sc

SPED_RUN = "inputs/runs/sped_v1"
C018_NAMES = {cc.BASE: ("fused", "BASE"), cc.VCDESIGN: ("fused", "A1_UNIT_TARGET_COSINE"),
              cc.INTERNAL_FTM: ("effect_only", "A1_UNIT_TARGET_COSINE")}
PREDICTORS = (cc.INTERNAL_FTM, cc.GEARS, cc.RIDGE_K562)


def _unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-300)


def _labels(pack) -> np.ndarray:
    identities = np.asarray(pack.ids, dtype=str)[np.asarray(pack.self_positions)]
    return np.concatenate([identities, identities])


def _common(held: str, pack, eligible, folds) -> dict:
    """A1's common direction per pool (SPED fits), the frozen reference of the prediction measure."""
    base = Path(SPED_RUN) / "fits" / held
    common = {"SEEN": np.repeat(np.load(base / "SEEN" / "effects.npz", allow_pickle=False)["common"][None], pack.ids.size, 0)}
    common["MASKED"] = common["SEEN"].copy()
    for fold in range(cc.MASK_FOLDS):
        common["MASKED"][eligible & (folds == fold)] = np.load(base / str(fold) / "effects.npz", allow_pickle=False)["common"]
    return common


def specific_cosine(effect: np.ndarray, truth: np.ndarray, common: np.ndarray) -> np.ndarray:
    """Evidence E6's measure: cosine of predicted and measured response after removing A1's common axis."""
    spec_e = effect - np.einsum("ij,ij->i", effect, common)[:, None] * common
    spec_t = truth - np.einsum("ij,ij->i", truth, common)[:, None] * common
    return np.einsum("ij,ij->i", _unit(spec_e), _unit(spec_t))


def _summary(values: np.ndarray) -> dict:
    return {"median": float(np.median(values)), "mean": float(np.mean(values)), "rows": int(values.size)}


def _contrast(per: dict, contexts: tuple, regime: str, better: str, worse: str, metric: str = cc.PRIMARY) -> dict | None:
    if any(better not in per[h][regime] or worse not in per[h][regime] for h in contexts):
        return None
    values = [per[h][regime][better][metric] - per[h][regime][worse][metric] for h in contexts]
    labels = [per[h]["labels"] for h in contexts]
    entry = st.cluster_bootstrap_labels(np.concatenate(values), np.concatenate(labels))
    entry["qwr"] = mt.qwr(np.concatenate([per[h][regime][better][metric] for h in contexts]),
                          np.concatenate([per[h][regime][worse][metric] for h in contexts]))
    entry["by_context"] = {}
    for h, v, lab in zip(contexts, values, labels):
        e = st.cluster_bootstrap_labels(v, lab)
        e["qwr"] = mt.qwr(per[h][regime][better][metric], per[h][regime][worse][metric])
        entry["by_context"][h] = {k: e[k] for k in ("median", "mean", "median_ci95", "state", "qwr")}
    return entry


def evaluate(run: Path, packs_dir: str) -> dict:
    packs = {h: hn.load_pack(h, packs_dir) for h in cc.TRAINING_CONTEXTS}
    genes = np.load(Path(cc.FOUR_CONTEXT_RUN) / "genes" / "G_4.npy", allow_pickle=True).astype(str)
    (run / "scores").mkdir(exist_ok=True)
    per, common_cov, phr_rows, cand, gates, ident, coverage = {}, {}, {}, {}, {}, {}, {}
    for held in cc.PRIMARY_CONTEXTS:
        pack = packs[held]
        built = sc.build(run, held, pack, genes, packs)
        sc.save(run / "scores" / f"{held}.npz", built, pack)
        eligible, folds = built["eligible"], built["folds"]
        columns = np.flatnonzero(eligible)
        per[held] = {"labels": _labels(pack)}
        for regime, block in built["scores"].items():
            per[held][regime] = {arm: _rows(pack, views, columns) for arm, views in block.items()}
        # provenance: VCDesign, BASE and the internal forward arm reproduce C-018 exactly
        vectors = np.load(Path(cc.C018_RUN) / f"vectors_{held}.npz", allow_pickle=False)
        gates[held] = {}
        for regime in cc.REGIMES:
            for arm, (mode, name) in C018_NAMES.items():
                rows = per[held][regime][arm]
                mbru = np.asarray([mbru_from_nru({10: rows["BU@10"][i], 20: rows["BU@20"][i], 50: rows["BU@50"][i]})
                                   for i in range(rows["BU@20"].size)])
                gates[held][f"{regime}|{arm}"] = float(np.abs(mbru - vectors[f"{regime}|{mode}|{name}|MBRU"]).max())
        # coverage and the common-coverage stratum
        coverage[held] = {arm: {"stratum": int(columns.size), "covered_in_stratum": int(mask[columns].sum()),
                                "fraction": float(mask[columns].mean())}
                          for arm, mask in built["covered"].items()}
        shared = eligible.copy()
        for arm, mask in built["covered"].items():
            if mask.any():
                shared &= mask
        common_cov[held] = {"columns": int(shared.sum())}
        for regime, block in built["scores"].items():
            common_cov[held][regime] = {arm: _rows(pack, views, np.flatnonzero(shared)) for arm, views in block.items()}
        # candidate-level prediction quality
        common = _common(held, pack, eligible, folds)
        truth = ev.consensus(np.asarray(pack.responses)[:, columns])
        cand[held] = {regime: {arm: specific_cosine(built["effects"][regime][arm][columns], truth, common[regime][columns])
                               for arm in PREDICTORS if arm in built["effects"][regime]} for regime in cc.REGIMES}
        # PHR on the four-way sub-pool
        gate = phr.reconstruction_gate(cc.FOUR_CONTEXT_RUN, held, pack)
        phr_rows[held] = {"gate": gate}
        if gate["passes"]:
            j = phr.realized_j(cc.FOUR_CONTEXT_RUN, held, pack, eligible)
            sub, qrows = j["candidates"], j["query_rows"]
            inside = {int(c): i for i, c in enumerate(sub.tolist())}
            selfp = np.asarray([inside.get(int(p), -1) for p in np.asarray(pack.self_positions)[qrows].tolist()])
            ids = np.asarray(pack.ids, dtype=str)[sub]
            phr_rows[held].update({"candidates": int(sub.size), "queries": int(qrows.size),
                                   "labels": np.concatenate([np.asarray(pack.ids, dtype=str)[np.asarray(pack.self_positions)[qrows]]] * 2),
                                   "J_positive_fraction": float(np.mean([np.mean(m > 0) for m in j["J"]]))})
            for regime, block in built["scores"].items():
                phr_rows[held][regime] = {}
                for arm, views in block.items():
                    out = [mt.row_metrics(views[v][np.ix_(qrows, sub)], np.asarray(pack.utility[v])[np.ix_(qrows, sub)],
                                          ids, selfp, realized_j=j["J"][v]) for v in (0, 1)]
                    phr_rows[held][regime][arm] = {k: np.concatenate([out[0][k], out[1][k]]) for k in out[0]}
        for arm, key in ((cc.CELLNAVI, "cellnavi"), (cc.CELLNAVI_NATIVE, "cellnavi_native")):
            if built[key]:
                navi = built[key]
                block = {"cell_top1_accuracy_over_all_classes": navi["cell_top1_accuracy"], "cells": navi["cells"],
                         "classes": int(navi["covered"].sum())}
                for view in (0, 1):
                    info = navi["identification"][view]
                    has = info["query_is_class"]
                    block[f"view{view}"] = {"queries_that_are_classes": int(has.sum()),
                                            "row_top1": float(np.mean(info["rank_of_own_class"][has] == 1)),
                                            "median_rank_of_own_class": float(np.median(info["rank_of_own_class"][has]))}
                ident.setdefault(held, {})[arm] = block
    if not all(v == 0.0 for g in gates.values() for v in g.values()):
        raise RuntimeError(f"C-018 provenance gate failed: {gates}")

    contexts = cc.PRIMARY_CONTEXTS
    result = {"mission": cc.MISSION, "read_anchor_G_check": False, "provenance_gate_max_abs": gates,
              "coverage": coverage, "legality": cc.LEGALITY, "taxonomy": cc.TAXONOMY}
    # main table: absolute per-row metrics, per context and pooled
    table = {}
    for regime in cc.REGIMES:
        table[regime] = {}
        for arm in cc.MAIN_ROWS + (cc.RIDGE_K562, cc.PROFILED):
            if arm not in per[contexts[0]][regime]:
                table[regime][arm] = "NOT_AVAILABLE"
                continue
            entry = {}
            for scope, hs in [(h, (h,)) for h in contexts] + [("pooled", contexts)]:
                rows = {m: np.concatenate([per[h][regime][arm][m] for h in hs]) for m in per[hs[0]][regime][arm]}
                base = np.concatenate([per[h][regime][cc.BASE][cc.PRIMARY] for h in hs])
                entry[scope] = {m: _summary(rows[m]) for m in ("BU@10", "BU@20", "BU@50", "MU@20", "HvHit@20")}
                entry[scope]["QWR@20_vs_BASE"] = mt.qwr(rows[cc.PRIMARY], base)
                phr_hs = [h for h in hs if regime in phr_rows[h] and arm in phr_rows[h][regime]]
                if phr_hs:
                    entry[scope]["PHR@20_four_way_subpool"] = {
                        h: _summary(phr_rows[h][regime][arm]["PHR@20"]) for h in phr_hs}
                    powered = [h for h in phr_hs if phr_rows[h]["candidates"] >= cc.MIN_PHR_POOL]
                    if scope == "pooled" and powered:
                        entry[scope]["PHR@20_four_way_subpool"]["pooled_powered"] = _summary(
                            np.concatenate([phr_rows[h][regime][arm]["PHR@20"] for h in powered]))
            table[regime][arm] = entry
    result["main_table"] = table
    # paired headline contrasts
    pairs = {"MASKED": [(cc.VCDESIGN, cc.BASE), (cc.VCDESIGN, cc.GEARS), (cc.GEARS, cc.BASE),
                        (cc.GEARS, cc.INTERNAL_FTM), (cc.GEARS, cc.RIDGE_K562)],
             "SEEN": [(cc.VCDESIGN, cc.BASE), (cc.VCDESIGN, cc.CELLNAVI), (cc.CELLNAVI, cc.BASE),
                      (cc.VCDESIGN, cc.CELLNAVI_NATIVE), (cc.CELLNAVI_NATIVE, cc.BASE),
                      (cc.VCDESIGN, cc.GEARS), (cc.GEARS, cc.RIDGE_K562), (cc.VCDESIGN, cc.PROFILED)]}
    result["contrasts"] = {regime: {f"{b}_minus_{w}": _contrast(per, contexts, regime, b, w) for b, w in items}
                           for regime, items in pairs.items()}
    result["contrasts_companion"] = {
        regime: {f"{b}_minus_{w}|{m}": _contrast(per, contexts, regime, b, w, m)
                 for b, w in items for m in ("BU@10", "BU@50", "MU@20", "HvHit@20")}
        for regime, items in pairs.items()}
    # common coverage
    result["common_coverage"] = {"columns": {h: common_cov[h]["columns"] for h in contexts}}
    for regime in cc.REGIMES:
        block = {}
        for arm in per[contexts[0]][regime]:
            rows = np.concatenate([common_cov[h][regime][arm][cc.PRIMARY] for h in contexts])
            block[arm] = _summary(rows)
        cc_per = {h: {"labels": per[h]["labels"], regime: common_cov[h][regime]} for h in contexts}
        block["contrasts"] = {f"{b}_minus_{w}": _contrast(cc_per, contexts, regime, b, w) for b, w in pairs[regime]}
        result["common_coverage"][regime] = block
    # open-vocabulary retention (same stratum, absolute BU@20 of the arm, and VCDesign's fused gain)
    retention = {}
    for arm in (cc.INTERNAL_FTM, cc.GEARS, cc.RIDGE_K562):
        if arm not in per[contexts[0]]["MASKED"]:
            continue
        retention[arm] = {}
        for h in contexts:
            seen, masked = per[h]["SEEN"][arm][cc.PRIMARY], per[h]["MASKED"][arm][cc.PRIMARY]
            diff = st.cluster_bootstrap_labels(masked - seen, per[h]["labels"])
            retention[arm][h] = {"SEEN_median": float(np.median(seen)), "MASKED_median": float(np.median(masked)),
                                 "MASKED_over_SEEN_median": float(np.median(masked) / np.median(seen)) if np.median(seen) != 0 else None,
                                 "MASKED_minus_SEEN": {k: diff[k] for k in ("median", "mean", "median_ci95", "state")}}
    retention[cc.VCDESIGN] = {}
    for h in contexts:
        gain = {r: per[h][r][cc.VCDESIGN][cc.PRIMARY] - per[h][r][cc.BASE][cc.PRIMARY] for r in cc.REGIMES}
        retention[cc.VCDESIGN][h] = {"SEEN_gain_median": float(np.median(gain["SEEN"])),
                                     "MASKED_gain_median": float(np.median(gain["MASKED"])),
                                     "ratio": float(np.median(gain["MASKED"]) / np.median(gain["SEEN"]))}
    result["open_vocabulary_retention"] = retention
    # PHR
    result["phr"] = {}
    for h in contexts:
        block = {"gate": phr_rows[h]["gate"]}
        if "candidates" in phr_rows[h]:
            block.update({k: phr_rows[h][k] for k in ("candidates", "queries", "J_positive_fraction")})
            block["powered"] = phr_rows[h]["candidates"] >= cc.MIN_PHR_POOL
            for regime in cc.REGIMES:
                block[regime] = {arm: {f"PHR@{b}": _summary(rows[f"PHR@{b}"]) for b in cc.BUDGETS}
                                 for arm, rows in phr_rows[h][regime].items()}
                if cc.BASE in phr_rows[h][regime]:
                    for arm in (cc.VCDESIGN, cc.GEARS, cc.CELLNAVI, cc.ORACLE):
                        if arm in phr_rows[h][regime]:
                            d = st.cluster_bootstrap_labels(phr_rows[h][regime][arm]["PHR@20"] - phr_rows[h][regime][cc.BASE]["PHR@20"],
                                                            phr_rows[h]["labels"])
                            block[regime][arm]["PHR@20_minus_BASE"] = {k: d[k] for k in ("median", "mean", "median_ci95", "mean_ci95", "state")}
        else:
            block["status"] = cc.PHR_UNAVAILABLE
        result["phr"][h] = block
    powered = [h for h in contexts if result["phr"][h].get("powered")]
    result["phr"]["pooled_powered_contexts"] = powered
    for regime in cc.REGIMES:
        for arm in (cc.VCDESIGN, cc.GEARS, cc.ORACLE):
            if powered and all(arm in phr_rows[h].get(regime, {}) for h in powered):
                d = st.cluster_bootstrap_labels(
                    np.concatenate([phr_rows[h][regime][arm]["PHR@20"] - phr_rows[h][regime][cc.BASE]["PHR@20"] for h in powered]),
                    np.concatenate([phr_rows[h]["labels"] for h in powered]))
                result["phr"][f"pooled|{regime}|{arm}_minus_BASE|PHR@20"] = {k: d[k] for k in ("median", "mean", "median_ci95", "mean_ci95", "state")}
    result["cellnavi_native_identification_companion"] = ident or "NOT_AVAILABLE"
    result["prediction_vs_design"] = prediction_vs_design(packs, per, cand)
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    return result


def prediction_vs_design(packs: dict, per: dict, cand: dict) -> dict:
    """Held-context prediction quality against design, across regimes/atlases and within estimator pairs."""
    points = []
    for h in cc.PRIMARY_CONTEXTS:
        pack = packs[h]
        c018 = np.load(Path(cc.C018_RUN) / f"effects_{h}.npz", allow_pickle=False)
        eligible, folds = c018["eligible"], c018["folds"]
        columns = np.flatnonzero(eligible)
        common = _common(h, pack, eligible, folds)
        truth = ev.consensus(np.asarray(pack.responses)[:, columns])
        base = Path(cc.EVIDENCE_RUN) / "fits" / h
        seen = np.load(base / "SEEN" / "effects.npz", allow_pickle=False)
        parts = [np.load(base / str(f) / "effects.npz", allow_pickle=False) for f in range(cc.MASK_FOLDS)]
        for spec in xc.specs(h):
            name = spec["name"]
            masked = seen[name].astype(np.float64).copy()
            for part in parts:
                masked[part["rows"]] = part[name]
            for regime, effect in (("SEEN", seen[name].astype(np.float64)), ("MASKED", masked)):
                cosine = dca.cosine_scores(pack, effect)
                design = _rows(pack, cosine, columns)[cc.PRIMARY]
                pred = specific_cosine(effect[columns], truth, common[regime][columns])
                points.append({"held": h, "regime": regime, "estimator": f"RIDGE_UNIT[{name}]", "family": "ridge",
                               "atlas": "+".join(spec["contexts"]) + ("@matched" if spec["matched"] else ""),
                               "prediction": float(np.median(pred)), "design_effect_only_BU@20": float(np.median(design))})
        for regime in cc.REGIMES:
            for arm, family in ((cc.GEARS, "gears"), (cc.RIDGE_K562, "ridge_fold_union")):
                if arm in cand[h][regime]:
                    points.append({"held": h, "regime": regime, "estimator": arm, "family": family,
                                   "atlas": "K562 (fold-union masks)",
                                   "prediction": float(np.median(cand[h][regime][arm])),
                                   "design_effect_only_BU@20": float(np.median(per[h][regime][arm][cc.PRIMARY]))})
    out = {"points": points}
    for regime in cc.REGIMES:
        # primary: the evidence ridge grid alone; companion: plus GEARS, whose points are degenerate
        for label, families in (("across", ("ridge",)), ("across_with_gears", ("ridge", "gears"))):
            sel = [p for p in points if p["regime"] == regime and p["family"] in families]
            rho = spearmanr([p["prediction"] for p in sel], [p["design_effect_only_BU@20"] for p in sel])
            out[f"{label}|{regime}"] = {"spearman": float(rho.correlation), "p_value": float(rho.pvalue), "n": len(sel)}
    # within comparable estimators: same atlas, different estimator or single-input change
    within = []
    def find(h, regime, estimator):
        return next(p for p in points if p["held"] == h and p["regime"] == regime and p["estimator"] == estimator)
    for h in cc.PRIMARY_CONTEXTS:
        for regime in cc.REGIMES:
            ref = find(h, regime, f"RIDGE_UNIT[{xc.FULL}]")
            for name in ("rank64", "rank128", "rank512", "STRING_only", "MAPKG_only"):
                p = find(h, regime, f"RIDGE_UNIT[{name}]")
                within.append({"held": h, "regime": regime, "pair": f"{name}_vs_full", "kind": "same atlas, one input changed",
                               "d_prediction": p["prediction"] - ref["prediction"],
                               "d_design": p["design_effect_only_BU@20"] - ref["design_effect_only_BU@20"]})
            if any(p["estimator"] == cc.GEARS and p["held"] == h and p["regime"] == regime for p in points):
                g, r = find(h, regime, cc.GEARS), find(h, regime, cc.RIDGE_K562)
                within.append({"held": h, "regime": regime, "pair": "GEARS_vs_RIDGE_UNIT_same_K562_atlas",
                               "kind": "same atlas, different estimator",
                               "d_prediction": g["prediction"] - r["prediction"],
                               "d_design": g["design_effect_only_BU@20"] - r["design_effect_only_BU@20"]})
    for w in within:
        w["same_sign"] = bool(np.sign(w["d_prediction"]) == np.sign(w["d_design"]))
    out["within"] = within
    for kind in sorted({w["kind"] for w in within}):
        for regime in cc.REGIMES:
            sel = [w for w in within if w["kind"] == kind and w["regime"] == regime]
            out[f"within_same_sign|{kind}|{regime}"] = {"agree": int(sum(w["same_sign"] for w in sel)), "pairs": len(sel)}
    out["within_sped_reference"] = ("SPED D5 (C-023, same atlas, estimator changed): M2 candidate-specific "
                                    "+0.0084 with fused BU@20 -0.0050; M3 -0.0059 with +0.0009; M4 +0.0058 with +0.0013")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified contract V1 evaluation")
    parser.add_argument("--run-dir", dest="run_dir", required=True)
    parser.add_argument("--packs", default=str(Path(cc.FOUR_CONTEXT_RUN) / "packs"))
    args = parser.parse_args()
    run = Path(args.run_dir)
    target = run / "external_baselines_v1.json"
    if target.exists():
        raise FileExistsError(f"{target} exists")
    result = evaluate(run, args.packs)
    target.write_text(json.dumps(result, indent=2, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
    print(json.dumps(result["contrasts"], indent=1, default=str)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
