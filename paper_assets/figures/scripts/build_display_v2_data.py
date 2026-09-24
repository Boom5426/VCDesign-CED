#!/usr/bin/env python3
"""Build the plotting artifact for the main-text displays (Table 1, Figures 2 to 4, display v2).

Nothing here refits or rescores a model. Every value is either copied from a frozen artifact or
recomputed from the row-level source data that the frozen artifacts already publish:

* Figure 2b, 3a, 4: medians and query-gene cluster intervals of paired BU@20 differences,
  recomputed from ``fig3_source_data.csv`` with the programme's bootstrap (10,000 replicates,
  seed 20260916, clusters = query gene; pooled rows keep a gene's views and contexts together).
* Figure 2d: paired high-value-hit differences on the K562 held-out split, from
  ``fig2_source_data.csv``, resampling query-view rows as the locked K562 analysis does.
* Figure 4c: per-context Spearman coefficients and a context-stratified permutation test on the
  deduplicated prediction-design grid of ``fig4_diagnosis_data.json``.

Before anything is written, the script reproduces registered values (the Figure 3 intervals, the
K562 medians, the external contrasts) and stops if any of them differs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

REPLICATES = 10_000
SEED = 20260916
PERMUTATIONS = 20_000
CONTEXTS = ("RPE1", "HepG2", "Jurkat")
BUDGETS = (10, 20, 50)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------------
# Bootstrap: a line-for-line copy of decision_consistent_effect_v1.stats.cluster_bootstrap_labels
# (gene_open_inverse/), so that recomputed intervals use exactly the registered procedure.
# --------------------------------------------------------------------------------------------
def cluster_bootstrap_labels(values, labels, replicates=REPLICATES, seed=SEED, chunk=200):
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=str)
    unique, inverse = np.unique(labels, return_inverse=True)
    sizes = np.bincount(inverse)
    width = int(sizes.max())
    padded = np.full((unique.size, width), np.nan)
    fill = np.zeros(unique.size, dtype=np.int64)
    for index, cluster in enumerate(inverse.tolist()):
        padded[cluster, fill[cluster]] = values[index]
        fill[cluster] += 1
    draws = np.random.default_rng(seed).integers(0, unique.size, size=(replicates, unique.size))
    medians, means = [], []
    for start in range(0, replicates, chunk):
        sample = padded[draws[start:start + chunk]].reshape(min(chunk, replicates - start), -1)
        medians.append(np.nanmedian(sample, axis=1))
        means.append(np.nanmean(sample, axis=1))
    medians, means = np.concatenate(medians), np.concatenate(means)
    return {"clusters": int(unique.size), "entries": int(values.size),
            "replicates": int(replicates), "seed": int(seed),
            "median": float(np.median(values)), "mean": float(values.mean()),
            "median_ci95": [float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))],
            "mean_ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
            "win_rate": float((values > 0).mean())}


def row_bootstrap(values, replicates=REPLICATES, seed=SEED):
    """Percentile bootstrap over rows (the K562 held-out analysis unit)."""
    values = np.asarray(values, dtype=np.float64)
    draws = np.random.default_rng(seed).integers(0, values.size, size=(replicates, values.size))
    sample = values[draws]
    med, mean = np.median(sample, axis=1), sample.mean(axis=1)
    return {"rows": int(values.size), "replicates": replicates, "seed": seed,
            "median": float(np.median(values)), "mean": float(values.mean()),
            "median_ci95": [float(np.quantile(med, 0.025)), float(np.quantile(med, 0.975))],
            "mean_ci95": [float(np.quantile(mean, 0.025)), float(np.quantile(mean, 0.975))],
            "win": float((values > 0).mean()), "tie": float((values == 0).mean()),
            "loss": float((values < 0).mean())}


def close(a, b, tol, what):
    if abs(float(a) - float(b)) > tol:
        raise RuntimeError(f"reproduction failed for {what}: {a} vs {b}")


# --------------------------------------------------------------------------------------------
def external_levels(path: Path) -> dict:
    out = defaultdict(dict)
    with path.open() as fh:
        for row in csv.DictReader(fh):
            out[row["regime"]][row["method_id"]] = {
                "label": row["method"], "RPE1": float(row["rpe1"]), "HepG2": float(row["hepg2"]),
                "Jurkat": float(row["jurkat"]), "pooled": float(row["pooled"])}
    return out


def external_contrasts(path: Path) -> dict:
    out = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            out[(row["regime"], row["contrast_id"])] = row
    return out


def fig3_rows(path: Path) -> dict:
    """{state: {context: {(query, view): value}}} for the Figure 3 paired rows."""
    rows = defaultdict(lambda: defaultdict(dict))
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if row["panel"] == "b" and row["record_type"] == "source_row":
                key = (row["query"], row["view"])
                rows[row["regime_or_state"]][row["context"]][key] = float(row["value"])
    return rows


def paired_block(rows, contexts, a_state, b_state=None):
    """Values (a - b, or a alone) and gene labels over the given contexts, aligned by row."""
    values, labels = [], []
    for ctx in contexts:
        keys = sorted(rows[a_state][ctx])
        if b_state is not None and set(keys) != set(rows[b_state][ctx]):
            raise RuntimeError(f"{a_state} and {b_state} rows differ in {ctx}")
        for key in keys:
            v = rows[a_state][ctx][key]
            if b_state is not None:
                v -= rows[b_state][ctx][key]
            values.append(v)
            labels.append(key[0])
    return np.asarray(values), np.asarray(labels)


def summarise(block):
    return {"median": block["median"], "ci95": block["median_ci95"], "mean": block["mean"],
            "mean_ci95": block["mean_ci95"], "win_rate": block["win_rate"],
            "clusters": block["clusters"], "rows": block["entries"]}


# --------------------------------------------------------------------------------------------
def build(args) -> tuple[dict, list]:
    csv_rows: list = []
    levels = external_levels(args.external_levels)
    contrasts = external_contrasts(args.external_contrasts)
    fig3 = json.loads(args.fig3.read_text())
    rows3 = fig3_rows(args.fig3_rows)

    # ---- Reproduce the registered Figure 3 intervals before computing anything new ----------
    for ctx in CONTEXTS + ("pooled",):
        ctxs = CONTEXTS if ctx == "pooled" else (ctx,)
        for state in ("full_measured", "size_controlled", "response_unseen"):
            vals, labs = paired_block(rows3, ctxs, state)
            got = cluster_bootstrap_labels(vals, labs)
            ref = (fig3["panel_b"]["pooled"][state] if ctx == "pooled"
                   else fig3["panel_b"]["contexts"][ctx]["states"][state])
            close(got["median"], ref["median"], 1e-12, f"{ctx} {state} median")
            close(got["median_ci95"][0], ref["ci95"][0], 1e-12, f"{ctx} {state} ci low")
            close(got["median_ci95"][1], ref["ci95"][1], 1e-12, f"{ctx} {state} ci high")
            if got["clusters"] != ref["clusters"]:
                raise RuntimeError(f"{ctx} {state} cluster count changed")
    # The full-atlas arm is the external SEEN contrast and the masked arm the MASKED contrast.
    for regime, state in (("SEEN", "full_measured"), ("MASKED", "response_unseen")):
        ref = contrasts[(regime, "vcdesign_minus_base")]
        for ctx, col in zip(CONTEXTS, ("rpe1", "hepg2", "jurkat")):
            close(fig3["panel_b"]["contexts"][ctx]["states"][state]["median"], ref[col], 6e-5,
                  f"{regime} {ctx} external contrast")
        close(fig3["panel_b"]["pooled"][state]["median"], ref["pooled"], 6e-5, f"{regime} pooled")

    data: dict = {}

    # ---- Figure 2a: per-context levels in both information regimes --------------------------
    method_ids = {
        "goal_blind_prior": ("Goal-blind prior", ("MASKED", "SEEN")),
        "internal_forward_then_match": ("CED effect only", ("MASKED", "SEEN")),
        "base": ("Base scorer", ("MASKED",)),
        "vcdesign_ced": ("VCDesign-CED", ("MASKED", "SEEN")),
        "cellnavi_official": ("CellNavi, official classes", ("SEEN",)),
        "cellnavi_native": ("CellNavi, candidate classes", ("SEEN",)),
        "profiled_signature_retrieval": ("Profile retrieval", ("SEEN",)),
    }
    a_levels = {}
    for mid, (label, regimes) in method_ids.items():
        a_levels[mid] = {"label": label, "levels": {r: levels[r][mid] for r in regimes}}
        for r in regimes:
            for ctx in CONTEXTS + ("pooled",):
                csv_rows.append(["2", "a", mid, r, ctx, "", levels[r][mid][ctx], "", "",
                                 "median BU@20 over query-view rows", "sec:heldout"])
    if levels["MASKED"]["base"] != {**levels["SEEN"]["base"], "label": levels["MASKED"]["base"]["label"]}:
        raise RuntimeError("the base scorer should not depend on the atlas regime")
    data["fig2_a"] = {"metric": "BU@20", "statistic": "median over query-view rows (point estimate)",
                      "methods": a_levels,
                      "random": {r: levels[r]["random"] for r in ("MASKED", "SEEN")},
                      "oracle": levels["SEEN"]["measured_response_oracle"]}

    # ---- Figure 2b: paired gain over the base scorer, both regimes ---------------------------
    b = {}
    for regime, state in (("MASKED", "response_unseen"), ("SEEN", "full_measured")):
        b[regime] = {}
        for ctx in CONTEXTS + ("pooled",):
            ctxs = CONTEXTS if ctx == "pooled" else (ctx,)
            vals, labs = paired_block(rows3, ctxs, state)
            b[regime][ctx] = summarise(cluster_bootstrap_labels(vals, labs))
            s = b[regime][ctx]
            csv_rows.append(["2", "b", "VCDesign-CED minus Base", regime, ctx, "", s["median"],
                             s["ci95"][0], s["ci95"][1],
                             "median paired BU@20 difference, query-gene cluster bootstrap",
                             "sec:heldout"])
    data["fig2_b"] = {"metric": "paired BU@20 difference, VCDesign-CED minus Base",
                      "bootstrap": {"replicates": REPLICATES, "seed": SEED,
                                    "unit": "query gene (views and contexts together)"},
                      "regimes": b}

    # ---- Figure 2c/2d: K562 held-out identities ----------------------------------------------
    fig2 = json.loads(args.fig2.read_text())
    k562 = {name: fig2["panel_a"]["methods"][name] for name in fig2["panel_a"]["methods"]}
    paired = fig2["panel_b"]["comparisons"]
    with args.fig2_rows.open() as fh:
        krows = list(csv.DictReader(fh))
    if len(krows) != 684:
        raise RuntimeError("K562 row count changed")
    ced = np.array([float(r["vcdesign_ced_mbu"]) for r in krows])
    close(np.median(ced), k562["VCDesign-CED"]["summary"]["median"], 1e-12, "K562 CED median")
    for comp, col in (("Base scorer", "ced_minus_base"), ("Text-augmented ranker", "ced_minus_text")):
        v = np.array([float(r[col]) for r in krows])
        close(np.median(v), paired[comp]["summary"]["median"], 1e-12, f"K562 {comp} paired median")
        close((v > 0).mean(), paired[comp]["win_rate"], 1e-12, f"K562 {comp} win rate")
    # Paired contrasts with the locked plan's statistical unit: a paired identity-cluster bootstrap
    # (both views of a query resampled together; 10,000 replicates, seed 20260916). The locked
    # record's intervals resampled query-view rows; they are kept alongside for reference.
    kq = np.array([r["query"] for r in krows])
    kcols = {"Goal-blind prior": "goal_blind_mbu", "Forward-then-match": "forward_match_mbu",
             "Base scorer": "base_mbu", "Text-augmented ranker": "text_mbu",
             "VCDesign-CED": "vcdesign_ced_mbu"}
    kv = {name: np.array([float(r[c]) for r in krows]) for name, c in kcols.items()}
    k562_contrasts = {}
    for a_name, b_name in (("VCDesign-CED", "Base scorer"), ("VCDesign-CED", "Text-augmented ranker"),
                           ("VCDesign-CED", "Goal-blind prior"), ("Base scorer", "Goal-blind prior"),
                           ("Base scorer", "Text-augmented ranker"),
                           ("Forward-then-match", "Goal-blind prior")):
        cb = cluster_bootstrap_labels(kv[a_name] - kv[b_name], kq)
        k562_contrasts[f"{a_name} minus {b_name}"] = {
            "median": cb["median"], "median_ci95": cb["median_ci95"], "mean": cb["mean"],
            "mean_ci95": cb["mean_ci95"], "win_rate": cb["win_rate"], "clusters": cb["clusters"]}
    for comp in ("Base scorer", "Text-augmented ranker"):
        close(k562_contrasts[f"VCDesign-CED minus {comp}"]["median"], paired[comp]["summary"]["median"],
              1e-12, f"K562 cluster median {comp}")
    data["fig2_c"] = {
        "k562_contrasts_identity_cluster": k562_contrasts,
        "metric": "mean BU (average of BU@10, BU@20, BU@50) per query-view row",
        "levels": {name: {"values": m["values"], "summary": m["summary"]} for name, m in k562.items()},
        "paired": {name: {"values": p["values"], "median": p["summary"]["median"],
                          "ci95": p["ci95"], "win_rate": p["win_rate"]}
                   for name, p in paired.items()},
        "ci_unit": "query-view row (locked K562 analysis)"}
    for name, p in paired.items():
        csv_rows.append(["2", "c", f"VCDesign-CED minus {name}", "K562 held-out", "K562", "",
                         p["summary"]["median"], p["ci95"][0], p["ci95"][1],
                         "median paired mean-BU difference, row bootstrap (locked record)",
                         "sec:heldout"])

    hits = {}
    medians_expected = {"Base scorer": [4, 6, 9], "Text-augmented ranker": [4, 7, 10],
                        "VCDesign-CED": [4, 7, 12]}
    cols = {"Base scorer": "base", "Text-augmented ranker": "text", "VCDesign-CED": "vcdesign_ced"}
    arr = {name: {B: np.array([float(r[f"{c}_hvhit_{B}"]) for r in krows]) for B in BUDGETS}
           for name, c in cols.items()}
    queries = np.array([r["query"] for r in krows])
    for name in cols:
        got = [float(np.median(arr[name][B])) for B in BUDGETS]
        if got != [float(x) for x in medians_expected[name]]:
            raise RuntimeError(f"K562 hit medians changed for {name}: {got}")
    for comp in ("Base scorer", "Text-augmented ranker"):
        hits[comp] = {}
        for B in BUDGETS:
            diff = arr["VCDesign-CED"][B] - arr[comp][B]
            rb = row_bootstrap(diff)
            cb = cluster_bootstrap_labels(diff, queries)
            hits[comp][str(B)] = {"mean": cb["mean"], "mean_ci95": cb["mean_ci95"],
                                  "median": rb["median"], "median_ci95_row": rb["median_ci95"],
                                  "win": rb["win"], "tie": rb["tie"], "loss": rb["loss"],
                                  "mean_ci95_row": rb["mean_ci95"]}
            csv_rows.append(["2", "d", f"VCDesign-CED minus {comp}", "K562 held-out", "K562", B,
                             cb["mean"], cb["mean_ci95"][0], cb["mean_ci95"][1],
                             "mean paired high-value-hit difference, identity-cluster bootstrap",
                             "sec:heldout"])
    data["fig2_d"] = {
        "statistic": "mean over query-view rows of the paired difference in high-value hits "
                     "(top-B candidates at or above the query's 95th utility percentile)",
        "ci_unit": "query identity cluster (both views), 10,000 replicates, seed 20260916",
        "budgets": list(BUDGETS),
        "means": {name: [float(arr[name][B].mean()) for B in BUDGETS] for name in cols},
        "medians": medians_expected,
        "random_expectation": [0.05 * B for B in BUDGETS],
        "paired": hits}

    # ---- Figure 3a: controlled masking decomposition -----------------------------------------
    decomp = {"atlas_size": ("full_measured", "size_controlled"),
              "own_response": ("size_controlled", "response_unseen")}
    d3 = {}
    for key, (a_state, b_state) in decomp.items():
        d3[key] = {}
        for ctx in CONTEXTS + ("pooled",):
            ctxs = CONTEXTS if ctx == "pooled" else (ctx,)
            vals, labs = paired_block(rows3, ctxs, a_state, b_state)
            d3[key][ctx] = summarise(cluster_bootstrap_labels(vals, labs))
            s = d3[key][ctx]
            csv_rows.append(["3", "a", f"{a_state} minus {b_state}", "paired arms", ctx, "",
                             s["median"], s["ci95"][0], s["ci95"][1],
                             "median paired BU@20 difference, query-gene cluster bootstrap",
                             "sec:context"])
    data["fig3_a"] = {"metric": "paired BU@20 difference between masking arms",
                      "contrasts": {"atlas_size": "full atlas minus size-matched atlas "
                                                  "(both keep the candidate's own responses)",
                                    "own_response": "size-matched minus own-response-masked "
                                                    "(equal atlas size)"},
                      "bootstrap": {"replicates": REPLICATES, "seed": SEED,
                                    "unit": "query gene (views and contexts together)"},
                      "values": d3,
                      "masked_gain": {ctx: b["MASKED"][ctx]["median"] for ctx in CONTEXTS + ("pooled",)}}

    # ---- Figure 3a: solver regime (pooled levels and the two registered paired contrasts) --------
    hc = fig3["panel_a"]["headline_contrasts"]
    retr = hc["SEEN|Profile_retrieval_minus_VCDesign-CED"]
    close(-retr["median"], float(contrasts[("SEEN", "vcdesign_minus_profiled_retrieval")]["pooled"]), 6e-5,
          "retrieval minus CED")
    close(hc["MASKED|VCDesign-CED_minus_Base"]["median"], b["MASKED"]["pooled"]["median"], 1e-12,
          "masked CED minus Base")
    lv = fig3["panel_a"]["methods"]
    data["fig3_regime"] = {
        "metric": "pooled median BU@20 over query-view rows; contrasts are paired medians with "
                  "query-gene cluster intervals",
        "levels": {"Base": lv["BASE"]["levels"], "VCDesign-CED": lv["VCDESIGN_BASE_PLUS_RIDGE_UNIT"]["levels"],
                   "Profile retrieval": lv["PROFILED_SIGNATURE_RETRIEVAL"]["levels"]},
        "contrasts": {"masked: VCDesign-CED minus Base": hc["MASKED|VCDesign-CED_minus_Base"],
                      "measured: Profile retrieval minus VCDesign-CED": retr}}

    # ---- Figure 3b: mixed libraries -------------------------------------------------------------
    data["fig3_b"] = {"metric": "median paired BU@20 gain over Base (point estimate)",
                      "contexts": fig3["panel_c"]["contexts"]}
    for ctx, block in fig3["panel_c"]["contexts"].items():
        for p in block["points"]:
            csv_rows.append(["3", "b", "VCDesign-CED minus Base", "mixed", ctx, p["share"],
                             p["median"], "", "", "median paired BU@20 difference", "sec:context"])

    # ---- Figure 4a: response-unseen atlas growth ------------------------------------------------
    atlas = json.loads(args.atlas.read_text())
    data["fig4_a"] = {"statistic": atlas["panel"]["statistic"], "ci": atlas["panel"]["ci_source"],
                      "arms": atlas["panel"]["arms"]}
    close(atlas["panel"]["arms"]["natural"][2]["median"], b["MASKED"]["pooled"]["median"], 1e-12,
          "three-context natural atlas equals the headline masked gain")
    for arm, pts in atlas["panel"]["arms"].items():
        for p in pts:
            csv_rows.append(["4", "a", arm, "MASKED", "pooled", p["contexts"], p["median"],
                             p["ci95"][0], p["ci95"][1],
                             f"median paired BU@20 gain over Base; atlas rows {p['atlas_rows']:.0f}",
                             "sec:analysis"])

    # ---- Figure 4b: capacity and information interventions -------------------------------------
    fig4 = json.loads(args.fig4.read_text())
    data["fig4_b"] = {"statistic": fig4["panel_c"]["statistic"],
                      "variants": fig4["panel_c"]["variants"]}
    for v in fig4["panel_c"]["variants"]:
        csv_rows.append(["4", "b", v["label"], "MASKED", "pooled", "", v["median"], v["ci95"][0],
                         v["ci95"][1], "median paired BU@20 difference vs unit-direction ridge",
                         "sec:analysis"])

    # ---- Figure 4c: prediction versus design ----------------------------------------------------
    pdd = fig4["panel_d"]
    grid = pdd["grid_points"]
    close(spearmanr([g["prediction"] for g in grid],
                    [g["design_effect_only_BU@20"] for g in grid]).statistic,
          pdd["spearman"], 1e-9, "global Spearman")
    unique, dup = [], []
    seen = set()
    for g in grid:
        key = (g["held"], round(g["prediction"], 12), round(g["design_effect_only_BU@20"], 12))
        (dup if key in seen else unique).append(g)
        seen.add(key)
    per_ctx = {}
    for ctx in CONTEXTS:
        pts = [g for g in unique if g["held"] == ctx]
        rho = spearmanr([g["prediction"] for g in pts],
                        [g["design_effect_only_BU@20"] for g in pts]).statistic
        per_ctx[ctx] = {"rho": float(rho), "n": len(pts)}
    # Stratified permutation: shuffle design values within each held context; the statistic is
    # the mean within-context Spearman coefficient.
    rng = np.random.default_rng(SEED)
    by_ctx = {ctx: (np.array([g["prediction"] for g in unique if g["held"] == ctx]),
                    np.array([g["design_effect_only_BU@20"] for g in unique if g["held"] == ctx]))
              for ctx in CONTEXTS}
    ranks = {ctx: (rankdata(x), rankdata(y)) for ctx, (x, y) in by_ctx.items()}

    def mean_rho(perms=None):
        out = []
        for i, ctx in enumerate(CONTEXTS):
            rx, ry = ranks[ctx]
            if perms is not None:
                ry = ry[perms[i]]
            out.append(np.corrcoef(rx, ry)[0, 1])
        return float(np.mean(out))

    observed = mean_rho()
    exceed = 0
    for _ in range(PERMUTATIONS):
        perms = [rng.permutation(len(ranks[ctx][1])) for ctx in CONTEXTS]
        exceed += mean_rho(perms) >= observed
    p_perm = (1 + exceed) / (1 + PERMUTATIONS)
    variants = defaultdict(dict)
    names = {"rank64_vs_full": "rank 64", "rank128_vs_full": "rank 128", "rank512_vs_full": "rank 512",
             "STRING_only_vs_full": "STRING only", "MAPKG_only_vs_full": "MAP-KG only"}
    for c in pdd["controlled_changes"]:
        variants[names[c["pair"]]][c["held"]] = {"d_prediction": c["d_prediction"],
                                                 "d_design": c["d_design"], "same_sign": c["same_sign"]}
    opposite = [v for v, d in variants.items() if all(not x["same_sign"] for x in d.values())]
    data["fig4_c"] = {
        "prediction_metric": pdd["prediction_metric"], "design_metric": pdd["design_metric"],
        "grid_points": grid, "duplicates_removed": [
            {"held": g["held"], "estimator": g["estimator"]} for g in dup],
        "n_unique": len(unique), "per_context": per_ctx,
        "mean_within_context_rho": observed, "stratified_permutation_p": p_perm,
        "permutations": PERMUTATIONS, "seed": SEED,
        "registered_global": {"spearman": pdd["spearman"], "p_value": pdd["p_value"], "n": pdd["n_grid"]},
        "controlled_changes": variants,
        "variants_opposite_in_every_context": opposite}
    dup_keys = {(g["held"], g["estimator"]) for g in dup}
    for g in grid:
        flag = ("; duplicate fit, excluded from the within-context test"
                if (g["held"], g["estimator"]) in dup_keys else "")
        csv_rows.append(["4", "c-left", g["estimator"], "MASKED", g["held"], g["prediction"],
                         g["design_effect_only_BU@20"], "", "",
                         "candidate-specific prediction cosine (x) vs effect-only BU@20 (y)" + flag,
                         "sec:prediction-design"])
    for v, d in variants.items():
        for ctx, x in d.items():
            csv_rows.append(["4", "c-right", v, "MASKED", ctx, x["d_prediction"], x["d_design"], "",
                             "", "within-atlas change: d prediction (x), d design (y)",
                             "sec:prediction-design"])

    data["provenance"] = {
        "generated_by": "figures/scripts/build_display_v2_data.py",
        "inputs": {str(p): sha256(p) for p in (args.external_levels, args.external_contrasts,
                                               args.fig2, args.fig2_rows, args.fig3,
                                               args.fig3_rows, args.fig4, args.atlas)},
        "reproduced_before_build": [
            "Figure 3 per-context and pooled medians and 95% intervals of all three masking arms",
            "external SEEN and MASKED VCDesign-CED minus Base contrasts",
            "K562 CED median, paired medians and win rates, high-value-hit medians",
            "global Spearman coefficient of the 57-point grid",
            "three-context natural atlas equals the headline masked gain"]}
    return data, csv_rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    d = Path("figures/data")
    s = Path("supplementary/data")
    p.add_argument("--external-levels", type=Path, default=s / "external_baseline_levels.csv")
    p.add_argument("--external-contrasts", type=Path, default=s / "external_baseline_contrasts.csv")
    p.add_argument("--fig2", type=Path, default=d / "fig2_hero_data.json")
    p.add_argument("--fig2-rows", type=Path, default=d / "fig2_source_data.csv")
    p.add_argument("--fig3", type=Path, default=d / "fig3_generalization_data.json")
    p.add_argument("--fig3-rows", type=Path, default=d / "fig3_source_data.csv")
    p.add_argument("--fig4", type=Path, default=d / "fig4_diagnosis_data.json")
    p.add_argument("--atlas", type=Path, default=d / "figS_atlas_context_data.json")
    p.add_argument("--output", type=Path, default=d / "display_v2_data.json")
    p.add_argument("--source-csv", type=Path, default=d / "display_v2_source_data.csv")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    for out in (args.output, args.source_csv):
        if out.exists() and not args.overwrite:
            raise SystemExit(f"{out} exists; pass --overwrite to replace it")
    data, rows = build(args)
    args.output.write_text(json.dumps(data, indent=1) + "\n")
    with args.source_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["figure", "panel", "series", "regime", "context", "x", "value", "ci95_low",
                    "ci95_high", "statistic", "owning_section"])
        w.writerows(rows)
    print(f"wrote {args.output} and {args.source_csv} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
