#!/usr/bin/env python3
"""Phase A: explain the natural unseen-candidate failure without training anything new.

The K562 fold is the only one whose unseen pool is not degenerate: 6255 candidates
against a largest budget of 50, where the other three folds hold 25, 73 and 105.  So
this phase runs there and nowhere else, and it trains no model that did not already
exist.  It refits the same three-context pooled V1 predictor the leave-one-context-out
phase fitted, under the same recipe and the same seeds, and the first thing it reports
is whether that refit reproduces the number it is trying to explain.

Two questions are answered here and neither is decisive on its own.  Does the deficit
track static support, and does it survive matching seen candidates to unseen ones on
that support?  Both are observational: nobody assigned a candidate to the unseen set,
and library membership stays confounded with everything.  The controlled masking phase
exists because of that.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..four_context_v1 import contract as fc
from ..four_context_v1 import evaluate as ev
from ..four_context_v1 import harness as hn
from ..four_context_v1 import predictors as pd
from . import contract as ov
from . import support as sp


def paired_all(vectors: dict, arm: str, reference: str) -> dict:
    """Every metric's paired difference between two arms on one stratum."""
    return {key: ev.paired(vectors[arm][key], vectors[reference][key])
            for key in vectors[reference]}


def quantile_strata(values: np.ndarray, count: int, names: tuple[str, ...]) -> dict:
    """Equal-count strata in ascending order of ``values``, ties broken deterministically.

    Ascending leverage means the first stratum is the best supported one, which is the
    order the protocol names them in.  Splitting by rank rather than by value keeps the
    strata equal in size, so a per-stratum result is never a statement about how many
    candidates happened to fall in a value band.
    """
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    blocks = np.array_split(order, count)
    masks = {}
    for name, block in zip(names, blocks):
        mask = np.zeros(values.size, dtype=bool)
        mask[block] = True
        masks[name] = mask
    return masks


def match_support(table: sp.SupportTable, treated: np.ndarray, control: np.ndarray,
                  caliper: float) -> dict:
    """One to one nearest-neighbour matching on coverage, leverage and neighbourhood.

    Coverage is matched exactly, so a candidate with no STRING is never matched to one
    that has it.  Leverage and the neighbourhood distance are matched inside a caliper
    on percentiles of the pooled population, which keeps the two axes on one scale
    without inventing a weighting between them.  Matching is without replacement and
    greedy in ascending pair distance, so the pools end exactly the same size and their
    absolute numbers are comparable.  Nothing about a score enters it.
    """
    treated = np.flatnonzero(np.asarray(treated, dtype=bool))
    control = np.flatnonzero(np.asarray(control, dtype=bool))
    leverage = table.leverage_percentile
    density = table.d10_percentile
    coverage = table.coverage

    pairs: list[tuple[int, int]] = []
    for pattern in ov.COVERAGE_PATTERNS:
        left = treated[coverage[treated] == pattern]
        right = control[coverage[control] == pattern]
        if left.size == 0 or right.size == 0:
            continue
        distance = np.abs(leverage[left][:, None] - leverage[right][None, :]) \
            + np.abs(density[left][:, None] - density[right][None, :])
        legal = (np.abs(leverage[left][:, None] - leverage[right][None, :]) <= caliper) \
            & (np.abs(density[left][:, None] - density[right][None, :]) <= caliper)
        distance = np.where(legal, distance, np.inf)
        flat = np.argsort(distance, axis=None, kind="stable")
        used_left, used_right = set(), set()
        for position in flat.tolist():
            row, column = divmod(position, right.size)
            if not np.isfinite(distance[row, column]):
                break
            if row in used_left or column in used_right:
                continue
            used_left.add(row)
            used_right.add(column)
            pairs.append((int(left[row]), int(right[column])))
    if not pairs:
        return {"matched": 0}
    treated_matched = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
    control_matched = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
    balance = {}
    for name, values in (("leverage_percentile", leverage), ("d10_percentile", density),
                         ("string_degree_percentile", sp.percentile_of(table.string_degree))):
        balance[name] = {
            "treated_mean": float(values[treated_matched].mean()),
            "control_mean": float(values[control_matched].mean()),
            "absolute_standardised_difference": float(
                abs(values[treated_matched].mean() - values[control_matched].mean())
                / max(np.sqrt(0.5 * (values[treated_matched].var() + values[control_matched].var())),
                      1e-12))}
    return {"matched": len(pairs), "treated_rows": treated_matched,
            "control_rows": control_matched, "balance": balance,
            "coverage_composition": {pattern: int((coverage[treated_matched] == pattern).sum())
                                     for pattern in ov.COVERAGE_PATTERNS}}


# --- controls added on 2026-09-18, after Phase A round 1 ---------------------
# Round 1 produced a decomposition that does not behave like a decomposition: every
# leverage stratum of the unseen pool was three to seven times more negative than the
# unseen pool it partitions, and the support-matched unseen pool came out positive
# while the pool it was drawn from is negative.  Both of those are exactly the shape of
# an artifact this programme has been burned by before, where restricting a candidate
# pool moves the per-query random and oracle references and the change is read as
# signal.  So the leverage stratification is left exactly as pre-registered and is
# given a control that shares its size and its stratum count and differs only in that
# membership is random.  If random strata of 1251 reproduce the stratified numbers,
# then the stratification says nothing about support and the pre-registered reading
# does not survive.  Seeds are fixed here before the control runs.
CONTROL_SEEDS = (20260918, 20260919, 20260920)
# The matched-size null needs more draws than the stratification null does.  Fifteen
# random strata put the leverage quintiles far outside their range, so three seeds
# settled that question.  Three draws at the matched size gave a null spanning 0.030,
# which is wider than the 0.027 it is supposed to bound, so the control had no
# resolution at all.  Twelve seeds are used there instead.  This changes only how
# precisely the null is estimated and touches nothing about the quantity under test.
MATCHED_NULL_SEEDS = tuple(20260918 + step for step in range(12))


def random_strata_control(size: int, count: int, seed: int) -> list[np.ndarray]:
    """``count`` equal random blocks of a pool of ``size``, the size-matched null."""
    order = np.random.default_rng(seed).permutation(size)
    return [np.sort(part) for part in np.array_split(order, count)]


def leverage_from_matrix(training: np.ndarray, features: np.ndarray,
                         penalty: float, columns: np.ndarray | None = None) -> np.ndarray:
    """Leverage recomputed from scratch, optionally on a subset of the feature columns.

    This is a second implementation of the primary diagnostic rather than a call into
    the first, so agreement between them is evidence and not a tautology.  The column
    subset exists because the frozen standardization clips a feature's scale at 1e-8:
    a column the training rows never vary gets divided by that clip, so a candidate
    with any nonzero value there acquires an enormous standardized coordinate and an
    enormous leverage.  That is arithmetically correct as a statement about
    extrapolation and it is also capable of dominating the whole diagnostic, so the
    share it contributes is measured rather than assumed away.  The primary diagnostic
    is not changed.
    """
    training = np.asarray(training, dtype=np.float64)
    features = np.asarray(features, dtype=np.float64)
    if columns is not None:
        training, features = training[:, columns], features[:, columns]
    mean = training.mean(axis=0)
    scale = np.maximum(training.std(axis=0), 1e-8)
    scaled = (training - mean) / scale
    centred = (features - mean) / scale
    _, singular, right = np.linalg.svd(scaled, full_matrices=False)
    projected = centred @ right.T
    remainder = np.maximum((centred ** 2).sum(axis=1) - (projected ** 2).sum(axis=1), 0.0)
    return ((projected ** 2) / (singular ** 2 + penalty)).sum(axis=1) + remainder / penalty


def evaluate_mask(pack, score_by_arm: dict, mask: np.ndarray) -> dict:
    block = ev.evaluate_arms(score_by_arm, pack.utility, pack.ids, pack.self_positions, mask)
    if "skipped" in block:
        return block
    vectors = block.pop("_vectors")
    block["paired_vs_base"] = {name: paired_all(vectors, name, "BASE")
                               for name in vectors if name != "BASE"}
    return block


def run(args: argparse.Namespace) -> dict:
    packs = {name: hn.load_pack(name, args.packs) for name in fc.CONTEXTS}
    held = ov.NATURAL_FOLD
    training = ov.NATURAL_TRAINING
    pack = packs[held]

    pool = hn.training_pool(packs, training)
    fitted = pd.fit(pool["features"], pool["responses"], pool["present"], pool["keys"], training)
    effect = fitted.effect(pack.features, pack.present)
    score_by_arm = {"BASE": pack.base, "EFFECT": hn.effect_arms(pack, effect)}

    supervised = np.unique([key.split("|", 1)[1] for key in pool["keys"].tolist()])
    seen_mask = np.isin(pack.ids, supervised)
    unseen_mask = ~seen_mask

    adjacency_file = np.load(args.string_adjacency, allow_pickle=False)
    adjacency = {key: adjacency_file[key] for key in
                 ("genes", "degree", "edge_row", "edge_col", "edge_w")}
    keep = np.flatnonzero(np.asarray(pool["present"], dtype=bool))
    training_features, training_identities = sp.unique_training(
        pool["features"][keep], pool["keys"][keep])
    table = sp.build(pack.ids, pack.features, fitted.path, fitted.penalty,
                     training_features, training_identities, adjacency)

    record: dict = {
        "schema": "VCDESIGN_OPEN_VOCAB_V1_NATURAL_UNSEEN_SUPPORT_AUDIT",
        "held": held, "training": list(training),
        "read_anchor_G_check": False,
        "support_is_never_a_model_input": True,
        "predictor": {"rows": fitted.rows, "identities": fitted.identities,
                      "penalty": fitted.penalty,
                      "held_out_gene_space_cosine": fitted.held_out_cosine,
                      "explained_energy": fitted.basis.explained},
        "pool": int(pack.ids.size), "queries": int(pack.query_rows.size),
        "seen_candidates": int(seen_mask.sum()), "unseen_candidates": int(unseen_mask.sum()),
        "training_identities": int(training_identities.size),
        "leverage_definition": ov.LEVERAGE_DEFINITION,
        "neighbour_space": ov.NEIGHBOUR_SPACE,
    }

    # The number this phase exists to explain, recomputed under the same recipe.
    record["reproduction"] = {}
    for name, mask in (("ALL", np.ones(pack.ids.size, dtype=bool)),
                       ("SEEN_CANDIDATE", seen_mask), ("UNSEEN_CANDIDATE", unseen_mask)):
        block = evaluate_mask(pack, score_by_arm, mask)
        record["reproduction"][name] = block

    # 3.1 to 3.4 and 4: the support table itself, summarised by candidate class.
    record["support_summary"] = {}
    for name, mask in (("SEEN_CANDIDATE", seen_mask), ("UNSEEN_CANDIDATE", unseen_mask)):
        rows = np.flatnonzero(mask)
        record["support_summary"][name] = {
            "count": int(rows.size),
            "coverage": {pattern: int((table.coverage[rows] == pattern).sum())
                         for pattern in ov.COVERAGE_PATTERNS},
            "string_in_graph": int(table.string_in_graph[rows].sum()),
            "string_degree": {"median": float(np.median(table.string_degree[rows])),
                              "mean": float(table.string_degree[rows].mean()),
                              "zero": int((table.string_degree[rows] == 0).sum())},
            "training_neighbours": {"median": float(np.median(table.training_neighbours[rows])),
                                    "mean": float(table.training_neighbours[rows].mean()),
                                    "zero": int((table.training_neighbours[rows] == 0).sum())},
            "training_neighbour_weight": {
                "median": float(np.median(table.training_neighbour_weight[rows])),
                "mean": float(table.training_neighbour_weight[rows].mean())},
            "leverage": {"median": float(np.median(table.leverage[rows])),
                         "mean": float(table.leverage[rows].mean()),
                         "p05": float(np.quantile(table.leverage[rows], 0.05)),
                         "p95": float(np.quantile(table.leverage[rows], 0.95))},
            "d1": {"median": float(np.median(table.d1[rows]))},
            "d10": {"median": float(np.median(table.d10[rows])),
                    "mean": float(table.d10[rows].mean())},
            "d50": {"median": float(np.median(table.d50[rows]))},
            "mapkg_d10": {"median": float(np.median(table.mapkg_d10[rows]))},
        }

    # 5: five equal strata of the unseen pool, by the primary support diagnostic.
    unseen_rows = np.flatnonzero(unseen_mask)
    strata = quantile_strata(table.leverage[unseen_rows], ov.SUPPORT_STRATA, ov.STRATUM_NAMES)
    record["unseen_support_strata"] = {}
    for name, inside in strata.items():
        rows = unseen_rows[inside]
        mask = np.zeros(pack.ids.size, dtype=bool)
        mask[rows] = True
        block = evaluate_mask(pack, score_by_arm, mask)
        block["support"] = {
            "count": int(rows.size),
            "leverage_range": [float(table.leverage[rows].min()), float(table.leverage[rows].max())],
            "leverage_median": float(np.median(table.leverage[rows])),
            "d10_median": float(np.median(table.d10[rows])),
            "string_degree_median": float(np.median(table.string_degree[rows])),
            "training_neighbours_median": float(np.median(table.training_neighbours[rows])),
            "coverage": {pattern: int((table.coverage[rows] == pattern).sum())
                         for pattern in ov.COVERAGE_PATTERNS}}
        record["unseen_support_strata"][name] = block

    # 6: seen against unseen at matched static support, both pools the same size.
    record["support_matched"] = {}
    for label, caliper in (("primary", ov.MATCH_CALIPER),
                           ("sensitivity", ov.MATCH_CALIPER_SENSITIVITY)):
        matched = match_support(table, seen_mask, unseen_mask, caliper)
        entry: dict = {"caliper": caliper, "matched": matched.get("matched", 0)}
        if matched.get("matched", 0) > 0:
            entry["balance"] = matched["balance"]
            entry["coverage_composition"] = matched["coverage_composition"]
            for side, rows in (("SEEN_MATCHED", matched["treated_rows"]),
                               ("UNSEEN_MATCHED", matched["control_rows"])):
                mask = np.zeros(pack.ids.size, dtype=bool)
                mask[rows] = True
                entry[side] = evaluate_mask(pack, score_by_arm, mask)
        record["support_matched"][label] = entry

    # The size-matched null for the stratification, and for the matched comparison.
    control: dict = {"seeds": list(CONTROL_SEEDS), "note": (
        "same pool, same sizes, random membership; the pre-registered stratification is "
        "unchanged and is read against this")}
    control["random_strata"] = {}
    for seed in CONTROL_SEEDS:
        blocks = random_strata_control(unseen_rows.size, ov.SUPPORT_STRATA, seed)
        values = []
        for block in blocks:
            mask = np.zeros(pack.ids.size, dtype=bool)
            mask[unseen_rows[block]] = True
            item = evaluate_mask(pack, score_by_arm, mask)
            values.append({"count": int(block.size),
                           "delta_MBRU": item["paired_vs_base"]["EFFECT"]["MBRU"]["median"],
                           "improved": item["paired_vs_base"]["EFFECT"]["MBRU"]["fraction_query_improved"],
                           "BASE_MBRU": item["arms"]["BASE"]["MBRU"]["median"],
                           "EFFECT_MBRU": item["arms"]["EFFECT"]["MBRU"]["median"]})
        control["random_strata"][str(seed)] = values
    matched_size = record["support_matched"]["primary"]["matched"]
    control["random_at_matched_size"] = []
    if matched_size:
        for seed in MATCHED_NULL_SEEDS:
            block = np.random.default_rng(seed + 1).choice(unseen_rows.size, size=matched_size,
                                                           replace=False)
            mask = np.zeros(pack.ids.size, dtype=bool)
            mask[unseen_rows[np.sort(block)]] = True
            item = evaluate_mask(pack, score_by_arm, mask)
            control["random_at_matched_size"].append({
                "seed": seed, "count": int(matched_size),
                "delta_MBRU": item["paired_vs_base"]["EFFECT"]["MBRU"]["median"],
                "improved": item["paired_vs_base"]["EFFECT"]["MBRU"]["fraction_query_improved"]})
    record["pool_size_control"] = control

    # Where the leverage diagnostic's mass actually comes from.
    training_all = pool["features"][keep]
    scale = np.maximum(training_all.std(axis=0), 0.0)
    degenerate = scale <= 1e-8
    healthy = np.flatnonzero(~degenerate)
    independent = leverage_from_matrix(training_all, pack.features, fitted.penalty)
    restricted = leverage_from_matrix(training_all, pack.features, fitted.penalty, healthy)
    record["leverage_diagnostic"] = {
        "degenerate_feature_columns": int(degenerate.sum()),
        "feature_width": int(training_all.shape[1]),
        "independent_recomputation_max_relative_error": float(
            np.abs(independent - table.leverage).max() / max(np.abs(table.leverage).max(), 1e-30)),
        "primary_leverage_p99": float(np.quantile(table.leverage, 0.99)),
        "primary_leverage_max": float(table.leverage.max()),
        "restricted_leverage_p99": float(np.quantile(restricted, 0.99)),
        "restricted_leverage_max": float(restricted.max()),
        "spearman_primary_vs_restricted": float(np.corrcoef(
            sp.percentile_of(table.leverage), sp.percentile_of(restricted))[0, 1]),
        "note": ("the primary diagnostic is unchanged; the restricted one drops the feature "
                 "columns the training atlas never varies, whose scale is the 1e-8 clip")}
    strata_restricted = quantile_strata(restricted[unseen_rows], ov.SUPPORT_STRATA,
                                        ov.STRATUM_NAMES)
    record["unseen_support_strata_restricted_sensitivity"] = {}
    for name, inside in strata_restricted.items():
        mask = np.zeros(pack.ids.size, dtype=bool)
        mask[unseen_rows[inside]] = True
        item = evaluate_mask(pack, score_by_arm, mask)
        record["unseen_support_strata_restricted_sensitivity"][name] = {
            "count": int(inside.sum()),
            "leverage_median": float(np.median(restricted[unseen_rows][inside])),
            "delta_MBRU": item["paired_vs_base"]["EFFECT"]["MBRU"]["median"],
            "ci95": item["paired_vs_base"]["EFFECT"]["MBRU"]["bootstrap"]["median"]["ci95"],
            "improved": item["paired_vs_base"]["EFFECT"]["MBRU"]["fraction_query_improved"]}

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record["runtime"] = {"timestamp": datetime.now(timezone.utc).isoformat(),
                         "python": platform.python_version(), "numpy": np.__version__}
    output.write_text(json.dumps(record, indent=2, sort_keys=True, default=float) + "\n")
    np.savez(output.parent / "k562_support_table.npz",
             **{name: value for name, value in table.as_dict().items()},
             seen=seen_mask)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase A: natural unseen-candidate support audit")
    parser.add_argument("--packs", required=True)
    parser.add_argument("--string-adjacency", dest="string_adjacency", required=True)
    parser.add_argument("--output", required=True)
    record = run(parser.parse_args())

    print(f"[{record['held']}] pool {record['pool']} queries {record['queries']}, "
          f"seen {record['seen_candidates']} unseen {record['unseen_candidates']}")
    for name, block in record["reproduction"].items():
        if "paired_vs_base" not in block:
            continue
        item = block["paired_vs_base"]["EFFECT"]["MBRU"]
        low, high = item["bootstrap"]["median"]["ci95"]
        print(f"  {name:18s} {block['candidates']:5d} candidates  EFFECT - BASE "
              f"{item['median']:+.4f} [{low:+.4f},{high:+.4f}] improved "
              f"{item['fraction_query_improved']:.3f}")
    print("\nunseen strata, ascending ridge leverage (best supported first)")
    for name, block in record["unseen_support_strata"].items():
        if "paired_vs_base" not in block:
            print(f"  {name:22s} {block.get('skipped')}")
            continue
        item = block["paired_vs_base"]["EFFECT"]["MBRU"]
        low, high = item["bootstrap"]["median"]["ci95"]
        print(f"  {name:22s} n={block['support']['count']:4d} "
              f"lev {block['support']['leverage_median']:.3g}  "
              f"EFFECT - BASE {item['median']:+.4f} [{low:+.4f},{high:+.4f}] "
              f"improved {item['fraction_query_improved']:.3f}")
    print("\nsize-matched null: the same pool, the same stratum sizes, random membership")
    for seed, values in record["pool_size_control"]["random_strata"].items():
        line = "  ".join(f"{item['delta_MBRU']:+.4f}" for item in values)
        print(f"  seed {seed}  {line}")
    draws = [item["delta_MBRU"] for item in record["pool_size_control"]["random_at_matched_size"]]
    print(f"  random at the matched size ({record['support_matched']['primary']['matched']}), "
          f"{len(draws)} draws: median {np.median(draws):+.4f} "
          f"range [{min(draws):+.4f},{max(draws):+.4f}] "
          f"p90 {np.quantile(draws, 0.90):+.4f}")
    diagnostic = record["leverage_diagnostic"]
    print(f"\nleverage: {diagnostic['degenerate_feature_columns']} of "
          f"{diagnostic['feature_width']} training columns are at the 1e-8 scale clip; "
          f"max {diagnostic['primary_leverage_max']:.3g} against "
          f"{diagnostic['restricted_leverage_max']:.3g} without them; independent "
          f"recomputation agrees to {diagnostic['independent_recomputation_max_relative_error']:.2e}")
    print("restricted-leverage strata (sensitivity, not the primary)")
    for name, entry in record["unseen_support_strata_restricted_sensitivity"].items():
        low, high = entry["ci95"]
        print(f"  {name:22s} n={entry['count']:4d} lev {entry['leverage_median']:.4g}  "
              f"EFFECT - BASE {entry['delta_MBRU']:+.4f} [{low:+.4f},{high:+.4f}] "
              f"improved {entry['improved']:.3f}")
    for label, entry in record["support_matched"].items():
        print(f"\nsupport matched ({label}, caliper {entry['caliper']}): "
              f"{entry['matched']} pairs")
        for side in ("SEEN_MATCHED", "UNSEEN_MATCHED"):
            block = entry.get(side)
            if not block or "paired_vs_base" not in block:
                continue
            item = block["paired_vs_base"]["EFFECT"]["MBRU"]
            low, high = item["bootstrap"]["median"]["ci95"]
            print(f"  {side:16s} EFFECT - BASE {item['median']:+.4f} "
                  f"[{low:+.4f},{high:+.4f}] improved {item['fraction_query_improved']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
