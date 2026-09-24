"""Canonical score matrices: every arm, every held context, both regimes, both views, on the pool axis.

A score file is self-sufficient for grading: ``evaluate`` reads it and the frozen pack utilities and
nothing else, so no model is rerun to reproduce a number.  Scores are ``[queries, pool]`` float64 per
view; a candidate an arm cannot represent follows ``UNSCORED`` and is recorded in ``{arm}|covered``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..decision_consistent_effect_v1 import arms as dca
from ..four_context_v1 import evaluate as ev
from ..open_vocab_generalization_v1 import masking as mk
from . import contract as cc

NOT_SCORED = -1.0e300   # an uncovered CellNavi column; never inside a stratum it is graded on


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-300)


def _gears_effect(run: Path, name: str, ids: np.ndarray, genes: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    # ``genes`` was written by ``gears_fit`` from a pandas column, so it is an object array; it is our own
    # file, and it must still equal the frozen G_4 axis exactly.
    store = np.load(run / "gears" / name / "effects.npz", allow_pickle=True)
    if not np.array_equal(np.asarray(store["genes"], dtype=str), genes):
        raise RuntimeError(f"GEARS fit {name} is not on the G_4 axis")
    index = {g: i for i, g in enumerate(store["identities"].astype(str).tolist())}
    covered = np.asarray([g in index for g in ids.tolist()], dtype=bool)
    effect = np.zeros((ids.size, genes.size), dtype=np.float64)
    effect[covered] = store["delta"][[index[g] for g in ids[covered].tolist()]]
    return effect, covered, {"excluded": set(store["excluded"].astype(str).tolist()),
                             "training": set(store["training_identities"].astype(str).tolist())}


def gears_effects(run: Path, held: str, pack, eligible: np.ndarray, folds: np.ndarray, genes: np.ndarray) -> dict:
    ids = np.asarray(pack.ids, dtype=str)
    seen, covered, _ = _gears_effect(run, cc.SEEN_POOL, ids, genes)
    parts = []
    for fold in range(cc.MASK_FOLDS):
        effect, part_covered, meta = _gears_effect(run, f"FOLD_{fold}", ids, genes)
        inside = set(ids[eligible & (folds == fold)].tolist())
        if not inside <= meta["excluded"] or inside & meta["training"]:
            raise RuntimeError(f"GEARS {held} fold {fold}: masked set not excluded or firewall breach")
        if not np.array_equal(part_covered, covered):
            raise RuntimeError("GEARS coverage differs between fits")
        parts.append(effect)
    return {"SEEN": seen, "MASKED": mk._crossfit(seen, parts, eligible, folds), "covered": covered}


def ridge_k562_effects(run: Path, held: str) -> dict:
    """The A1 recipe on the K562-only atlas under GEARS's fold-union masks (``ridge_k562.py``)."""
    store = np.load(run / "ridge_k562" / f"effects_{held}.npz", allow_pickle=False)
    return {"SEEN": store["SEEN"].astype(np.float64), "MASKED": store["MASKED"].astype(np.float64)}


def cellnavi_scores(run: Path, held: str, pack, subdir: str = "cellnavi") -> dict | None:
    path = run / subdir / held / "scores.npz"
    if not path.exists():
        return None
    store = np.load(path, allow_pickle=False)
    ids = np.asarray(pack.ids, dtype=str)
    queries = ids[np.asarray(pack.self_positions)]
    row_of = {(q, int(v)): i for i, (q, v) in enumerate(zip(store["queries"].astype(str).tolist(), store["views"].tolist()))}
    class_of = {c: i for i, c in enumerate(store["classes"].astype(str).tolist())}
    covered = np.asarray([c in class_of for c in ids.tolist()], dtype=bool)
    columns = np.asarray([class_of.get(c, 0) for c in ids.tolist()])
    by_view = []
    for view in (0, 1):
        rows = np.asarray([row_of[(q, view)] for q in queries.tolist()])
        matrix = store["mean_log_prob"][rows][:, columns].astype(np.float64)
        matrix[:, ~covered] = NOT_SCORED
        by_view.append(matrix)
    own = np.asarray([class_of.get(q, -1) for q in queries.tolist()])
    ident = {}
    for view in (0, 1):
        rows = np.asarray([row_of[(q, view)] for q in queries.tolist()])
        logp = store["mean_log_prob"][rows]
        has = own >= 0
        rank = np.asarray([int((logp[i] > logp[i, own[i]]).sum()) + 1 if has[i] else -1 for i in range(len(own))])
        ident[view] = {"query_is_class": has, "rank_of_own_class": rank}
    cell_hits = store["cell_top1"] == np.asarray([class_of.get(q, -1) for q in store["cell_query"].astype(str).tolist()])
    return {"by_view": by_view, "covered": covered, "identification": ident,
            "cell_top1_accuracy": float(cell_hits.mean()), "cells": int(store["cells"].sum())}


def profiles(training_packs: dict, held: str, pack) -> tuple[np.ndarray, np.ndarray]:
    """``s_c``: unit mean of the candidate's unit consensus responses over the training contexts
    whose ``RIDGE_UNIT`` rows may include it (K562: ``G_fit``; others: every usable identity)."""
    sums: dict = {}
    for context, other in training_packs.items():
        if context == held:
            continue
        rows = np.flatnonzero(other.fit_mask)
        units = _unit(ev.consensus(np.asarray(other.responses)[:, rows]))
        for identity, vector in zip(np.asarray(other.ids, dtype=str)[rows].tolist(), units):
            sums[identity] = sums.get(identity, 0.0) + vector
    ids = np.asarray(pack.ids, dtype=str)
    covered = np.asarray([i in sums for i in ids.tolist()], dtype=bool)
    signature = np.zeros((ids.size, np.asarray(pack.responses).shape[2]), dtype=np.float64)
    signature[covered] = _unit(np.stack([sums[i] for i in ids[covered].tolist()]))
    return signature, covered


def build(run: Path, held: str, pack, genes: np.ndarray, training_packs: dict) -> dict:
    """All arms for one held context.  Returns ``scores[regime][arm] = [view0, view1]`` and effects."""
    c018 = np.load(Path(cc.C018_RUN) / f"effects_{held}.npz", allow_pickle=False)
    eligible, folds = c018["eligible"], c018["folds"]
    context_index = cc.PRIMARY_CONTEXTS.index(held)
    queries, pool = pack.base[0].shape
    effects = {"SEEN": {cc.INTERNAL_FTM: c018["unit_seen"].astype(np.float64)},
               "MASKED": {cc.INTERNAL_FTM: c018["unit_cross"].astype(np.float64)}}
    raw = {"SEEN": c018["raw_seen"].astype(np.float64), "MASKED": c018["raw_cross"].astype(np.float64)}
    gears = gears_effects(run, held, pack, eligible, folds, genes) if (run / "gears" / cc.SEEN_POOL).exists() else None
    k562 = ridge_k562_effects(run, held)
    navi = cellnavi_scores(run, held, pack)
    native = cellnavi_scores(run, held, pack, "cellnavi_native")
    signature, profiled = profiles(training_packs, held, pack)
    if not profiled[eligible].all():
        raise RuntimeError(f"{held}: a C-016 eligible candidate has no training-context signature")
    covered = {cc.GEARS: gears["covered"] if gears else np.zeros(pool, dtype=bool),
               cc.CELLNAVI: navi["covered"] if navi else np.zeros(pool, dtype=bool),
               cc.CELLNAVI_NATIVE: native["covered"] if native else np.zeros(pool, dtype=bool)}
    scores = {}
    for r_index, regime in enumerate(cc.REGIMES):
        if gears:
            effects[regime][cc.GEARS] = gears[regime]
        effects[regime][cc.RIDGE_K562] = k562[regime]
        block = {}
        rng = [np.random.default_rng([cc.RANDOM_SEED, context_index, r_index, v]) for v in (0, 1)]
        block[cc.RANDOM] = [g.standard_normal((queries, pool)) for g in rng]
        magnitude = np.linalg.norm(raw[regime], axis=1)
        block[cc.PRIOR] = [np.broadcast_to(magnitude, (queries, pool)).copy() for _ in (0, 1)]
        block[cc.BASE] = [np.asarray(pack.base[v], dtype=np.float64) for v in (0, 1)]
        cosine = dca.cosine_scores(pack, effects[regime][cc.INTERNAL_FTM])
        block[cc.INTERNAL_FTM] = cosine
        block[cc.VCDESIGN] = [ev.fuse(pack.base[v], cosine[v]) for v in (0, 1)]
        block[cc.ORACLE] = [ev.effect_cosine(pack.query_responses, pack.responses[v], v) for v in (0, 1)]
        block[cc.RIDGE_K562] = dca.cosine_scores(pack, k562[regime])
        if gears:
            block[cc.GEARS] = dca.cosine_scores(pack, gears[regime])
        if navi and regime == "SEEN":
            block[cc.CELLNAVI] = navi["by_view"]
        if native and regime == "SEEN":
            block[cc.CELLNAVI_NATIVE] = native["by_view"]
        if regime == "SEEN":
            block[cc.PROFILED] = dca.cosine_scores(pack, signature)
        scores[regime] = block
    return {"scores": scores, "effects": effects, "eligible": eligible, "folds": folds,
            "covered": covered, "cellnavi": navi, "cellnavi_native": native}


def save(path: Path, built: dict, pack) -> None:
    payload = {"ids": np.asarray(pack.ids, dtype=str), "eligible": built["eligible"], "folds": built["folds"],
               "self_positions": np.asarray(pack.self_positions)}
    for regime, block in built["scores"].items():
        for arm, views in block.items():
            for view in (0, 1):
                payload[f"{regime}|{arm}|{view}"] = views[view]
    for arm, mask in built["covered"].items():
        payload[f"{arm}|covered"] = mask
    np.savez(path, **payload)


def load(path: Path) -> dict:
    store = np.load(path, allow_pickle=False)
    scores: dict = {}
    for key in store.files:
        parts = key.split("|")
        if len(parts) == 3:
            scores.setdefault(parts[0], {}).setdefault(parts[1], [None, None])[int(parts[2])] = store[key]
    return {"scores": scores, "ids": store["ids"].astype(str), "eligible": store["eligible"], "folds": store["folds"],
            "self_positions": store["self_positions"],
            "covered": {k.split("|")[0]: store[k] for k in store.files if k.endswith("|covered")}}
