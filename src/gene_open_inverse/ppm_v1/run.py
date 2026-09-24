#!/usr/bin/env python3
"""PERTURBATION_PROGRAM_MEMORY_V1 from the cached four-context packs.

Stages
  smoke      one short selection run with every health statistic and firewall probe
  fit        one arm on one pool (SEEN or a MASKED fold) of one held context: select, refit,
             predict the held pool, save predictions, model and firewall record
  ridge      A1 refitted on the SEEN and fold pools of one held context, gated against C-018
  grade      every available arm of one held context against A1, SEEN and MASKED
  stress     PPM_FULL on K562's 6255 unseen candidates (secondary)
  summarize  pooled contrasts and the frozen decision
The score is always the frozen cosine and the fusion the frozen ``row_z`` sum.  No stage
reads ``G_check`` and no stage overwrites its own output.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ..decision_consistent_effect_v1 import arms as dca
from ..decision_consistent_effect_v1 import stats as st
from ..four_context_v1 import contract as fc
from ..four_context_v1 import evaluate as ev
from ..four_context_v1 import harness as hn
from ..four_context_v1 import predictors as pd
from ..open_vocab_generalization_v1 import contract as ov
from ..open_vocab_generalization_v1 import masking as mk
from ..rwed_v1 import contract as rc
from ..rwed_v1 import estimator as es
from ..utility_gt_v2.validate import summarize
from . import contract as pc
from . import data as dt
from . import decide as dd
from . import train as tr

READ_ANCHOR_G_CHECK = False
HEADLINE = ("MBRU", "Mean@10", "Mean@20", "Mean@50")
COMPANION = ("HighValueCount@10", "HighValueCount@20", "HighValueCount@50", "Best@10", "Best@20", "Best@50")


def _refuse(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"{path} exists; this stage never overwrites its own output")


def _runtime() -> dict:
    return {"timestamp": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "numpy": np.__version__, "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}


def _unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-30)


def _remove(effect: np.ndarray, directions: np.ndarray) -> np.ndarray:
    effect = np.asarray(effect, dtype=np.float64)
    return effect - np.einsum("ij,ij->i", effect, directions)[:, None] * directions


def held_setup(packs: dict, held: str, training: tuple | None = None) -> dict:
    """The frozen C-016 eligibility and folds for one held context."""
    training = training or tuple(name for name in fc.CONTEXTS if name != held)
    pack = packs[held]
    full = es.views_pool(packs, training)
    supervised = np.unique([key.split("|", 1)[1] for key in full["keys"].tolist()])
    eligible = np.isin(pack.ids, supervised) & np.asarray(pack.present, dtype=bool)
    return {"training": training, "pack": pack, "eligible": eligible, "folds": mk.fold_assignment(pack.ids),
            "unseen": ~np.isin(pack.ids, supervised)}


def pool_exclusion(setup: dict, pool_name: str) -> np.ndarray:
    if pool_name == pc.SEEN_POOL:
        return np.asarray([], dtype=str)
    fold = int(pool_name)
    return setup["pack"].ids[setup["eligible"] & (setup["folds"] == fold)]


def firewall_record(pool: dt.Pool, fitted: tr.Fitted, excluded: np.ndarray) -> dict:
    """The masked identities must be absent from the training rows and from the memory."""
    excluded = set(np.asarray(excluded, dtype=str).tolist())
    training = set(fitted.pool.identities.tolist())
    memory = set(fitted.memory.identities.tolist()) if fitted.memory is not None else set()
    record = {"excluded_identities": len(excluded), "training_identities": len(training),
              "memory_identities": len(memory),
              "excluded_in_training": len(excluded & training), "excluded_in_memory": len(excluded & memory),
              "excluded_in_pool": len(excluded & set(pool.identities.tolist()))}
    record["passes"] = bool(record["excluded_in_training"] == 0 and record["excluded_in_memory"] == 0
                            and record["excluded_in_pool"] == 0)
    if not record["passes"]:
        raise RuntimeError(f"firewall breach: {record}")
    return record


def _attention_entropy(weights: np.ndarray) -> np.ndarray:
    return -(weights * np.log(np.maximum(weights, 1e-30))).sum(axis=1)


def memory_diagnostics(prediction: dict, fitted: tr.Fitted, truth: np.ndarray, common: np.ndarray) -> dict:
    """D2 per predicted candidate: gate, attention, neighbour knowledge and effect similarity.

    ``truth`` is the held-context consensus of each predicted candidate.  It enters nothing
    but this diagnostic, which is computed after the prediction is fixed.
    """
    if "neighbours" not in prediction:
        return {}
    directions = dt.identity_directions(fitted.pool).astype(np.float64)
    rows = prediction["rows"]
    neighbours = prediction["neighbours"]
    weights = prediction["attention"].astype(np.float64)
    target = _unit(truth[rows])
    effect = np.einsum("ig,ikg->ik", target, directions[neighbours])
    specific_target = _unit(_remove(target, np.repeat(common[None], rows.size, 0)))
    specific_memory = directions - np.outer(directions @ common, common)
    specific_memory = _unit(specific_memory)
    specific = np.einsum("ig,ikg->ik", specific_target, specific_memory[neighbours])
    out = {"memory_weight": 1.0 - prediction["gate"] if "gate" in prediction else np.ones(rows.size),
           "attention_entropy": _attention_entropy(weights),
           "neighbour_knowledge_similarity": prediction["neighbour_similarity"].mean(axis=1),
           "neighbour_effect_similarity_weighted": (weights * effect).sum(axis=1),
           "neighbour_effect_similarity_mean": effect.mean(axis=1),
           "neighbour_specific_similarity_weighted": (weights * specific).sum(axis=1)}
    return {key: np.asarray(value, dtype=np.float64) for key, value in out.items()}


def retrieval_quality(selection: tr.Fitted, seed: int) -> dict:
    """D3: effect similarity of retrieved neighbours for never-trained validation identities."""
    held, pool, model = selection.held, selection.pool, selection.model
    common = dt.common_direction(pool, dt.program_basis(pool))
    val_dir = dt.identity_directions(held).astype(np.float64)
    mem_dir = dt.identity_directions(pool).astype(np.float64)
    remove = lambda v: _unit(v - np.outer(v @ common, common))
    val_spec, mem_spec = remove(val_dir), remove(mem_dir)

    def neighbours(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
        return np.argsort(-(_unit(query) @ _unit(keys).T), axis=1, kind="stable")[:, :pc.NEIGHBOURS]

    std = selection.standardizer
    model.eval()
    with torch.no_grad():
        encode = lambda f: model.encoder(tr._tensor(std.transform(f)), tr._tensor(std.presence(f))).cpu().numpy()
        h_val, h_mem = encode(held.features), encode(pool.features)
    sets = {"learned_h": neighbours(h_val, h_mem),
            "raw_standardized": neighbours(std.transform(held.features), std.transform(pool.features))}
    widths = np.cumsum((0,) + pc.MODALITY_WIDTHS)
    flags = std.flag_columns.start
    for index, name in enumerate(("raw_STRING", "raw_MAPKG")):
        block = slice(widths[index], widths[index + 1])
        has_val = held.features[:, flags + index] > 0
        has_mem = pool.features[:, flags + index] > 0
        chosen = np.full((held.identities.size, pc.NEIGHBOURS), -1)
        chosen[has_val] = np.flatnonzero(has_mem)[neighbours(held.features[has_val, block],
                                                             pool.features[has_mem, block])]
        sets[name] = chosen
    generator = np.random.default_rng(seed)
    sets["random"] = np.stack([generator.choice(pool.identities.size, pc.NEIGHBOURS, replace=False)
                               for _ in range(held.identities.size)])
    per, labels = {}, held.identities
    for name, index in sets.items():
        valid = (index >= 0).all(axis=1)
        full = np.full(index.shape[0], np.nan)
        spec = np.full(index.shape[0], np.nan)
        full[valid] = np.einsum("ig,ikg->ik", val_dir[valid], mem_dir[index[valid]]).mean(axis=1)
        spec[valid] = np.einsum("ig,ikg->ik", val_spec[valid], mem_spec[index[valid]]).mean(axis=1)
        per[name] = {"full": full, "specific": spec}
    summary = {name: {kind: float(np.nanmedian(values)) for kind, values in entry.items()}
               for name, entry in per.items()}
    for other in ("raw_standardized", "raw_STRING", "raw_MAPKG", "random"):
        for kind in ("full", "specific"):
            difference = per["learned_h"][kind] - per[other][kind]
            keep = np.isfinite(difference)
            summary[f"learned_minus_{other}_{kind}"] = st.cluster_bootstrap_labels(difference[keep], labels[keep])
    summary["validation_identities"] = int(held.identities.size)
    return summary


def ridge_same_split(selection: tr.Fitted) -> float:
    """The A1 recipe on the selection run's own training part, scored on its validation rows."""
    pool, held = selection.pool, selection.held
    model = pd.fit(pool.features[pool.row_identity], pool.unit.astype(np.float64),
                   np.ones(pool.rows, dtype=bool),
                   np.asarray([f"{c}|{pool.identities[i]}" for c, i in zip(pool.row_context, pool.row_identity)]),
                   pool.contexts)
    effect = _unit(model.effect(held.features, np.ones(held.identities.size, dtype=bool)))
    return float(np.einsum("ij,ij->i", effect[held.row_identity], held.unit.astype(np.float64)).mean())


def fit_stage(packs: dict, held: str, arm: str, pool_name: str, seed: int, output: Path,
              training: tuple | None = None) -> dict:
    setup = held_setup(packs, held, training)
    pack = setup["pack"]
    excluded = pool_exclusion(setup, pool_name)
    pool = dt.assemble(packs, setup["training"], excluded)
    log_path = output / "train.log"

    def log(entry: dict) -> None:
        with log_path.open("a") as handle:
            handle.write(json.dumps(entry, default=float) + "\n")

    selection, final = tr.select_and_refit(pool, arm, seed, log=log)
    firewall = {"selection": firewall_record(pool, selection, excluded),
                "final": firewall_record(pool, final, excluded)}
    anchor = (final.anchor_model.effect(pack.features, pack.present) if final.anchor_model is not None else None)
    prediction = tr.predict(final.model, final.standardizer, final.memory, pack.features, pack.present, anchor)
    common = dt.common_direction(final.pool, dt.program_basis(final.pool))
    truth = ev.consensus(pack.responses)
    memory = memory_diagnostics(prediction, final, truth, common)
    record = {"held": held, "arm": arm, "pool": pool_name, "seed": seed, "training": list(setup["training"]),
              "firewall": firewall, "selection": selection.record, "final": final.record,
              "programs_training_codes": tr.program_usage(final),
              "programs_predicted_codes": tr.code_statistics(prediction["codes"][prediction["rows"]])}
    if pool_name == pc.SEEN_POOL:
        record["retrieval_quality"] = retrieval_quality(selection, seed)
        record["ridge_same_split_validation_cosine"] = ridge_same_split(selection)
    if "router_probabilities" in prediction:
        top = prediction["router_probabilities"].argmax(axis=1)
        record["router_top1_share_predicted"] = (np.bincount(top, minlength=pc.EXPERTS) / top.size).tolist()
    np.savez_compressed(output / "prediction.npz", effect=prediction["effect"].astype(np.float32),
                        codes=prediction["codes"], rows=prediction["rows"], ids=pack.ids.astype(str),
                        memory_identities=(final.memory.identities if final.memory is not None
                                           else np.asarray([], dtype=str)),
                        training_identities=final.pool.identities, excluded_identities=excluded.astype(str),
                        common_direction=common,
                        **({"anchor": anchor.astype(np.float32)} if anchor is not None else {}),
                        **{f"memory|{k}": v for k, v in memory.items()},
                        **{k: prediction[k] for k in ("gate", "neighbours", "attention", "router_probabilities")
                           if k in prediction})
    torch.save({"state_dict": final.model.state_dict(), "arm": arm,
                "standardizer_mean": final.standardizer.mean, "standardizer_scale": final.standardizer.scale,
                "memory_identities": (final.memory.identities if final.memory is not None else None),
                "training_identities": final.pool.identities}, output / "model.pt")
    return record


def _grade(pack, effects: dict, directions: np.ndarray, mask: np.ndarray, contrasts: dict,
           bootstrap: bool = True) -> tuple[dict, dict]:
    effect_only = {name: dca.cosine_scores(pack, effect) for name, effect in effects.items()}
    for name, effect in effects.items():
        effect_only[f"{name}_COMMON_REMOVED"] = dca.cosine_scores(pack, _remove(effect, directions))
    scores = {"effect_only": effect_only,
              "fused": {name: [ev.fuse(pack.base[v], values[v]) for v in (0, 1)]
                        for name, values in effect_only.items()}}
    record, flat = {}, {}
    for mode, by_arm in scores.items():
        by_arm = {pc.BASE: pack.base, **by_arm}
        block = ev.evaluate_arms(by_arm, pack.utility, pack.ids, pack.self_positions, mask)
        if "skipped" in block:
            record[mode] = block
            continue
        vectors = block.pop("_vectors")
        for name, by_view in by_arm.items():
            vectors[name].update(st.best_at(by_view, pack.utility, pack.ids, pack.self_positions, mask))
        table = {}
        for label, (better, worse) in contrasts.items():
            if better not in vectors or worse not in vectors:
                continue
            table[label] = {}
            for metric in HEADLINE + COMPANION:
                difference = vectors[better][metric] - vectors[worse][metric]
                entry = {"median": float(np.median(difference)), "mean": float(difference.mean()),
                         "fraction_query_views_improved": float((difference > 0).mean())}
                if bootstrap and metric in HEADLINE + COMPANION[:3]:
                    entry["query_identity"] = st.cluster_bootstrap_rows([st.view_pairs(difference)])
                table[label][metric] = entry
        record[mode] = {"candidates": block["candidates"], "query_views": block["queries"],
                        "arms": {name: {m: summarize(v) for m, v in vec.items()} for name, vec in vectors.items()},
                        "contrasts": table}
        for name, vec in vectors.items():
            for metric, values in vec.items():
                flat[f"{mode}|{name}|{metric}"] = np.asarray(values, dtype=np.float64)
    return record, flat


def contrast_table(arms: tuple) -> dict:
    table = {f"{pc.RIDGE}_minus_{pc.BASE}": (pc.RIDGE, pc.BASE)}
    for arm in arms:
        table[f"{arm}_minus_{pc.RIDGE}"] = (arm, pc.RIDGE)
        table[f"{arm}_minus_{pc.BASE}"] = (arm, pc.BASE)
        table[f"COMMON_REMOVED_{arm}_minus_{pc.RIDGE}"] = (f"{arm}_COMMON_REMOVED", f"{pc.RIDGE}_COMMON_REMOVED")
    for full, ablations in pc.ABLATION_OF.items():
        for other in ablations:
            if full in arms and other in arms:
                table[f"{full}_minus_{other}"] = (full, other)
    return table


def ridge_stage(packs: dict, held: str, c018: Path, output: Path) -> dict:
    """A1 on the SEEN and fold pools, gated against C-018's saved unit predictions."""
    setup = held_setup(packs, held)
    pack, eligible, folds = setup["pack"], setup["eligible"], setup["folds"]

    def fit(excluded) -> object:
        unit_pool = dca.unit_target_pool(es.views_pool(packs, setup["training"], excluded))
        return pd.fit(unit_pool["features"], unit_pool["responses"], unit_pool["present"],
                      unit_pool["keys"], setup["training"])

    seen_model = fit(())
    fold_models = [fit(pool_exclusion(setup, str(f))) for f in range(ov.MASK_FOLDS)]
    seen = seen_model.effect(pack.features, pack.present)
    per_fold = [m.effect(pack.features, pack.present) for m in fold_models]
    cross = mk._crossfit(seen, per_fold, eligible, folds)
    saved = np.load(c018 / f"effects_{held}.npz", allow_pickle=False)
    gate = {"eligible_identical": bool(np.array_equal(saved["eligible"], eligible)),
            "folds_identical": bool(np.array_equal(saved["folds"], folds)),
            "a1_seen_max_abs_error": float(np.abs(seen - saved["unit_seen"]).max()),
            "a1_cross_max_abs_error": float(np.abs(cross - saved["unit_cross"]).max())}
    gate["passes"] = bool(gate["eligible_identical"] and gate["folds_identical"]
                          and gate["a1_seen_max_abs_error"] <= pc.A1_EFFECT_TOLERANCE
                          and gate["a1_cross_max_abs_error"] <= pc.A1_EFFECT_TOLERANCE)
    if not gate["passes"]:
        raise RuntimeError(f"A1 provenance gate failed for {held}: {gate}")
    np.savez(output / f"ridge_{held}.npz", eligible=eligible, folds=folds, seen=seen, cross=cross,
             per_fold=np.stack(per_fold), seen_common=_unit(mk.intercept_effect(seen_model)),
             fold_common=np.stack([_unit(mk.intercept_effect(m)) for m in fold_models]),
             penalties=np.asarray([seen_model.penalty] + [m.penalty for m in fold_models]),
             held_out_cosine=np.asarray([seen_model.held_out_cosine] + [m.held_out_cosine for m in fold_models]))
    return {"held": held, "provenance_gate": gate, "seen_penalty": seen_model.penalty,
            "fold_penalties": [m.penalty for m in fold_models],
            "held_out_cosine": [seen_model.held_out_cosine] + [m.held_out_cosine for m in fold_models]}


def _load_arm(fits: Path, arm: str, held: str) -> dict:
    base = fits / arm / held
    seen = np.load(base / "SEEN" / "prediction.npz", allow_pickle=False)
    folds = [np.load(base / str(f) / "prediction.npz", allow_pickle=False) for f in range(ov.MASK_FOLDS)]
    records = {name: json.loads((base / name / "record.json").read_text())
               for name in [pc.SEEN_POOL] + [str(f) for f in range(ov.MASK_FOLDS)]}
    return {"seen": seen, "folds": folds, "records": records}


def homogeneity(effects: list, rows: np.ndarray, common: np.ndarray) -> dict:
    """D6: can five fold models' columns be mixed into one score row (C-016 floor and ratio)."""
    means = np.stack([_unit(np.asarray(e, dtype=np.float64)[rows].mean(axis=0)) for e in effects])
    similarity = means @ means.T
    off = similarity[~np.eye(len(effects), dtype=bool)]
    shares = [float(np.median(np.linalg.norm(_remove(_unit(np.asarray(e, dtype=np.float64)[rows]),
                                                     np.repeat(c[None], rows.size, 0)), axis=1)))
              for e, c in zip(effects, common)]
    ratio = max(shares) / max(min(shares), 1e-30)
    return {"minimum_pairwise_cosine": float(off.min()), "candidate_specific_share": shares,
            "share_ratio": float(ratio),
            "passes": bool(off.min() >= pc.HOMOGENEITY_FLOOR and ratio <= pc.SHARE_RATIO_TOLERANCE)}


def candidate_diagnostics(pack, effects: dict, directions: np.ndarray, rows: np.ndarray) -> tuple:
    """D4 per candidate: full and candidate-specific cosine to the measured held-context response."""
    truth = ev.consensus(np.asarray(pack.responses)[:, rows])
    direction = directions[rows]
    specific_truth = _remove(truth, direction)
    per = {}
    for label, effect in effects.items():
        predicted = np.asarray(effect, dtype=np.float64)[rows]
        norm = np.linalg.norm(predicted, axis=1)
        per[label] = {"full_cosine": np.einsum("ij,ij->i", _unit(predicted), _unit(truth)),
                      "specific_cosine": np.einsum("ij,ij->i", _unit(_remove(predicted, direction)),
                                                   _unit(specific_truth)),
                      "common_share": np.einsum("ij,ij->i", predicted, direction) ** 2 / np.maximum(norm ** 2, 1e-30)}
    summary = {label: {k: float(np.median(v)) for k, v in entry.items()} for label, entry in per.items()}
    deltas = {}
    for label in effects:
        if label == pc.RIDGE:
            continue
        for key in ("full_cosine", "specific_cosine", "common_share"):
            deltas[f"{label}|{key}"] = per[label][key] - per[pc.RIDGE][key]
    summary["delta_median_vs_ridge"] = {k: float(np.median(v)) for k, v in deltas.items()}
    return summary, deltas


def grade_stage(packs: dict, held: str, arms: tuple, fits: Path, output: Path) -> dict:
    setup = held_setup(packs, held)
    pack = setup["pack"]
    ridge = np.load(output / f"ridge_{held}.npz", allow_pickle=False)
    eligible, folds = ridge["eligible"], ridge["folds"]
    if not (np.array_equal(eligible, setup["eligible"]) and np.array_equal(folds, setup["folds"])):
        raise RuntimeError("ridge cache does not match the frozen eligibility and folds")
    rows = np.flatnonzero(eligible)
    loaded = {arm: _load_arm(fits, arm, held) for arm in arms}
    for arm, entry in loaded.items():
        files = {pc.SEEN_POOL: entry["seen"], **{str(f): entry["folds"][f] for f in range(ov.MASK_FOLDS)}}
        for name, record in entry["records"].items():
            if not (record["firewall"]["selection"]["passes"] and record["firewall"]["final"]["passes"]):
                raise RuntimeError(f"{arm} {held} {name}: firewall record does not pass")
            if not np.array_equal(files[name]["ids"], pack.ids.astype(str)):
                raise RuntimeError(f"{arm} {held} {name}: prediction axis differs from the pack")
            expected = set(pool_exclusion(setup, name).tolist())
            if set(files[name]["excluded_identities"].tolist()) != expected:
                raise RuntimeError(f"{arm} {held} {name}: the fit excluded a different identity set")
    seen_effects = {pc.RIDGE: ridge["seen"], **{a: loaded[a]["seen"]["effect"].astype(np.float64) for a in arms}}
    cross_effects = {pc.RIDGE: ridge["cross"]}
    for arm in arms:
        cross_effects[arm] = mk._crossfit(seen_effects[arm], [f["effect"].astype(np.float64) for f in loaded[arm]["folds"]],
                                          eligible, folds)
    seen_dirs = np.repeat(ridge["seen_common"][None], pack.ids.size, axis=0)
    cross_dirs = seen_dirs.copy()
    for fold in range(ov.MASK_FOLDS):
        cross_dirs[eligible & (folds == fold)] = ridge["fold_common"][fold]
    table = contrast_table(arms)
    seen, seen_vec = _grade(pack, seen_effects, seen_dirs, eligible, table)
    masked, masked_vec = _grade(pack, cross_effects, cross_dirs, eligible, table)

    per_fold, per_fold_vec = {}, {}
    for fold in range(ov.MASK_FOLDS):
        inside = eligible & (folds == fold)
        fold_dirs = np.repeat(ridge["fold_common"][fold][None], pack.ids.size, axis=0)
        effects = {pc.RIDGE: ridge["per_fold"][fold],
                   **{a: loaded[a]["folds"][fold]["effect"].astype(np.float64) for a in arms}}
        block, vectors = _grade(pack, effects, fold_dirs, inside, table, bootstrap=False)
        block["masked_candidates"] = int(inside.sum())
        block["enters_primary"] = bool(inside.sum() >= pc.PER_FOLD_PRIMARY_FLOOR)
        per_fold[str(fold)] = block
        per_fold_vec.update({f"{fold}|{k}": v for k, v in vectors.items()})

    seen_diag, seen_delta = candidate_diagnostics(pack, seen_effects, seen_dirs, rows)
    masked_diag, masked_delta = candidate_diagnostics(pack, cross_effects, cross_dirs, rows)
    fits_summary, memory_summary = {}, {}
    for arm in arms:
        records = loaded[arm]["records"]
        fits_summary[arm] = {name: {"selected_epoch": r["selection"]["selected_epoch"],
                                    "best_validation_cosine": r["selection"]["best_validation_cosine"],
                                    "ridge_same_split_validation_cosine": r.get("ridge_same_split_validation_cosine"),
                                    "programs_training_codes": r["programs_training_codes"],
                                    "programs_predicted_codes": r["programs_predicted_codes"],
                                    "router_top1_share_predicted": r.get("router_top1_share_predicted"),
                                    "firewall_final": r["firewall"]["final"]}
                             for name, r in records.items()}
        fits_summary[arm]["retrieval_quality"] = records[pc.SEEN_POOL].get("retrieval_quality")
        if arm in pc.ANCHORED:
            references = {pc.SEEN_POOL: ridge["seen"], **{str(f): ridge["per_fold"][f] for f in range(ov.MASK_FOLDS)}}
            files = {pc.SEEN_POOL: loaded[arm]["seen"], **{str(f): loaded[arm]["folds"][f] for f in range(ov.MASK_FOLDS)}}
            fits_summary[arm]["anchor_vs_ridge_unit_max_abs"] = {
                name: float(np.abs(_unit(files[name]["anchor"].astype(np.float64)) - _unit(references[name])).max())
                for name in files}
            fits_summary[arm]["correction_scale"] = {name: r["final"].get("correction_scale") for name, r in records.items()}
        if pc.USES_MEMORY[arm]:
            memory_summary[arm] = {}
            sources = {"SEEN": [loaded[arm]["seen"]] * ov.MASK_FOLDS, "MASKED": loaded[arm]["folds"]}
            for regime, files in sources.items():
                block = {}
                for key in ("memory_weight", "attention_entropy", "neighbour_knowledge_similarity",
                            "neighbour_effect_similarity_weighted", "neighbour_specific_similarity_weighted"):
                    values = []
                    for fold in range(ov.MASK_FOLDS):
                        source = files[fold]
                        position = np.searchsorted(source["rows"], rows[folds[rows] == fold])
                        values.append(source[f"memory|{key}"][position])
                    values = np.concatenate(values)
                    block[key] = {q: float(np.quantile(values, float(q))) for q in ("0.1", "0.5", "0.9")}
                memory_summary[arm][regime] = block
    record = {"held": held, "arms": list(arms), "training": list(setup["training"]),
              "pool": int(pack.ids.size), "eligible_candidates": int(eligible.sum()),
              "SEEN": seen, "MASKED": masked, "MASKED_PER_FOLD": per_fold,
              "homogeneity": {arm: homogeneity([f["effect"] for f in loaded[arm]["folds"]], rows, ridge["fold_common"])
                              for arm in arms},
              "candidate_diagnostics": {"SEEN": seen_diag, "MASKED": masked_diag},
              "fits": fits_summary, "memory": memory_summary}
    identities = np.asarray(pack.ids, dtype=str)
    tag = "_".join(arms)
    np.savez(output / f"vectors_{held}__{tag}.npz",
             query_identities=identities[np.asarray(pack.self_positions)], candidate_identities=identities[rows],
             **{f"SEEN|{k}": v for k, v in seen_vec.items()}, **{f"MASKED|{k}": v for k, v in masked_vec.items()},
             **{f"MASKED_PER_FOLD|{k}": v for k, v in per_fold_vec.items()},
             **{f"CANDIDATE|SEEN|{k}": v for k, v in seen_delta.items()},
             **{f"CANDIDATE|MASKED|{k}": v for k, v in masked_delta.items()})
    return record


def stress_stage(packs: dict, fits: Path, arm: str, output: Path) -> dict:
    held, training = pc.STRESS_CONTEXT, ov.NATURAL_TRAINING
    setup = held_setup(packs, held, training)
    pack, unseen = setup["pack"], setup["unseen"]
    full = es.views_pool(packs, training)
    raw, unit = dca.fit_both(full, training)
    prediction = np.load(fits / arm / held / pc.SEEN_POOL / "prediction.npz", allow_pickle=False)
    effects = {pc.RIDGE: unit.effect(pack.features, pack.present), arm: prediction["effect"].astype(np.float64),
               "A0_RAW": raw.effect(pack.features, pack.present)}
    directions = np.repeat(_unit(mk.intercept_effect(unit))[None], pack.ids.size, axis=0)
    table = {**contrast_table((arm,)), "RIDGE_minus_A0": (pc.RIDGE, "A0_RAW")}
    block, _ = _grade(pack, effects, directions, unseen, table)
    gate = {"unseen_candidates": int(unseen.sum()),
            "A1_minus_A0_median_error": abs(block["fused"]["contrasts"]["RIDGE_minus_A0"]["MBRU"]["median"]
                                            - rc.RECORDED_STRESS["A1_minus_A0"])}
    gate["passes"] = bool(gate["unseen_candidates"] == 6255 and gate["A1_minus_A0_median_error"] <= rc.MEDIAN_TOLERANCE)
    if not gate["passes"]:
        raise RuntimeError(f"stress provenance gate failed: {gate}")
    return {"held": held, "training": list(training), "stratum": "UNSEEN_CANDIDATE", "provenance_gate": gate,
            "ridge_penalty": unit.penalty, "contrasts": {mode: block[mode]["contrasts"] for mode in pc.MODES}}


def _pooled(vectors: dict, better: str, worse: str) -> dict:
    values, labels = [], []
    for name in pc.PRIMARY_CONTEXTS:
        identities = np.asarray(vectors[name]["query_identities"], dtype=str)
        values.append(vectors[name][better] - vectors[name][worse])
        labels.append(np.concatenate([identities, identities]))
    return st.cluster_bootstrap_labels(np.concatenate(values), np.concatenate(labels))


def _pooled_candidates(vectors: dict, key: str) -> dict:
    values = np.concatenate([vectors[n][key] for n in pc.PRIMARY_CONTEXTS])
    labels = np.concatenate([vectors[n]["candidate_identities"] for n in pc.PRIMARY_CONTEXTS])
    return st.cluster_bootstrap_labels(values, labels)


def summarize_stage(output: Path, arms: tuple, replicate: Path | None) -> dict:
    tag = "_".join(arms)
    records = {n: json.loads((output / f"context_{n}__{tag}.json").read_text()) for n in pc.PRIMARY_CONTEXTS}
    vectors = {n: dict(np.load(output / f"vectors_{n}__{tag}.npz", allow_pickle=False)) for n in pc.PRIMARY_CONTEXTS}
    table = contrast_table(arms)
    pooled = {regime: {mode: {label: {metric: _pooled(vectors, f"{regime}|{mode}|{b}|{metric}",
                                                      f"{regime}|{mode}|{w}|{metric}")
                                      for metric in HEADLINE + COMPANION[:3]}
                              for label, (b, w) in table.items()}
                       for mode in pc.MODES}
              for regime in pc.REGIMES}
    candidates = {regime: {f"{arm}|{key}": _pooled_candidates(vectors, f"CANDIDATE|{regime}|{arm}|{key}")
                           for arm in arms for key in ("full_cosine", "specific_cosine", "common_share")}
                  for regime in pc.REGIMES}
    per_fold = {}
    for arm in arms:
        values, labels, used = [], [], []
        for name in pc.PRIMARY_CONTEXTS:
            identities = np.asarray(vectors[name]["query_identities"], dtype=str)
            for fold, block in records[name]["MASKED_PER_FOLD"].items():
                if not block.get("enters_primary"):
                    continue
                key = f"MASKED_PER_FOLD|{fold}|fused|"
                values.append(vectors[name][f"{key}{arm}|MBRU"] - vectors[name][f"{key}{pc.RIDGE}|MBRU"])
                labels.append(np.concatenate([identities, identities]))
                used.append(f"{name}:{fold}")
        per_fold[arm] = {**st.cluster_bootstrap_labels(np.concatenate(values), np.concatenate(labels)),
                         "folds_used": used}
    result = {"schema": pc.SCHEMA, "mission": pc.MISSION, "arms": list(arms), "read_anchor_G_check": READ_ANCHOR_G_CHECK,
              "pooled": pooled, "pooled_candidates": candidates, "pooled_per_fold": per_fold,
              "homogeneity": {n: records[n]["homogeneity"] for n in pc.PRIMARY_CONTEXTS}}
    full = next((arm for arm in arms if arm in pc.FULL_ARMS), None)
    if full is not None:
        replicate_p1 = None
        if replicate is not None:
            other = json.loads(replicate.read_text())
            replicate_p1 = dd.primary_passes(other["decision_input"])
        decision_input = dd.decision_input(pooled, candidates, per_fold, records, full)
        result["decision_input"] = decision_input
        result["decision"] = dd.decide(decision_input, replicate_p1)
    necessity = {}
    labels = [f"{f}_minus_{o}" for f, others in pc.ABLATION_OF.items() for o in others]
    labels += [f"{a}_minus_{pc.RIDGE}" for a in arms if a not in pc.FULL_ARMS]
    for label in labels:
        if label in pooled["MASKED"]["fused"]:
            entry = pooled["MASKED"]["fused"][label]["MBRU"]
            necessity[label] = {k: entry[k] for k in ("median", "median_ci95", "state")}
    result["module_necessity"] = necessity
    result["runtime"] = _runtime()
    return result


def smoke_stage(packs: dict, held: str, arm: str, epochs: int, output: Path) -> dict:
    """A short selection run with every health statistic, plus firewall and identity probes."""
    setup = held_setup(packs, held)
    pack = setup["pack"]
    excluded = pool_exclusion(setup, "0")
    fold_pool = dt.assemble(packs, setup["training"], excluded)
    pool = dt.assemble(packs, setup["training"])
    history = []
    selection = tr.train(pool, arm, pc.SEED, validation=dt.validation_mask(pool.identities),
                         max_epochs=epochs, log=lambda e: (history.append(e), print(json.dumps(
                             {k: (round(v, 5) if isinstance(v, float) else v) for k, v in e.items()
                              if k != "gradient_norms_last_step"} | {"grad": {k: round(v, 4) for k, v in
                                                                              e["gradient_norms_last_step"].items()}}),
                             flush=True)))
    model, std, memory = selection.model, selection.standardizer, selection.memory
    prediction = tr.predict(model, std, memory, pack.features, pack.present)
    # identity probe: the same features under a permuted candidate order give the same rows
    order = np.random.default_rng(0).permutation(pack.ids.size)
    permuted = tr.predict(model, std, memory, pack.features[order], pack.present[order])
    duplicate = tr.predict(model, std, memory, np.repeat(pack.features[prediction["rows"][:1]], 2, axis=0),
                           np.ones(2, dtype=bool))
    probes = {"permutation_max_abs_difference": float(np.abs(permuted["effect"] - prediction["effect"][order]).max()),
              "duplicate_features_max_abs_difference": float(np.abs(duplicate["effect"][0] - duplicate["effect"][1]).max()),
              "forward_signature": "features, presence, memory, exclude, encoded (no identity argument)"}
    # self exclusion during training: a training batch never retrieves its own identity
    if memory is not None:
        model.train()
        ids = torch.arange(min(512, selection.pool.identities.size), device=tr.DEVICE)
        with torch.no_grad():
            out = model(memory.features[ids], memory.presence[ids], memory, ids)
        probes["self_retrieved_in_training"] = int((out["neighbours"] == ids[:, None]).any(dim=1).sum())
        with torch.no_grad():
            unmasked = model(memory.features[ids], memory.presence[ids], memory, None)
        probes["self_retrieved_without_exclusion_share"] = float(
            (unmasked["neighbours"] == ids[:, None]).any(dim=1).float().mean())
    masked = set(excluded.tolist())
    firewall = {"fold0_masked": len(masked),
                "fold0_masked_in_fold_pool": len(masked & set(fold_pool.identities.tolist())),
                "fold0_masked_in_seen_pool": len(masked & set(pool.identities.tolist())),
                "fold0_pool_rows": fold_pool.rows, "seen_pool_rows": pool.rows}
    health = {"programs_training_codes": tr.program_usage(selection),
              "programs_predicted_codes": tr.code_statistics(prediction["codes"][prediction["rows"]])}
    if "gate" in prediction:
        health["memory_weight_quantiles"] = {q: float(np.quantile(1 - prediction["gate"], float(q)))
                                             for q in ("0.05", "0.5", "0.95")}
    if "attention" in prediction:
        entropy = _attention_entropy(prediction["attention"].astype(np.float64))
        health["attention_entropy_quantiles"] = {q: float(np.quantile(entropy, float(q))) for q in ("0.05", "0.5", "0.95")}
        health["attention_entropy_uniform"] = float(np.log(pc.NEIGHBOURS))
        health["attention_max_weight_median"] = float(np.median(prediction["attention"].max(axis=1)))
    if "router_probabilities" in prediction:
        top = prediction["router_probabilities"].argmax(axis=1)
        health["router_top1_share"] = (np.bincount(top, minlength=pc.EXPERTS) / top.size).tolist()
    health["ridge_same_split_validation_cosine"] = ridge_same_split(selection)
    health["best_validation_cosine"] = selection.record["best_validation_cosine"]
    health["selected_epoch"] = selection.record["selected_epoch"]
    health["all_losses_finite"] = bool(all(np.isfinite(v) for e in history for k, v in e.items() if k.startswith("loss_")))
    return {"held": held, "arm": arm, "epochs": epochs, "probes": probes, "firewall": firewall,
            "health": health, "history": history}


def main() -> int:
    parser = argparse.ArgumentParser(description=pc.MISSION)
    parser.add_argument("--stage", required=True, choices=("smoke", "fit", "ridge", "grade", "stress", "summarize"))
    parser.add_argument("--held", default=None)
    parser.add_argument("--arm", default=pc.FULL)
    parser.add_argument("--arms", default=pc.FULL, help="comma separated, for grade and summarize")
    parser.add_argument("--pool", default=pc.SEEN_POOL)
    parser.add_argument("--seed", type=int, default=pc.SEED)
    parser.add_argument("--epochs", type=int, default=30, help="smoke only")
    parser.add_argument("--packs", default=None)
    parser.add_argument("--c018", default=None)
    parser.add_argument("--fits", default=None, help="the fits directory of this run")
    parser.add_argument("--replicate", default=None, help="summary JSON of the seed replicate")
    parser.add_argument("--output-dir", dest="output_dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    arms = tuple(args.arms.split(","))
    if args.stage == "summarize":
        target = output / f"summary__{'_'.join(arms)}.json"
        _refuse(target)
        record = summarize_stage(output, arms, Path(args.replicate) if args.replicate else None)
        target.write_text(json.dumps(record, indent=2, sort_keys=True, default=float) + "\n")
        print(json.dumps(record.get("decision", {}), indent=1, default=float))
        print(json.dumps(record["module_necessity"], indent=1, default=float))
        return 0
    packs = {name: hn.load_pack(name, args.packs) for name in fc.CONTEXTS}
    if args.stage == "fit":
        held = args.held or pc.STRESS_CONTEXT
        training = ov.NATURAL_TRAINING if held == pc.STRESS_CONTEXT else None
        directory = output / args.arm / held / args.pool
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "record.json"
        _refuse(target)
        record = fit_stage(packs, held, args.arm, args.pool, args.seed, directory, training)
    elif args.stage == "smoke":
        target = output / f"smoke_{args.held}_{args.arm}.json"
        _refuse(target)
        record = smoke_stage(packs, args.held, args.arm, args.epochs, output)
    elif args.stage == "ridge":
        target = output / f"ridge_{args.held}.json"
        _refuse(target)
        record = ridge_stage(packs, args.held, Path(args.c018), output)
    elif args.stage == "grade":
        target = output / f"context_{args.held}__{'_'.join(arms)}.json"
        _refuse(target)
        record = grade_stage(packs, args.held, arms, Path(args.fits), output)
    else:
        target = output / f"stress_{args.arm}.json"
        _refuse(target)
        record = stress_stage(packs, Path(args.fits), args.arm, output)
    record.update({"schema": pc.SCHEMA, "read_anchor_G_check": READ_ANCHOR_G_CHECK, "runtime": _runtime()})
    target.write_text(json.dumps(record, indent=2, sort_keys=True, default=float) + "\n")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
