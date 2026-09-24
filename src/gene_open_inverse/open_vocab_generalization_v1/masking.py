#!/usr/bin/env python3
"""Phase B: controlled candidate masking, the decisive experiment.

A natural unseen candidate differs from a seen one in library membership, in assay
history and in everything correlated with those.  A masked candidate differs from
itself.  So this phase manufactures the response-unseen condition rather than
inheriting it: the candidate identity, its static knowledge, the evaluation pool, the
queries, the base ranker and the graded utility are all held fixed, and the single
thing that changes is whether that identity's measured responses were available to
fit the effect predictor.

Two designs, and they are not equivalent.

``cross_context`` holds out a whole context, trains on the other three, and masks the
candidates in all three.  Neither arm has ever seen the held context's responses, so
neither can memorize the quantity it is graded on.  This is the primary design.

``within_context`` trains on one context and evaluates masked candidates in that same
context.  Its SEEN arm fits on the evaluated candidate's own response, which is also
the response the utility is built from, so ``SEEN - MASKED`` there is an upper bound
and its interaction with leverage is mechanically confounded: a high-leverage row
influences its own prediction more.  It is reported as a within-context reference and
never as the estimate of record.

Everything the removal legally touches is refitted.  The response basis and the ridge
are rebuilt inside the masked pool, because a deployment in which the candidate was
never measured is a deployment in which it never entered the basis either.  Reusing
the seen arm's basis would leak the masked responses through the very object the
protocol says must be rebuilt.
"""
from __future__ import annotations

import argparse
import hashlib
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
from .natural import paired_all, quantile_strata


def fold_assignment(ids: np.ndarray, folds: int = ov.MASK_FOLDS,
                    namespace: str = ov.MASK_NAMESPACE) -> np.ndarray:
    """A candidate's masking fold, from its symbol alone.

    Deterministic, outcome independent and stable across contexts, so the same
    identity is masked in the same fold everywhere it appears.  Nothing about a
    response, a feature value or a result enters it.
    """
    digest = [hashlib.sha256(f"{namespace}:{value}".encode()).digest()
              for value in np.asarray(ids, dtype=str).tolist()]
    return np.asarray([int.from_bytes(value[:8], "big") % folds for value in digest],
                      dtype=np.int64)


def masked_training_pool(packs: dict, names: tuple[str, ...],
                         excluded: np.ndarray) -> dict:
    """The pooled training rows of ``names`` with every row of ``excluded`` removed.

    Removal is by identity and applies in every named context at once, so a masked
    candidate has no response supervision anywhere in the atlas rather than in one
    context of it.
    """
    excluded = set(np.asarray(excluded, dtype=str).tolist())
    features, responses, present, keys = [], [], [], []
    for name in names:
        pack = packs[name]
        rows = np.flatnonzero(pack.fit_mask)
        keep = np.asarray([value not in excluded for value in pack.ids[rows].tolist()], dtype=bool)
        rows = rows[keep]
        features.append(pack.features[rows])
        responses.append(ev.consensus(pack.responses[:, rows]))
        present.append(pack.present[rows])
        keys.append(np.asarray([f"{name}|{value}" for value in pack.ids[rows].tolist()], dtype=str))
    return {"features": np.concatenate(features), "responses": np.concatenate(responses),
            "present": np.concatenate(present), "keys": np.concatenate(keys),
            "contexts": tuple(names)}


def intercept_effect(fitted) -> np.ndarray:
    """What the predictor returns for a candidate at the training feature centre.

    This is the common response direction the ridge carries in its intercept.  Two
    predictors whose intercepts point in different directions cannot have their
    per-candidate effects mixed into one score matrix, because the cosine against a
    query would then carry a predictor-dependent offset rather than a candidate one.
    """
    return np.asarray(fitted.basis.reconstruct(fitted.path.target_mean[None, :])[0],
                      dtype=np.float64)


def candidate_specific_share(effect: np.ndarray, intercept: np.ndarray,
                             rows: np.ndarray) -> float:
    """How much of a predictor's output is candidate specific rather than the common part.

    ``ev.effect_cosine`` unit normalizes each predicted effect, so a predictor whose
    output is mostly the shared intercept produces cosines with a high mean and a small
    spread, and one that shrinks less produces the opposite.  That ratio, not the
    intercept's direction, is what makes two predictors' per-candidate columns
    incomparable when they are mixed into a single score row.
    """
    values = np.asarray(effect, dtype=np.float64)[rows]
    common = np.asarray(intercept, dtype=np.float64)
    scale = max(float(np.linalg.norm(common)), 1e-30)
    return float(np.median(np.linalg.norm(values - common[None, :], axis=1) / scale))


def homogeneity(predictors: list, effects: list | None = None,
                rows: np.ndarray | None = None) -> dict:
    """Can the five fold predictors' per-candidate columns be mixed into one score row.

    The direction test alone cannot fail.  ``path.target_mean`` is a penalty-free,
    weight-free mean of the training targets, and five pools sharing four fifths of their
    rows always agree on its direction, so a gate built only on that verifies nothing
    about the fitted maps.  What actually has to agree is the shrinkage: the ridge penalty
    each fold selected independently, and the resulting share of each prediction that is
    candidate specific rather than the common response.  Both are checked here.

    The mission's primary quantity does not depend on this gate passing.
    ``EFFECT_MASKED`` and ``EFFECT_SIZE_CONTROL`` are built the same way out of the same
    five predictors, permuted by one fold, so any mixing distortion is common to both and
    cancels in their contrast.  The gate governs the secondary ``SEEN minus MASKED``,
    whose two arms are not built alike.
    """
    directions = np.stack([intercept_effect(item) for item in predictors])
    unit = directions / np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-30)
    similarity = unit @ unit.T
    off = similarity[~np.eye(len(predictors), dtype=bool)]
    penalties = [float(item.penalty) for item in predictors]
    record = {"minimum_pairwise_cosine": float(off.min()),
              "mean_pairwise_cosine": float(off.mean()),
              "floor": ov.CROSSFIT_HOMOGENEITY_FLOOR,
              "selected_penalties": penalties,
              "penalties_agree": bool(len(set(penalties)) == 1),
              "scale_tolerance": ov.CROSSFIT_SCALE_TOLERANCE}
    if effects is not None and rows is not None and rows.size:
        shares = [candidate_specific_share(effect, intercept_effect(item), rows)
                  for effect, item in zip(effects, predictors)]
        ratio = max(shares) / max(min(shares), 1e-30)
        record["candidate_specific_share"] = shares
        record["candidate_specific_share_ratio"] = float(ratio)
        record["scale_agrees"] = bool(ratio <= ov.CROSSFIT_SCALE_TOLERANCE)
    record["passes"] = bool(off.min() >= ov.CROSSFIT_HOMOGENEITY_FLOOR
                            and record["penalties_agree"]
                            and record.get("scale_agrees", True))
    return record


def reconstruction_cosine(effect: np.ndarray, responses: np.ndarray,
                          rows: np.ndarray) -> float:
    """``cos(r_hat, r_bar)`` on one candidate set.  A mechanism diagnostic, never a verdict."""
    truth = ev.consensus(responses[:, rows])
    predicted = np.asarray(effect, dtype=np.float64)[rows]
    norm = np.linalg.norm(predicted, axis=1) * np.linalg.norm(truth, axis=1)
    return float(np.median(np.einsum("ij,ij->i", predicted, truth) / np.maximum(norm, 1e-30)))


def query_cluster_bootstrap(difference: np.ndarray, replicates: int = 10000,
                            seed: int = 20260916) -> dict:
    """The interval the pre-registration asks for, on the layout the evaluator produces.

    ``evaluate_arms`` concatenates the two batch-disjoint query views into one vector of
    length ``2Q``, so entry ``i`` and entry ``i + Q`` are the same query identity seen
    twice.  The frozen ``cluster_bootstrap`` resamples that vector entry by entry, which
    treats the two views of one query as two independent clusters and makes every
    interval in the four-context programme narrower than the cluster unit its own
    docstring names.  The frozen function is not changed here, because changing it would
    make these numbers incomparable with C-013 and C-014.  This one resamples query
    identities and carries both views of a drawn identity together, and is reported
    beside the frozen interval so a reader can see whether any conclusion depends on it.
    """
    difference = np.asarray(difference, dtype=np.float64)
    if difference.size % 2:
        raise ValueError("the evaluator emits two views per query, so the vector is even")
    half = difference.size // 2
    paired = np.stack([difference[:half], difference[half:]], axis=1)
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, half, size=(replicates, half))
    resampled = paired[draws].reshape(replicates, -1)
    medians = np.median(resampled, axis=1)
    return {"clusters": int(half), "replicates": int(replicates), "seed": int(seed),
            "median_ci95": [float(np.quantile(medians, 0.025)),
                            float(np.quantile(medians, 0.975))],
            "fraction_of_replicates_above_zero": float((medians > 0).mean())}


def evaluate_mask(pack, score_by_arm: dict, mask: np.ndarray, reference: str = "BASE") -> dict:
    block = ev.evaluate_arms(score_by_arm, pack.utility, pack.ids, pack.self_positions, mask)
    if "skipped" in block:
        return block
    vectors = block.pop("_vectors")
    block["paired_vs_base"] = {name: paired_all(vectors, name, reference)
                               for name in vectors if name != reference}
    for name, entry in block["paired_vs_base"].items():
        entry["MBRU"]["query_cluster_bootstrap"] = query_cluster_bootstrap(
            vectors[name]["MBRU"] - vectors[reference]["MBRU"])
    # The quantity of record for this mission, and the two halves it decomposes into.
    contrasts = {"exposure_value": ("EFFECT_SEEN", "EFFECT_MASKED"),
                 "exposure_value_size_controlled": ("EFFECT_SIZE_CONTROL", "EFFECT_MASKED"),
                 "atlas_shrinkage_cost": ("EFFECT_SEEN", "EFFECT_SIZE_CONTROL")}
    for label, (better, worse) in contrasts.items():
        if better in vectors and worse in vectors:
            block[label] = paired_all(vectors, better, worse)
            block[label]["MBRU"]["query_cluster_bootstrap"] = query_cluster_bootstrap(
                vectors[better]["MBRU"] - vectors[worse]["MBRU"])
    return block


def _fold_record(pack, score_by_arm: dict, mask: np.ndarray, leverage: np.ndarray,
                 table: sp.SupportTable | None) -> dict:
    block = evaluate_mask(pack, score_by_arm, mask)
    rows = np.flatnonzero(mask)
    block["masked_candidates"] = int(rows.size)
    block["enters_primary"] = bool(rows.size >= ov.MIN_PRIMARY_POOL)
    block["leverage_median"] = float(np.median(leverage[rows])) if rows.size else None
    if table is not None and rows.size:
        block["coverage"] = {pattern: int((table.coverage[rows] == pattern).sum())
                             for pattern in ov.COVERAGE_PATTERNS}
    return block


def _support_strata(pack, score_by_arm: dict, eligible: np.ndarray,
                    leverage: np.ndarray) -> dict:
    """Section 16: the same masked candidates split into leverage terciles.

    The leverage axis is the one computed against the full training atlas, so the
    strata are identical in both arms.  Splitting each arm by its own leverage would
    make the two arms rank different candidates and the comparison would stop being
    paired.
    """
    rows = np.flatnonzero(eligible)
    strata = quantile_strata(leverage[rows], ov.MASK_SUPPORT_STRATA, ov.MASK_STRATUM_NAMES)
    record = {}
    for name, inside in strata.items():
        mask = np.zeros(eligible.size, dtype=bool)
        mask[rows[inside]] = True
        block = evaluate_mask(pack, score_by_arm, mask)
        block["count"] = int(mask.sum())
        block["leverage_range"] = [float(leverage[rows[inside]].min()),
                                   float(leverage[rows[inside]].max())]
        record[name] = block
    return record


def _fit_folds(packs: dict, training: tuple[str, ...], pack, eligible: np.ndarray,
               folds: np.ndarray) -> tuple[list, list, dict]:
    """One predictor per masking fold, plus its predicted effect over the whole pool.

    All five are fitted before anything is evaluated, because fold ``M``'s size control
    is fold ``M + 1``'s masked predictor and nothing should be refitted to serve as a
    control for something else.
    """
    fitted, effects, record = [], [], {}
    for fold in range(ov.MASK_FOLDS):
        inside = eligible & (folds == fold)
        pool = masked_training_pool(packs, training, pack.ids[inside])
        model = pd.fit(pool["features"], pool["responses"], pool["present"],
                       pool["keys"], training)
        fitted.append(model)
        effects.append(model.effect(pack.features, pack.present))
        record[str(fold)] = {"masked_identities": int(inside.sum()),
                             "rows": model.rows, "identities": model.identities,
                             "penalty": model.penalty,
                             "held_out_gene_space_cosine": model.held_out_cosine,
                             "explained_energy": model.basis.explained}
    return fitted, effects, record


def _crossfit(seen_effect: np.ndarray, effects: list, eligible: np.ndarray,
              folds: np.ndarray, offset: int = 0) -> np.ndarray:
    """Each eligible candidate takes the effect of the predictor ``offset`` folds along.

    The starting point is the seen arm's own prediction rather than zero.  A candidate
    outside the eligible set is not part of the manipulation: no training context
    measured it, or it has no static modality, so its exposure is the same in both arms
    and its predicted effect must be too.  Filling those columns with zero instead would
    leave the masked arm's effect row mostly zeros while the seen arm's stays dense, and
    the frozen fusion standardizes that row over the whole pool before the evaluation
    restricts it, so the two arms would enter the fusion at different effective weights
    and the contrast would be a fusion artifact rather than an exposure result.
    """
    output = np.array(seen_effect, dtype=np.float64, copy=True)
    for fold in range(ov.MASK_FOLDS):
        inside = eligible & (folds == fold)
        output[inside] = effects[(fold + offset) % ov.MASK_FOLDS][inside]
    return output


def cross_context(packs: dict, held: str, adjacency: dict) -> dict:
    """Section 12: held context fixed, candidate response exposure in the training atlas varied.

    Three effect arms, not two.  Removing a candidate's rows also removes rows, and this
    programme has already established that atlas size is the primary driver of effect
    distillation quality, so ``SEEN - MASKED`` alone cannot separate "this candidate was
    never measured" from "the atlas is a fifth smaller".  The size control removes a
    different fold of the same size instead, which is exactly the masked predictor of the
    next fold, so it costs no additional fit and it is fixed by the same hash rather than
    by a draw.
    """
    training = tuple(name for name in fc.CONTEXTS if name != held)
    pack = packs[held]
    full = hn.training_pool(packs, training)
    seen_predictor = pd.fit(full["features"], full["responses"], full["present"],
                            full["keys"], training)
    supervised = np.unique([key.split("|", 1)[1] for key in full["keys"].tolist()])

    eligible = np.isin(pack.ids, supervised) & np.asarray(pack.present, dtype=bool)
    folds = fold_assignment(pack.ids)

    keep = np.flatnonzero(np.asarray(full["present"], dtype=bool))
    training_features, training_identities = sp.unique_training(
        full["features"][keep], full["keys"][keep])
    table = sp.build(pack.ids, pack.features, seen_predictor.path, seen_predictor.penalty,
                     training_features, training_identities, adjacency)

    seen_effect = seen_predictor.effect(pack.features, pack.present)
    fitted, effects, fit_record = _fit_folds(packs, training, pack, eligible, folds)

    fold_blocks = {}
    for fold in range(ov.MASK_FOLDS):
        inside = eligible & (folds == fold)
        control = (fold + 1) % ov.MASK_FOLDS
        arms = {"BASE": pack.base,
                "EFFECT_SEEN": hn.effect_arms(pack, seen_effect),
                "EFFECT_MASKED": hn.effect_arms(pack, effects[fold]),
                "EFFECT_SIZE_CONTROL": hn.effect_arms(pack, effects[control])}
        block = _fold_record(pack, arms, inside, table.leverage, table)
        rows = np.flatnonzero(inside)
        block["predictor"] = fit_record[str(fold)]
        block["size_control_fold"] = control
        block["size_control_predictor"] = fit_record[str(control)]
        block["training_rows_removed"] = int(seen_predictor.rows - fitted[fold].rows)
        block["size_control_rows_removed"] = int(seen_predictor.rows - fitted[control].rows)
        block["reconstruction_cosine"] = {
            "EFFECT_SEEN": reconstruction_cosine(seen_effect, pack.responses, rows),
            "EFFECT_MASKED": reconstruction_cosine(effects[fold], pack.responses, rows),
            "EFFECT_SIZE_CONTROL": reconstruction_cosine(effects[control], pack.responses, rows)}
        fold_blocks[str(fold)] = block

    crossfit_effect = _crossfit(seen_effect, effects, eligible, folds, offset=0)
    crossfit_control = _crossfit(seen_effect, effects, eligible, folds, offset=1)
    arms = {"BASE": pack.base,
            "EFFECT_SEEN": hn.effect_arms(pack, seen_effect),
            "EFFECT_MASKED": hn.effect_arms(pack, crossfit_effect),
            "EFFECT_SIZE_CONTROL": hn.effect_arms(pack, crossfit_control)}
    crossfit = evaluate_mask(pack, arms, eligible)
    crossfit["masked_candidates"] = int(eligible.sum())
    crossfit["homogeneity"] = homogeneity(
        [seen_predictor] + fitted, [seen_effect] + effects, np.flatnonzero(eligible))
    crossfit["primary_quantity_is_immune"] = (
        "EFFECT_MASKED and EFFECT_SIZE_CONTROL mix the same five predictors permuted by "
        "one fold, so a mixing distortion is common to both and cancels in their "
        "contrast; the gate governs SEEN minus MASKED, whose arms are not built alike")
    rows = np.flatnonzero(eligible)
    crossfit["reconstruction_cosine"] = {
        "EFFECT_SEEN": reconstruction_cosine(seen_effect, pack.responses, rows),
        "EFFECT_MASKED": reconstruction_cosine(crossfit_effect, pack.responses, rows),
        "EFFECT_SIZE_CONTROL": reconstruction_cosine(crossfit_control, pack.responses, rows)}
    crossfit["support_strata"] = _support_strata(pack, arms, eligible, table.leverage)

    cosines = {f"{label}_{view}": ev.effect_cosine(pack.query_responses, source, view)
               for label, source in (("seen", seen_effect), ("masked", crossfit_effect),
                                     ("size_control", crossfit_control))
               for view in (0, 1)}

    return {"design": "cross_context", "held": held, "training": list(training),
            "_cosines": cosines, "_eligible": eligible, "_folds": folds,
            "eligible_candidates": int(eligible.sum()),
            "queries": int(pack.query_rows.size), "pool": int(pack.ids.size),
            "base_training_identity_overlap": int(np.isin(
                pack.ids[eligible], np.asarray(supervised, dtype=str)).sum()),
            "seen_predictor": {"rows": seen_predictor.rows,
                               "identities": seen_predictor.identities,
                               "penalty": seen_predictor.penalty,
                               "held_out_gene_space_cosine": seen_predictor.held_out_cosine,
                               "explained_energy": seen_predictor.basis.explained},
            "fold_predictors": fit_record,
            "folds": fold_blocks, "crossfit": crossfit,
            "support_table": table}


def within_context(packs: dict, name: str, adjacency: dict) -> dict:
    """Section 10: training and evaluation in the same context, exposure varied.

    Reported with the caveat in the contract attached: this design's seen arm fits on the
    evaluated candidate's own graded response, so its exposure value is an upper bound.
    """
    pack = packs[name]
    full = hn.training_pool(packs, (name,))
    seen_predictor = pd.fit(full["features"], full["responses"], full["present"],
                            full["keys"], (name,))
    supervised = np.unique([key.split("|", 1)[1] for key in full["keys"].tolist()])
    eligible = np.isin(pack.ids, supervised) & np.asarray(pack.present, dtype=bool)
    folds = fold_assignment(pack.ids)

    keep = np.flatnonzero(np.asarray(full["present"], dtype=bool))
    training_features, training_identities = sp.unique_training(
        full["features"][keep], full["keys"][keep])
    table = sp.build(pack.ids, pack.features, seen_predictor.path, seen_predictor.penalty,
                     training_features, training_identities, adjacency)

    seen_effect = seen_predictor.effect(pack.features, pack.present)
    fitted, effects, fit_record = _fit_folds(packs, (name,), pack, eligible, folds)
    fold_blocks = {}
    for fold in range(ov.MASK_FOLDS):
        inside = eligible & (folds == fold)
        control = (fold + 1) % ov.MASK_FOLDS
        arms = {"BASE": pack.base,
                "EFFECT_SEEN": hn.effect_arms(pack, seen_effect),
                "EFFECT_MASKED": hn.effect_arms(pack, effects[fold]),
                "EFFECT_SIZE_CONTROL": hn.effect_arms(pack, effects[control])}
        block = _fold_record(pack, arms, inside, table.leverage, table)
        block["predictor"] = fit_record[str(fold)]
        block["size_control_fold"] = control
        fold_blocks[str(fold)] = block
    return {"design": "within_context", "context": name,
            "caveat": ov.WITHIN_CONTEXT_CAVEAT,
            "eligible_candidates": int(eligible.sum()),
            "queries": int(pack.query_rows.size), "pool": int(pack.ids.size),
            "seen_predictor": {"rows": seen_predictor.rows,
                               "identities": seen_predictor.identities,
                               "penalty": seen_predictor.penalty,
                               "held_out_gene_space_cosine": seen_predictor.held_out_cosine},
            "fold_predictors": fit_record, "folds": fold_blocks}


def run(args: argparse.Namespace) -> dict:
    packs = {name: hn.load_pack(name, args.packs) for name in fc.CONTEXTS}
    adjacency_file = np.load(args.string_adjacency, allow_pickle=False)
    adjacency = {key: adjacency_file[key] for key in
                 ("genes", "degree", "edge_row", "edge_col", "edge_w")}

    record: dict = {
        "schema": "VCDESIGN_OPEN_VOCAB_V1_CONTROLLED_CANDIDATE_MASKING",
        "read_anchor_G_check": False,
        "mask_namespace": ov.MASK_NAMESPACE, "mask_folds": ov.MASK_FOLDS,
        "primary_design": ov.PRIMARY_MASKING_DESIGN,
        "primary_pool_floor": ov.MIN_PRIMARY_POOL,
        "primary_contexts": list(ov.MASK_PRIMARY_CONTEXTS),
        "secondary_contexts": list(ov.MASK_SECONDARY_CONTEXTS),
        "secondary_reason": ov.MASK_SECONDARY_REASON,
        "within_context_caveat": ov.WITHIN_CONTEXT_CAVEAT,
        "refit_rule": "the response basis and the ridge are both rebuilt inside the masked pool",
        "cross_context": {}, "within_context": {},
    }
    tables, cached = {}, {}
    for held in fc.CONTEXTS:
        block = cross_context(packs, held, adjacency)
        tables[held] = block.pop("support_table")
        cached[held] = {"cosines": block.pop("_cosines"), "eligible": block.pop("_eligible"),
                        "folds": block.pop("_folds")}
        record["cross_context"][held] = block
    for name in fc.CONTEXTS:
        record["within_context"][name] = within_context(packs, name, adjacency)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record["runtime"] = {"timestamp": datetime.now(timezone.utc).isoformat(),
                         "python": platform.python_version(), "numpy": np.__version__}
    output.write_text(json.dumps(record, indent=2, sort_keys=True, default=float) + "\n")
    for held, table in tables.items():
        np.savez(output.parent / f"support_{held}.npz", **table.as_dict())
        entry = cached[held]
        np.savez(output.parent / f"effect_cosine_{held}.npz", eligible=entry["eligible"],
                 folds=entry["folds"], **entry["cosines"])
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase B: controlled candidate masking")
    parser.add_argument("--packs", required=True)
    parser.add_argument("--string-adjacency", dest="string_adjacency", required=True)
    parser.add_argument("--output", required=True)
    record = run(parser.parse_args())

    for design in ("cross_context", "within_context"):
        print(f"\n=== {design} ===")
        for name, block in record[design].items():
            label = block.get("held", block.get("context"))
            tag = "primary" if label in ov.MASK_PRIMARY_CONTEXTS else "secondary"
            print(f"\n[{label}] {tag}  eligible {block['eligible_candidates']} "
                  f"queries {block['queries']}")
            for fold, entry in block["folds"].items():
                if "paired_vs_base" not in entry:
                    print(f"  fold {fold}: {entry.get('skipped')}")
                    continue
                vs = entry["paired_vs_base"]
                loss = entry["exposure_value"]["MBRU"]
                net = entry["exposure_value_size_controlled"]["MBRU"]
                low, high = net["query_cluster_bootstrap"]["median_ci95"]
                print(f"  fold {fold} n={entry['masked_candidates']:4d} "
                      f"{'primary' if entry['enters_primary'] else 'diagnostic':10s} "
                      f"seen {vs['EFFECT_SEEN']['MBRU']['median']:+.4f}  "
                      f"masked {vs['EFFECT_MASKED']['MBRU']['median']:+.4f}  "
                      f"sizectl {vs['EFFECT_SIZE_CONTROL']['MBRU']['median']:+.4f}  "
                      f"| exposure {loss['median']:+.4f}  size-controlled "
                      f"{net['median']:+.4f} [{low:+.4f},{high:+.4f}]")
            crossfit = block.get("crossfit")
            if crossfit and "paired_vs_base" in crossfit:
                vs = crossfit["paired_vs_base"]
                loss = crossfit["exposure_value"]["MBRU"]
                net = crossfit["exposure_value_size_controlled"]["MBRU"]
                shrink = crossfit["atlas_shrinkage_cost"]["MBRU"]
                low, high = net["query_cluster_bootstrap"]["median_ci95"]
                print(f"  crossfit n={crossfit['masked_candidates']:4d} "
                      f"homogeneity {crossfit['homogeneity']['minimum_pairwise_cosine']:.5f} "
                      f"{'PASS' if crossfit['homogeneity']['passes'] else 'FAIL'}")
                print(f"           vs BASE: seen {vs['EFFECT_SEEN']['MBRU']['median']:+.4f}  "
                      f"masked {vs['EFFECT_MASKED']['MBRU']['median']:+.4f}  "
                      f"sizectl {vs['EFFECT_SIZE_CONTROL']['MBRU']['median']:+.4f}")
                print(f"           exposure {loss['median']:+.4f}  = size-controlled "
                      f"{net['median']:+.4f} [{low:+.4f},{high:+.4f}] + atlas shrinkage "
                      f"{shrink['median']:+.4f}")
                for stratum, entry in crossfit.get("support_strata", {}).items():
                    if "paired_vs_base" not in entry:
                        continue
                    sm = entry["paired_vs_base"]["EFFECT_MASKED"]["MBRU"]
                    ss = entry["paired_vs_base"]["EFFECT_SEEN"]["MBRU"]
                    net = entry["exposure_value_size_controlled"]["MBRU"]
                    print(f"           {stratum:16s} n={entry['count']:4d} "
                          f"masked {sm['median']:+.4f}  seen {ss['median']:+.4f}  "
                          f"size-controlled exposure {net['median']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
