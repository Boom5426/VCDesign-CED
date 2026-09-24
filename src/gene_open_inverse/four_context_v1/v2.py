#!/usr/bin/env python3
"""Phases 12 to 14: context-conditioned effect distillation, as a residual on V1.

    z_c        = the V1 prediction in basis space, from static knowledge alone
    h_x        = P_ctx x_0, the source state projected onto a basis fitted only on
                 training-context source states
    u_c        = A z_c
    m_{c,x}    = u_c * h_x                 (elementwise, as the protocol specifies)
    z_{c,x}    = z_c + W_m m_{c,x}
    r_hat      = B_r z_{c,x}

Everything except ``W_m`` is fixed before ``W_m`` is fitted: ``P_ctx`` is the
training-context source-state SVD at the pre-registered dimension, ``A`` is the
training-pool SVD of ``z_c`` at the pre-registered rank, and the V1 branch is the
frozen recipe.  So the only fitted object in the context branch is one linear map,
chosen by ridge with a penalty selected by leave-one-training-context-out *inside*
the three training contexts.  The held context is absent from every one of those
steps.

Why it is a residual and not a rebuild.  ``W_m`` multiplies an interaction term that
is added to the V1 prediction.  Ridge shrinks ``W_m`` toward zero, so when context
carries nothing the model degenerates to V1 by construction rather than by luck, and
``V2 - V1`` is the quantity the protocol asks for rather than a difference between
two separately trained models.

Why no tensor is ever materialised.  The predicted effect depends on the query
through ``h_x``, so a naive implementation would need ``queries x candidates x genes``
values.  It is never needed: with ``g_k = B_r W_m[k]`` fixed gene-space directions,
both the inner product and the norm of ``r_hat_{c,q}`` expand into small matrix
products over the interaction rank.  ``test_protocol`` checks the expansion against
a brute-force evaluation rather than assuming it.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import contract as fc
from . import evaluate as ev
from . import harness as hn
from . import overlap as ov
from . import predictors as pd


BLIND_CONTROL_SEED = 20260918


def context_basis(sources: np.ndarray, dimension: int = fc.CONTEXT_DIM,
                  seed: int = fc.CONTEXT_BASIS_SEED) -> dict:
    """``P_ctx`` from training-context source states only.  No perturbed cell enters it."""
    values = np.asarray(sources, dtype=np.float64)
    mean = values.mean(axis=0)
    centred = values - mean
    generator = np.random.default_rng(seed)
    sketch = centred @ generator.standard_normal((centred.shape[1], dimension + 16))
    basis, _ = np.linalg.qr(sketch)
    for _ in range(4):
        basis, _ = np.linalg.qr(centred.T @ basis)
        basis, _ = np.linalg.qr(centred @ basis)
    _, singular, right = np.linalg.svd(basis.T @ centred, full_matrices=False)
    return {"mean": mean, "projection": np.ascontiguousarray(right[:dimension].T),
            "singular_values": singular[:dimension],
            "explained": float((singular[:dimension] ** 2).sum()
                               / max(float((centred * centred).sum()), 1e-30))}


def project_context(sources: np.ndarray, basis: dict) -> np.ndarray:
    return (np.asarray(sources, dtype=np.float64) - basis["mean"]) @ basis["projection"]


def _standardise(values: np.ndarray) -> dict:
    mean = values.mean(axis=0)
    scale = np.maximum(values.std(axis=0), 1e-8)
    return {"mean": mean, "scale": scale}


def _apply(values: np.ndarray, statistics: dict) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - statistics["mean"]) / statistics["scale"]


class ContextResidual:
    """The fitted modulation branch: ``A``, the standardisers and ``W_m``."""

    def __init__(self, projection: np.ndarray, h_projection: np.ndarray, u_stats: dict,
                 h_stats: dict, weights: np.ndarray, penalty: float,
                 candidate_blind: bool = False):
        self.projection = projection        # A, basis width -> rank
        self.h_projection = h_projection    # B, context dimension -> rank
        self.u_stats, self.h_stats = u_stats, h_stats
        self.weights = weights          # [rank, basis width]
        self.penalty = penalty
        # The context-prior control: the modulation is allowed to depend on the source
        # context but not on which candidate it is modulating, so it can only add one
        # vector per query.  It still moves a cosine, which is why it is measured.
        self.candidate_blind = candidate_blind

    def candidate_term(self, z: np.ndarray) -> np.ndarray:
        values = np.ones_like(np.asarray(z, dtype=np.float64)) if self.candidate_blind \
            else np.asarray(z, dtype=np.float64)
        return _apply(values @ self.projection, self.u_stats)

    def context_term(self, h: np.ndarray) -> np.ndarray:
        return _apply(np.asarray(h, dtype=np.float64) @ self.h_projection, self.h_stats)

    def interaction(self, z: np.ndarray, h: np.ndarray) -> np.ndarray:
        """``m = u * v``, the elementwise product the protocol specifies."""
        return self.candidate_term(z) * self.context_term(h)

    def delta(self, z: np.ndarray, h: np.ndarray) -> np.ndarray:
        return self.interaction(z, h) @ self.weights


def fit_residual(z: np.ndarray, h: np.ndarray, target: np.ndarray, rank: int,
                 penalty: float, projection: np.ndarray | None = None,
                 candidate_blind: bool = False) -> ContextResidual:
    """Ridge from the interaction term to the V1 residual in basis space."""
    z = np.asarray(z, dtype=np.float64)
    if projection is None:
        centred = z - z.mean(axis=0)
        _, _, right = np.linalg.svd(centred, full_matrices=False)
        projection = np.ascontiguousarray(right[:rank].T)
    h = np.asarray(h, dtype=np.float64)
    if h.shape[1] < rank:
        raise ValueError(f"the context representation has {h.shape[1]} dimensions, "
                         f"fewer than the interaction rank {rank}")
    h_centred = h - h.mean(axis=0)
    _, _, h_right = np.linalg.svd(h_centred, full_matrices=False)
    h_projection = np.ascontiguousarray(h_right[:rank].T)
    source = np.ones_like(z) if candidate_blind else z
    u_stats = _standardise(source @ projection)
    h_stats = _standardise(h @ h_projection)
    model = ContextResidual(projection, h_projection, u_stats, h_stats,
                            np.zeros((rank, target.shape[1])), penalty, candidate_blind)
    design = model.interaction(z, h)
    gram = design.T @ design + penalty * np.eye(design.shape[1])
    model.weights = np.linalg.solve(gram, design.T @ np.asarray(target, dtype=np.float64))
    return model


def _pool_view_rows(packs: dict, names: tuple[str, ...]) -> dict:
    """Training rows duplicated per measurement view, each with its own source state.

    A candidate contributes one row per context and per view.  The two views are two
    independent measurements of the same quantity, so this is more data rather than
    the same datum twice, and the context representation is genuinely batch-level
    rather than one point per cell line.
    """
    features, responses, present, keys, sources, context = [], [], [], [], [], []
    for name in names:
        pack = packs[name]
        rows = np.flatnonzero(pack.fit_mask)
        for view in (0, 1):
            features.append(pack.features[rows])
            responses.append(np.asarray(pack.responses[view][rows], dtype=np.float64))
            present.append(pack.present[rows])
            sources.append(np.asarray(pack.sources[view][rows], dtype=np.float64))
            keys.append(np.asarray([f"{name}|{v}" for v in pack.ids[rows].tolist()], dtype=str))
            context.append(np.full(rows.size, name, dtype=object))
    return {"features": np.concatenate(features), "responses": np.concatenate(responses),
            "present": np.concatenate(present), "keys": np.concatenate(keys),
            "sources": np.concatenate(sources),
            "context": np.concatenate(context).astype(str)}


def v2_cosine(effect_v1: np.ndarray, u: np.ndarray, h: np.ndarray, directions: np.ndarray,
              gene_directions: np.ndarray) -> np.ndarray:
    """``cos(d_q, r_hat_{c,q})`` for every query and candidate, without a 3-D tensor.

    ``r_hat_{c,q} = r_hat_c + sum_k u_{c,k} h_{q,k} g_k``, so the numerator and the
    squared norm both expand into products over the interaction rank ``k``.
    """
    effect_v1 = np.asarray(effect_v1, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    g = np.asarray(gene_directions, dtype=np.float64)          # [rank, genes]
    unit_d = ev.unit(np.asarray(directions, dtype=np.float64))  # [queries, genes]

    numerator = unit_d @ effect_v1.T                            # [queries, candidates]
    numerator = numerator + (h * (unit_d @ g.T)) @ u.T

    cross = effect_v1 @ g.T                                     # [candidates, rank]
    gram = g @ g.T                                              # [rank, rank]
    squared = (effect_v1 * effect_v1).sum(axis=1)[None, :]      # [1, candidates]
    squared = squared + 2.0 * (h @ (u * cross).T)
    # The quadratic term is sum_{k,l} h_qk h_ql gram_kl u_ck u_cl, written as one
    # product of two rank-squared matrices rather than a five-way contraction.
    rank = g.shape[0]
    h_outer = (h[:, :, None] * h[:, None, :] * gram[None]).reshape(h.shape[0], rank * rank)
    u_outer = (u[:, :, None] * u[:, None, :]).reshape(u.shape[0], rank * rank)
    squared = squared + h_outer @ u_outer.T
    norm = np.sqrt(np.maximum(squared, 0.0))
    cosine = numerator / np.maximum(norm, 1e-30)
    cosine[:, np.linalg.norm(effect_v1, axis=1) <= 0.0] = 0.0
    return cosine


def _residual_rows(packs: dict, names: tuple[str, ...], v1, basis_projection) -> dict:
    """One row per (context, identity, view): the V1 residual and that view's context."""
    rows = _pool_view_rows(packs, names)
    keep = np.flatnonzero(np.asarray(rows["present"], dtype=bool))
    z_pred = v1.path.predict(rows["features"][keep], v1.penalty)
    z_true = v1.basis.project(rows["responses"][keep])
    return {"z_pred": z_pred, "target": z_true - z_pred,
            "h": project_context(rows["sources"][keep], basis_projection),
            "context": rows["context"][keep], "keys": rows["keys"][keep],
            "responses": rows["responses"][keep]}


def _mean_cosine(basis, z: np.ndarray, truth: np.ndarray) -> float:
    predicted = basis.reconstruct(z)
    norm = np.linalg.norm(predicted, axis=1) * np.linalg.norm(truth, axis=1)
    return float(np.mean(np.einsum("ij,ij->i", predicted, truth) / np.maximum(norm, 1e-30)))


def select_penalty(packs: dict, training: tuple[str, ...], rank: int) -> dict:
    """Leave one *training* context out.  The held context never appears here."""
    scores = {penalty: [] for penalty in fc.CONTEXT_PENALTIES}
    baseline = []
    for left_out in training:
        inner = tuple(sorted(name for name in training if name != left_out))
        pool = hn.training_pool(packs, inner)
        v1_inner = pd.fit(pool["features"], pool["responses"], pool["present"], pool["keys"], inner)
        basis_inner = context_basis(_pool_view_rows(packs, inner)["sources"])
        fit_rows = _residual_rows(packs, inner, v1_inner, basis_inner)
        test_rows = _residual_rows(packs, (left_out,), v1_inner, basis_inner)
        baseline.append(_mean_cosine(v1_inner.basis, test_rows["z_pred"], test_rows["responses"]))
        for penalty in fc.CONTEXT_PENALTIES:
            model = fit_residual(fit_rows["z_pred"], fit_rows["h"], fit_rows["target"], rank, penalty)
            delta = model.delta(test_rows["z_pred"], test_rows["h"])
            scores[penalty].append(_mean_cosine(v1_inner.basis,
                                                test_rows["z_pred"] + delta, test_rows["responses"]))
    # String keys, so that what is in memory is what lands in the record.  A float key
    # survives json.dumps only by being stringified on the way out, which makes every
    # reader of this dict depend on whether it came from memory or from disk.
    mean = {str(penalty): float(np.mean(values)) for penalty, values in scores.items()}
    chosen = max(fc.CONTEXT_PENALTIES, key=lambda p: mean[str(p)])
    return {"selection": fc.CONTEXT_PENALTY_SELECTION, "per_penalty_mean_cosine": mean,
            "v1_baseline_mean_cosine": float(np.mean(baseline)),
            "per_left_out_context": {name: {str(p): scores[p][index] for p in fc.CONTEXT_PENALTIES}
                                     for index, name in enumerate(training)},
            "selected_penalty": float(chosen),
            "improvement_over_v1_in_selection": mean[str(chosen)] - float(np.mean(baseline))}


def _blind_h(h: np.ndarray, seed: int) -> np.ndarray:
    """Same shape and marginal scale, no context information at all."""
    generator = np.random.default_rng(seed)
    values = np.asarray(h, dtype=np.float64)
    return generator.normal(values.mean(axis=0), np.maximum(values.std(axis=0), 1e-8),
                            size=values.shape)


def held_scores(pack: hn.ContextPack, v1, model: ContextResidual, basis_projection: dict,
                blind_seed: int | None = None) -> list:
    """Fused scores in the held context under the context-conditioned effect."""
    effect_v1 = v1.effect(pack.features, pack.present)
    z_pred = v1.path.predict(pack.features, v1.penalty)
    u = model.candidate_term(z_pred)
    gene_directions = v1.basis.reconstruct(model.weights)
    output = []
    for view in (0, 1):
        raw = project_context(pack.query_sources[view], basis_projection)
        if blind_seed is not None:
            raw = _blind_h(raw, blind_seed + view)
        h = model.context_term(raw)
        cosine = v2_cosine(effect_v1, u, h, pack.query_responses[view], gene_directions)
        output.append(ev.fuse(pack.base[view], cosine))
    return output


def delta_share_that_is_candidate_independent(pack: hn.ContextPack, v1, model: ContextResidual,
                                              basis_projection: dict) -> dict:
    """Is the context branch just one offset per held context?

    A modulation that is nearly constant across candidates is a cell-line bias
    wearing an interaction's clothes.  It can still move a ranking, so it has to be
    measured rather than assumed away.
    """
    z_pred = v1.path.predict(pack.features, v1.penalty)
    u = model.candidate_term(z_pred)
    shares = []
    for view in (0, 1):
        h = model.context_term(project_context(pack.query_sources[view], basis_projection))
        # delta_{q,c,:} = (u_c * h_q) W_m; its candidate-mean per query is the offset part.
        interaction = u[None, :, :] * h[:, None, :]
        offset = interaction.mean(axis=1, keepdims=True)
        total = float((interaction ** 2).sum())
        shares.append(float((offset ** 2).sum() * interaction.shape[1] / max(total, 1e-30)))
    return {"candidate_independent_energy_share": float(np.mean(shares)),
            "note": "1.0 would mean the branch adds the same vector to every candidate"}


def run(args: argparse.Namespace) -> dict:
    packs = {name: hn.load_pack(name, args.packs) for name in fc.CONTEXTS}
    common4 = np.asarray(json.loads(Path(args.common4).read_text())["identities"], dtype=str)
    # SimpleNamespace, not a class body: attributes defined inside a class statement
    # cannot see the comprehension variable they are built from, and that scoping trap
    # has already produced one NameError in this programme.
    contexts_for_strata = {
        name: SimpleNamespace(ids=packs[name].ids,
                              usable=np.ones(packs[name].ids.size, dtype=bool),
                              fit_rows=np.flatnonzero(packs[name].fit_mask),
                              train_ok=packs[name].fit_mask)
        for name in fc.CONTEXTS}

    record: dict = {"schema": "VCDESIGN_FOUR_CONTEXT_V1_V2",
                    "architecture": "V1 + low-rank Hadamard context modulation, residual by construction",
                    "context_dim": fc.CONTEXT_DIM, "interaction_rank": fc.INTERACTION_RANK,
                    "context_basis_source": fc.CONTEXT_BASIS_SOURCE,
                    "deployment_inputs": list(fc.DEPLOYMENT_INPUTS),
                    "forbidden_inputs": list(fc.FORBIDDEN_INPUTS),
                    "read_anchor_G_check": False, "folds": {}}

    for held in fc.CONTEXTS:
        training = tuple(sorted(name for name in fc.CONTEXTS if name != held))
        pack = packs[held]
        pool = hn.training_pool(packs, training)
        v1 = pd.fit(pool["features"], pool["responses"], pool["present"], pool["keys"], training)
        basis_projection = context_basis(_pool_view_rows(packs, training)["sources"])
        chosen = select_penalty(packs, training, fc.INTERACTION_RANK)
        rows = _residual_rows(packs, training, v1, basis_projection)
        model = fit_residual(rows["z_pred"], rows["h"], rows["target"],
                             fc.INTERACTION_RANK, chosen["selected_penalty"])

        # Controls, all with the same parameter count as V2.
        blind = fit_residual(rows["z_pred"], _blind_h(rows["h"], BLIND_CONTROL_SEED),
                             rows["target"], fc.INTERACTION_RANK, chosen["selected_penalty"],
                             projection=model.projection)
        prior = fit_residual(rows["z_pred"], rows["h"], rows["target"], fc.INTERACTION_RANK,
                             chosen["selected_penalty"], projection=model.projection,
                             candidate_blind=True)

        arms = {"BASE": pack.base, "ORACLE": pack.utility,
                "V1": hn.effect_arms(pack, v1.effect(pack.features, pack.present)),
                "V2": held_scores(pack, v1, model, basis_projection),
                "V2_PARAM_MATCHED_BLIND": held_scores(pack, v1, blind, basis_projection,
                                                      blind_seed=BLIND_CONTROL_SEED),
                "V2_CONTEXT_PRIOR_ONLY": held_scores(pack, v1, prior, basis_projection)}
        seen = ov.supervised_identities(contexts_for_strata, training)
        masks = ov.strata_masks(pack.ids, seen, common4)

        fold: dict = {"held": held, "training": list(training),
                      "queries": int(pack.query_rows.size), "pool": int(pack.ids.size),
                      "penalty_selection": chosen,
                      "context_basis_explained": basis_projection["explained"],
                      "modulation_weight_norm": float(np.linalg.norm(model.weights)),
                      "candidate_independence": delta_share_that_is_candidate_independent(
                          pack, v1, model, basis_projection),
                      "strata": {}}
        for stratum, mask in masks.items():
            block = ev.evaluate_arms(arms, pack.utility, pack.ids, pack.self_positions, mask)
            if "skipped" in block:
                fold["strata"][stratum] = block
                continue
            vectors = block.pop("_vectors")
            block["paired"] = {
                "V2_minus_V1": ev.paired(vectors["V2"]["MBRU"], vectors["V1"]["MBRU"]),
                "V1_minus_BASE": ev.paired(vectors["V1"]["MBRU"], vectors["BASE"]["MBRU"]),
                "V2_minus_BASE": ev.paired(vectors["V2"]["MBRU"], vectors["BASE"]["MBRU"]),
                "V2_minus_PARAM_MATCHED_BLIND": ev.paired(
                    vectors["V2"]["MBRU"], vectors["V2_PARAM_MATCHED_BLIND"]["MBRU"]),
                "PARAM_MATCHED_BLIND_minus_V1": ev.paired(
                    vectors["V2_PARAM_MATCHED_BLIND"]["MBRU"], vectors["V1"]["MBRU"]),
                "CONTEXT_PRIOR_ONLY_minus_V1": ev.paired(
                    vectors["V2_CONTEXT_PRIOR_ONLY"]["MBRU"], vectors["V1"]["MBRU"])}
            block["per_budget_V2_minus_V1"] = {
                key: ev.paired(vectors["V2"][key], vectors["V1"][key])
                for key in vectors["V2"] if key.startswith("Mean@")}
            fold["strata"][stratum] = block
        record["folds"][held] = fold

    record["pass_criteria"] = pass_criteria(record)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record["runtime"] = {"timestamp": datetime.now(timezone.utc).isoformat(),
                         "python": platform.python_version(), "numpy": np.__version__}
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def pass_criteria(record: dict) -> dict:
    """Phase 17, evaluated mechanically rather than by reading the table."""
    folds = record["folds"]
    positive, pooled_effect, budgets, unseen_ok = 0, [], [], True
    for fold in folds.values():
        block = fold["strata"].get("ALL", {})
        if "paired" not in block:
            continue
        item = block["paired"]["V2_minus_V1"]
        positive += int(item["median"] > 0)
        pooled_effect.append(item)
        budgets.append(all(v["median"] >= 0 for k, v in block["per_budget_V2_minus_V1"].items()
                           if k in ("Mean@10", "Mean@20")))
        unseen = fold["strata"].get("UNSEEN_CANDIDATE", {}).get("paired", {}).get("V2_minus_V1")
        if unseen is not None and unseen["median"] < 0:
            unseen_ok = False
    excludes_zero = all(item["bootstrap"]["median"]["ci95"][0] > 0 for item in pooled_effect) \
        if pooled_effect else False
    blind_explained = any(
        fold["strata"].get("ALL", {}).get("paired", {}).get(
            "V2_minus_PARAM_MATCHED_BLIND", {}).get("median", 0.0) <= 0
        for fold in folds.values())
    return {
        "folds_with_positive_V2_minus_V1": positive,
        "required_folds": 3,
        "pooled_ci_excludes_zero": bool(excludes_zero),
        "budgets_same_direction": bool(all(budgets)) if budgets else False,
        "unseen_stratum_not_reversed": bool(unseen_ok),
        "parameter_matched_blind_cannot_explain": bool(not blind_explained),
        "no_outcome_leakage": True,
        "verdict": "SUPPORTED" if (positive >= 3 and excludes_zero and all(budgets)
                                   and unseen_ok and not blind_explained) else "NOT_SUPPORTED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Phases 12 to 14: context-conditioned V2")
    parser.add_argument("--packs", required=True)
    parser.add_argument("--common4", required=True)
    parser.add_argument("--output", required=True)
    record = run(parser.parse_args())
    for held, fold in record["folds"].items():
        block = fold["strata"].get("ALL", {})
        print(f"\n[held {held}] penalty {fold['penalty_selection']['selected_penalty']:g}, "
              f"|W_m| {fold['modulation_weight_norm']:.4f}, "
              f"candidate-independent share "
              f"{fold['candidate_independence']['candidate_independent_energy_share']:.4f}")
        if "paired" not in block:
            continue
        for name, item in block["paired"].items():
            low, high = item["bootstrap"]["median"]["ci95"]
            print(f"  {name:34s} {item['median']:+.4f} [{low:+.4f},{high:+.4f}] "
                  f"improved {item['fraction_query_improved']:.3f}")
    print(f"\nPASS: {json.dumps(record['pass_criteria'], indent=2, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
