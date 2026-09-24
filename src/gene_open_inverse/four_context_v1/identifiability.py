#!/usr/bin/env python3
"""V2_CONTEXT_IDENTIFIABILITY_POSITIVE_CONTROL.

The question this answers is not whether context conditioning helps biologically.
It is narrower and prior to that:

    if a candidate's effect really did carry a residual of exactly the pre-registered
    V2 form, and that residual really were a function of the deployment-available
    context representation, could this training, regularization and selection
    pipeline recover it in a held context from three training contexts?

Without that answer a real ``V2 - V1 <= 0`` is ambiguous between "biology does not
support context conditioning" and "three training contexts cannot identify this
model".  The previous mission already had to record the two-context version of that
ambiguity, so the ambiguity is not hypothetical.

What is held fixed
------------------
Everything the real experiment will use: ``CONTEXT_DIM``, ``INTERACTION_RANK``, the
penalty grid, the leave-one-training-context-out selection rule, the basis rank and
the ridge recipe.  Only the data are synthetic.  No fourth architecture is added and
no hyperparameter is chosen on a result from this test.

How the ground truth is made
----------------------------
The same functional family as the model, with independently seeded parameters the
model never sees:

    z_c            = k_c W_k*                      static knowledge to basis space
    dz*_{c,x}      = W_m*[(A* z_c) . (B* h_x)]     the planted residual
    r*_{c,x}       = B_r*(z_c + gamma dz*_{c,x})

``dz*`` is centred across the contexts a candidate appears in, so it is a residual by
construction and contributes no context-blind component that V1 could absorb.  Being
exactly orthogonal to the shared part in the energy sum, ``gamma`` can then be solved
for a requested residual energy share rather than tuned toward one.

Where the context representation comes from
-------------------------------------------
Real control and source states, never perturbation outcomes.  K562, RPE1 and HepG2
supply the three training contexts.  Until Jurkat lands the held context uses a
deterministic synthetic source state placed at the scale of the real between-context
spread; afterwards the same test is repeated with the real Jurkat control cells,
before any Jurkat perturbation result is read.

The share curve
---------------
``0, 0.10, 0.25, 0.50`` is an identifiability curve, not a search.  No level is
selected as best; the shape across levels is the finding.
"""
from __future__ import annotations

import argparse
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..candidate_effect_distillation_v1 import contract as ced
from . import contract as fc
from . import contexts as cx
from . import evaluate as ev
from . import harness as hn
from . import predictors as pd
from . import v2


# --- frozen before the first number ----------------------------------------
SHARE_CURVE = (0.0, 0.10, 0.25, 0.50)

# Two ways of planting, because they answer different questions and only one of them is
# the identifiability question.
#
# The frozen V2 does not learn A and B.  It fixes A as the leading directions of the
# predicted z and B as the leading directions of the training h, and fits W_m alone.
# The Hadamard product (Az) . (Bh) is not invariant to that choice: a different A pairs
# different coordinates together.  So planting with independently drawn A* and B*, as
# the first version of this control did, plants a residual that the model may be unable
# to *represent* at all, and a failure then says nothing about whether W_m could have
# been *estimated*.
#
#   MATCHED_BASIS  A and B are the ones the model will use; only W_m* is independent.
#                  The truth is inside the model's family, so a failure is squarely an
#                  estimation failure.  This is the identifiability question.
#   FREE_BASIS     A*, B* and W_m* are all independent.  A failure here can also be a
#                  representability failure, and comparing the two modes says which.
PLANTING_MODES = ("MATCHED_BASIS", "FREE_BASIS")
REPEAT_SEEDS = (20260918, 20260919, 20260920)     # the same seeds at every share level
GROUND_TRUTH_SEED = 20260931                      # independent of every model seed
HELD_EMBEDDING_SEED = 20260932
HELD_CONTEXT_NAME = "SYNTHETIC_HELD"
MEASUREMENT_NOISE_SHARE = 0.25                    # noise energy relative to the effect
BASE_NOISE_GRID = (0.5, 1.0, 2.0, 4.0, 8.0)       # calibrated once, at share zero only
BASE_MBRU_TARGET = 0.35                           # the level the real contexts sit at
TRUE_LATENT = 256                                 # width of the planted basis space


@dataclass(frozen=True)
class GroundTruth:
    """The planted effect, and the pieces a recovery check needs to compare against."""

    shared_z: np.ndarray                 # [candidates, TRUE_LATENT]
    residual_z: dict                     # context -> [2, candidates, TRUE_LATENT]
    response: dict                       # context -> [2, candidates, genes] noiseless
    basis: np.ndarray                    # [TRUE_LATENT, genes], the planted latent map
    gamma: float
    achieved_share: float
    mode: str

    def residual_response(self, context: str) -> np.ndarray:
        """The planted residual in gene space, at the scale it was actually planted."""
        return self.gamma * self.residual_z[context] @ self.basis


def _unit(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-30)


def synthetic_held_sources(training: dict, candidates: int, genes: int,
                           seed: int = HELD_EMBEDDING_SEED) -> np.ndarray:
    """A deterministic held-context source state that is genuinely a new context.

    The obvious construction, a random direction in gene space at the typical
    between-context distance, does not work.  ``P_ctx`` has ``CONTEXT_DIM`` directions
    fitted on three contexts, and a random direction in 6835 dimensions is very nearly
    orthogonal to all of them, so it projects to almost nothing.  The held context
    would then land at the centre of the training distribution and the test would be
    asking whether the pipeline can recover a residual for an average context, which
    is not the question.

    So the offset is built inside the span the context basis can actually see: a fixed
    contrast between the training context centres, scaled to the real between-context
    distance.  That places the held context at a realistic distance from every training
    context while remaining a representable point, which is what a fourth real cell
    line would be.  The within-context spread is carried over so that h varies across
    candidates in the held context as it does in the real ones.
    """
    names = sorted(training)
    centres = np.stack([np.asarray(training[name], dtype=np.float64).reshape(-1, genes).mean(axis=0)
                        for name in names])
    centre = centres.mean(axis=0)
    deviations = centres - centre
    generator = np.random.default_rng(seed)
    # A fixed contrast across the training centres, orthogonalised against none of them
    # on purpose: the point is to be a different mixture, not a different subspace.
    weights = generator.standard_normal(len(names))
    weights -= weights.mean()
    weights /= np.linalg.norm(weights)
    direction = weights @ deviations
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise RuntimeError("the training context centres are degenerate")
    spread = float(np.median([np.linalg.norm(row) for row in deviations]))
    offset = centre + spread * direction / norm

    # Within-context variation, reused from a real context so the held context's per
    # candidate spread is the real one rather than an invented scale.
    reference = np.asarray(training[names[0]], dtype=np.float64)
    within = reference - reference.reshape(-1, genes).mean(axis=0)
    return (offset[None, None, :] + within[:, :candidates]).astype(np.float32)


def context_geometry(h_by_context: dict, training: tuple) -> dict:
    """Where the held context sits relative to the training ones, in h space.

    Reported so that a recovery number can be read against how new the held context
    actually is.  A held context that collapses onto the training centre would make a
    successful recovery meaningless.
    """
    centres = {name: np.asarray(h, dtype=np.float64).reshape(-1, h.shape[-1]).mean(axis=0)
               for name, h in h_by_context.items()}
    training_centres = np.stack([centres[name] for name in training])
    pooled = np.concatenate([np.asarray(h_by_context[name], dtype=np.float64).reshape(-1, training_centres.shape[1])
                             for name in training])
    within = float(np.median(np.linalg.norm(pooled - pooled.mean(axis=0), axis=1)))
    between = [float(np.linalg.norm(training_centres[i] - training_centres[j]))
               for i in range(len(training)) for j in range(i + 1, len(training))]
    held = [name for name in centres if name not in training]
    record = {"median_within_context_h_norm": within,
              "between_training_context_centre_distance": {
                  "median": float(np.median(between)), "min": float(np.min(between)),
                  "max": float(np.max(between))}}
    for name in held:
        distances = [float(np.linalg.norm(centres[name] - training_centres[i]))
                     for i in range(len(training))]
        record[f"{name}_distance_to_training_centres"] = {
            "min": float(np.min(distances)), "median": float(np.median(distances)),
            "max": float(np.max(distances)),
            "ratio_to_between_training_median": float(np.median(distances) / max(np.median(between), 1e-30)),
            "is_a_distinguishable_context": bool(
                np.min(distances) > 0.25 * np.median(between))}
    return record


def _leading_directions(values: np.ndarray, rank: int) -> np.ndarray:
    """The projection the frozen model builds: leading right-singular vectors, centred."""
    flat = np.asarray(values, dtype=np.float64).reshape(-1, np.asarray(values).shape[-1])
    _, _, right = np.linalg.svd(flat - flat.mean(axis=0), full_matrices=False)
    return np.ascontiguousarray(right[:rank].T)


def plant(features: np.ndarray, h_by_context: dict, present_by_context: dict,
          genes: int, share: float, mode: str = "MATCHED_BASIS",
          training: tuple | None = None, seed: int = GROUND_TRUTH_SEED) -> GroundTruth:
    """Build the planted effect at a requested residual energy share.

    ``gamma`` is solved, not searched: with the residual centred across contexts the
    shared and residual parts are orthogonal in the energy sum, so
    ``share = g^2 S / (T + g^2 S)`` inverts exactly.
    """
    if mode not in PLANTING_MODES:
        raise ValueError(f"unknown planting mode {mode}")
    generator = np.random.default_rng(seed)
    width = features.shape[1]
    w_k = generator.standard_normal((width, TRUE_LATENT)) / np.sqrt(width)
    context_dim = next(iter(h_by_context.values())).shape[-1]
    free_a = generator.standard_normal((TRUE_LATENT, fc.INTERACTION_RANK)) / np.sqrt(TRUE_LATENT)
    free_b = generator.standard_normal((context_dim, fc.INTERACTION_RANK)) / np.sqrt(context_dim)
    w_m = generator.standard_normal((fc.INTERACTION_RANK, TRUE_LATENT)) / np.sqrt(fc.INTERACTION_RANK)
    basis = generator.standard_normal((TRUE_LATENT, genes)) / np.sqrt(TRUE_LATENT)

    shared_z = features @ w_k                                   # [candidates, latent]
    if mode == "MATCHED_BASIS":
        # The model fixes A from the predicted z and B from the training h.  The true z
        # stands in for the predicted one here, which is sound because with no residual
        # V1 recovers it at cosine 0.9965, and using the predicted z would make the
        # ground truth depend on a model fitted to it.
        names = tuple(training) if training else tuple(h_by_context)
        a_star = _leading_directions(shared_z, fc.INTERACTION_RANK)
        b_star = _leading_directions(np.concatenate(
            [np.asarray(h_by_context[name], dtype=np.float64).reshape(-1, context_dim)
             for name in names]), fc.INTERACTION_RANK)
    else:
        a_star, b_star = free_a, free_b
    u = shared_z @ a_star                                       # [candidates, rank]

    raw: dict = {}
    for name, h in h_by_context.items():
        v = np.asarray(h, dtype=np.float64) @ b_star            # [2, candidates, rank]
        raw[name] = (u[None] * v) @ w_m                         # [2, candidates, latent]

    # Centre across the contexts in which each candidate is present, so what is planted
    # is a residual and not a shared component wearing one's clothes.
    stack = np.stack([raw[name] for name in raw])               # [contexts, 2, cand, latent]
    presence = np.stack([np.asarray(present_by_context[name], dtype=np.float64) for name in raw])
    weight = presence[:, None, :, None]
    mean = (stack * weight).sum(axis=0) / np.maximum(weight.sum(axis=0), 1e-12)
    residual_z = {name: (raw[name] - mean) * np.asarray(present_by_context[name],
                                                        dtype=np.float64)[None, :, None]
                  for name in raw}

    shared_energy = float(((shared_z @ basis) ** 2).sum()) * 2 * len(raw)
    residual_energy = float(sum(((residual_z[name] @ basis) ** 2).sum() for name in raw))
    if share <= 0.0 or residual_energy <= 0.0:
        gamma = 0.0
    else:
        gamma = float(np.sqrt(share * shared_energy / ((1.0 - share) * residual_energy)))
    achieved = (gamma ** 2 * residual_energy) / max(shared_energy + gamma ** 2 * residual_energy, 1e-30)

    response = {}
    for name in raw:
        z = shared_z[None] + gamma * residual_z[name]
        response[name] = (z @ basis).astype(np.float64)
    return GroundTruth(shared_z=shared_z, residual_z=residual_z, response=response,
                       basis=basis, gamma=gamma, achieved_share=float(achieved), mode=mode)


def _pack(name: str, ids: np.ndarray, response: np.ndarray, sources: np.ndarray,
          features: np.ndarray, present: np.ndarray, noise_scale: float,
          base_noise: float, seed: int) -> hn.ContextPack:
    """One synthetic context in exactly the shape the real pipeline consumes."""
    generator = np.random.default_rng(seed)
    # ``response`` already carries both measurement views, because the context
    # representation h_x is built per view from that view's own source state, so the
    # planted effect genuinely differs between them.  Independent noise is added on
    # top of each; stacking it again would have produced a four-dimensional array.
    if response.ndim != 3 or response.shape[0] != 2:
        raise ValueError(f"expected a [2, candidates, genes] response, got {response.shape}")
    scale = noise_scale * float(np.sqrt((response ** 2).mean()))
    measured = (response + scale * generator.standard_normal(response.shape)).astype(np.float32)
    utility = [np.ascontiguousarray(ev.cross_view_utility(measured, view)) for view in (0, 1)]
    base = []
    for view in (0, 1):
        row_scale = np.maximum(utility[view].std(axis=1, keepdims=True), 1e-12)
        base.append(utility[view] + base_noise * row_scale
                    * generator.standard_normal(utility[view].shape))
    rows = np.arange(ids.size, dtype=np.int64)
    return hn.ContextPack(
        name=name, ids=ids, pool_rows=rows, query_rows=rows, self_positions=rows,
        responses=measured, sources=np.asarray(sources, dtype=np.float32),
        query_responses=measured, query_sources=np.asarray(sources, dtype=np.float32),
        features=features, present=present, base=base, utility=utility,
        fit_mask=np.ones(ids.size, dtype=bool),
        verification={"synthetic": True, "measurement_noise_share": noise_scale})


def calibrate_base(build, held_key: str, grid=BASE_NOISE_GRID) -> dict:
    """Pick the base noise once, at share zero, against a target the arms do not see.

    The base ranker is common to V1, V2 and the blind control, so its strength cannot
    bias a comparison between them.  It is calibrated anyway, because a base that is
    pure noise dilutes every arm equally but by so much that a real difference could
    be buried, and a base that is near-perfect would leave nothing for an effect to
    add.  The target is where the real contexts actually sit.
    """
    record = {}
    for noise in grid:
        packs = build(0.0, REPEAT_SEEDS[0], noise)
        held = packs[held_key]
        block = ev.evaluate_arms({"BASE": held.base}, held.utility, held.ids,
                                 held.self_positions, np.ones(held.ids.size, dtype=bool))
        record[str(noise)] = float(block["arms"]["BASE"]["MBRU"]["median"])
    chosen = min(grid, key=lambda n: abs(record[str(n)] - BASE_MBRU_TARGET))
    return {"grid": list(grid), "base_MBRU_by_noise": record, "target": BASE_MBRU_TARGET,
            "selected_noise": float(chosen), "achieved_base_MBRU": record[str(chosen)],
            "calibrated_at_share": 0.0,
            "note": "the base is shared by every arm, so this cannot move V2 against V1"}


def recovery(model, v1, pack: hn.ContextPack, basis_projection: dict,
             truth: GroundTruth) -> dict:
    """How much of the planted residual, and of the whole effect, came back."""
    z_pred = v1.path.predict(pack.features, v1.penalty)
    planted_residual = truth.residual_response(pack.name)               # [2, n, genes]
    planted_total = truth.response[pack.name]                           # already [2, n, genes]
    for label, array in (("residual", planted_residual), ("total", planted_total)):
        if array.shape != (2,) + pack.responses.shape[1:]:
            raise ValueError(f"planted {label} has shape {array.shape}, expected "
                             f"{(2,) + pack.responses.shape[1:]}")

    per_view = {"residual_cosine": [], "total_cosine": [], "residual_norm_ratio": []}
    for view in (0, 1):
        h = project_view(pack, view, basis_projection)
        delta = model.delta(z_pred, h)                                   # [n, latent]
        recovered_residual = v1.basis.reconstruct(delta)                 # [n, genes]
        recovered_total = v1.basis.reconstruct(z_pred + delta)
        per_view["residual_cosine"].append(
            np.einsum("ij,ij->i", _unit(recovered_residual), _unit(planted_residual[view])))
        per_view["total_cosine"].append(
            np.einsum("ij,ij->i", _unit(recovered_total), _unit(planted_total[view])))
        per_view["residual_norm_ratio"].append(
            np.linalg.norm(recovered_residual, axis=1)
            / np.maximum(np.linalg.norm(planted_residual[view], axis=1), 1e-30))

    flat_recovered = np.concatenate([
        v1.basis.reconstruct(model.delta(z_pred, project_view(pack, view, basis_projection))).ravel()
        for view in (0, 1)])
    flat_planted = np.concatenate([planted_residual[view].ravel() for view in (0, 1)])
    if flat_recovered.std() > 0 and flat_planted.std() > 0:
        correlation = float(np.corrcoef(flat_recovered, flat_planted)[0, 1])
    else:
        correlation = 0.0

    output = {"coefficient_space_correlation": correlation}
    for key, values in per_view.items():
        joined = np.concatenate(values)
        output[key] = {"median": float(np.median(joined)), "mean": float(joined.mean()),
                       "p05": float(np.quantile(joined, 0.05)),
                       "p95": float(np.quantile(joined, 0.95))}
    return output


def project_view(pack: hn.ContextPack, view: int, basis_projection: dict) -> np.ndarray:
    return v2.project_context(pack.sources[view], basis_projection)


def real_measurement_reliability(args) -> dict:
    """How clean the real contexts are, so the synthetic noise level can be read against it.

    The control plants its effects at ``MEASUREMENT_NOISE_SHARE``, which implies a
    cross-view response cosine of about ``1 / (1 + share^2)``.  If that is higher than
    the real contexts achieve, the control is running on data cleaner than reality, and
    a failure to identify the residual there is conservative: the real data cannot be
    easier.  A success would instead have needed re-checking at the real noise level.
    """
    record = {"synthetic_measurement_noise_share": MEASUREMENT_NOISE_SHARE,
              "implied_synthetic_cross_view_cosine":
                  float(1.0 / (1.0 + MEASUREMENT_NOISE_SHARE ** 2)),
              "real": {}}
    for item in args.context:
        name, path = item.split("=", 1)
        responses = np.load(Path(path) / "side_responses.npy")
        usable = np.load(Path(path) / "usable.npy")
        left = np.asarray(responses[0][usable], dtype=np.float64)
        right = np.asarray(responses[1][usable], dtype=np.float64)
        cosine = np.einsum("ij,ij->i", _unit(left), _unit(right))
        record["real"][name] = {"identities": int(cosine.size),
                                "median": float(np.median(cosine)),
                                "mean": float(cosine.mean()),
                                "p05": float(np.quantile(cosine, 0.05)),
                                "p95": float(np.quantile(cosine, 0.95))}
    medians = [entry["median"] for entry in record["real"].values()]
    record["synthetic_is_cleaner_than_every_real_context"] = bool(
        all(record["implied_synthetic_cross_view_cosine"] > m for m in medians))
    record["reading"] = ("a negative verdict obtained on data this much cleaner than the "
                         "real contexts is conservative" if record[
                             "synthetic_is_cleaner_than_every_real_context"]
                         else "the synthetic noise level is not more favourable than reality")
    return record


def load_real_contexts(args) -> dict:
    """K562, RPE1 and HepG2 on one shared gene axis, with their real source states."""
    from ..model.decision_alignment_v1.assets import AuditConfig
    from ..model.global_objective_knowledge_attribution_v1.assets import GlobalAttributionAssets
    from ..candidate_effect_distillation_v1 import predictor as pr
    from ..external_context_v1.transfer import external_capabilities
    from types import SimpleNamespace

    config = AuditConfig.load(Path(args.config))
    assets = GlobalAttributionAssets(modalities=("STRING", "MAPKG", "TEXT"), **config.asset_arguments())
    anchor_genes = np.asarray([str(v) for v in np.load(args.anchor_genes, allow_pickle=True)])

    directories = {}
    for item in args.context:
        name, path = item.split("=", 1)
        directories[name] = Path(path)
    held_name, held_directory = None, None
    if getattr(args, "held_context", None):
        held_name, held_path = args.held_context.split("=", 1)
        held_directory = Path(held_path)
        directories[held_name] = held_directory

    shared = anchor_genes
    for path in directories.values():
        shared = np.intersect1d(
            shared, np.asarray([str(v) for v in np.load(path / "gene_axis.npy", allow_pickle=True)]))
    shared = np.sort(shared)

    contexts = {fc.ANCHOR_CONTEXT: cx.load_anchor(assets, args.anchor_sources, args.anchor_genes,
                                                  args.eligibility, shared)}
    for name, path in directories.items():
        contexts[name] = cx.load_built(name, path, shared)

    candidates = None
    for context in contexts.values():
        usable = context.ids[context.usable]
        candidates = usable if candidates is None else np.intersect1d(candidates, usable)
    candidates = np.sort(candidates)

    sources = {}
    for name, context in contexts.items():
        position = {value: row for row, value in enumerate(context.ids.tolist())}
        rows = np.asarray([position[value] for value in candidates.tolist()], dtype=np.int64)
        sources[name] = np.ascontiguousarray(context.sources[:, rows])

    capabilities, present_by_name, verification = external_capabilities(assets, candidates)
    features = pr.static_features(
        SimpleNamespace(capabilities=capabilities, capability_present=present_by_name),
        ced.PRIMARY_MODALITIES)
    # A real held context contributes its gene axis, its candidate vocabulary and its
    # control-derived source states, and nothing else.  Its measured responses are never
    # read: the held context's effects in this test are planted, so the real ones are not
    # touched even in principle.  The only perturbation-derived quantity anywhere in
    # ``sources`` is the per-batch cell incidence used to weight control means, which is
    # a count of which batches an identity was measured in and carries no response.
    training = tuple(sorted(name for name in contexts if name != held_name))
    return {"genes": shared, "candidates": candidates, "sources": sources,
            "features": features.values, "present": features.present_any,
            "candidate_feature_check": verification,
            "training": training, "held_name": held_name,
            "held_is_real": bool(held_name is not None)}


def build_world(real: dict, share: float, seed: int, base_noise: float,
                mode: str = "MATCHED_BASIS") -> tuple[dict, GroundTruth, dict, dict]:
    """Every context, planted and packed, with the context basis the model will refit."""
    genes = int(real["genes"].size)
    candidates = real["candidates"]
    training = real["training"]

    training_sources = {name: real["sources"][name] for name in training}
    if real["held_is_real"]:
        sources = {name: real["sources"][name] for name in training}
        sources[real["held_name"]] = real["sources"][real["held_name"]]
    else:
        sources = dict(training_sources)
        sources[HELD_CONTEXT_NAME] = synthetic_held_sources(training_sources, candidates.size, genes)

    # P_ctx is fitted on the training contexts only, in exactly the order and shape the
    # model will see, so the representation the residual is planted against is the one
    # the model recovers rather than a private copy of it.
    pooled = np.concatenate([np.asarray(sources[name][view], dtype=np.float64)
                             for name in training for view in (0, 1)])
    basis_projection = v2.context_basis(pooled)
    h_by_context = {name: np.stack([v2.project_context(sources[name][view], basis_projection)
                                    for view in (0, 1)]) for name in sources}
    present_by_context = {name: np.ones(candidates.size, dtype=bool) for name in sources}

    geometry = context_geometry(h_by_context, training)
    truth = plant(real["features"], h_by_context, present_by_context, genes, share,
                  mode=mode, training=training)

    packs = {}
    for index, name in enumerate(sources):
        packs[name] = _pack(name, candidates, truth.response[name], sources[name],
                            real["features"], real["present"], MEASUREMENT_NOISE_SHARE,
                            base_noise, seed + 1000 * index)
    return packs, truth, basis_projection, geometry


def held_name(real: dict) -> str:
    return real["held_name"] if real["held_is_real"] else HELD_CONTEXT_NAME


def run_fold(packs: dict, truth: GroundTruth, real: dict, basis_projection: dict) -> dict:
    """Three training contexts to one held context, under the frozen V2 pipeline.

    The frozen pipeline decides everything that enters the verdict.  A held-context
    penalty sweep is computed alongside it purely as a diagnostic, because without it a
    failure cannot be attributed between the model and the rule that selects its
    regularization.
    """
    training = real["training"]
    pack = packs[held_name(real)]

    pool = hn.training_pool(packs, training)
    v1 = pd.fit(pool["features"], pool["responses"], pool["present"], pool["keys"], training)
    chosen = v2.select_penalty(packs, training, fc.INTERACTION_RANK)
    rows = v2._residual_rows(packs, training, v1, basis_projection)
    model = v2.fit_residual(rows["z_pred"], rows["h"], rows["target"],
                            fc.INTERACTION_RANK, chosen["selected_penalty"])
    blind = v2.fit_residual(rows["z_pred"], v2._blind_h(rows["h"], v2.BLIND_CONTROL_SEED),
                            rows["target"], fc.INTERACTION_RANK, chosen["selected_penalty"],
                            projection=model.projection)

    arms = {"BASE": pack.base, "ORACLE": pack.utility,
            "V1": hn.effect_arms(pack, v1.effect(pack.features, pack.present)),
            "V2": v2.held_scores(pack, v1, model, basis_projection),
            "PARAM_MATCHED_BLIND": v2.held_scores(pack, v1, blind, basis_projection,
                                                  blind_seed=v2.BLIND_CONTROL_SEED)}
    block = ev.evaluate_arms(arms, pack.utility, pack.ids, pack.self_positions,
                             np.ones(pack.ids.size, dtype=bool))
    vectors = block.pop("_vectors")
    paired = {
        "V2_minus_V1": ev.paired(vectors["V2"]["MBRU"], vectors["V1"]["MBRU"]),
        "V2_minus_PARAM_MATCHED_BLIND": ev.paired(vectors["V2"]["MBRU"],
                                                  vectors["PARAM_MATCHED_BLIND"]["MBRU"]),
        "PARAM_MATCHED_BLIND_minus_V1": ev.paired(vectors["PARAM_MATCHED_BLIND"]["MBRU"],
                                                  vectors["V1"]["MBRU"]),
        "V1_minus_BASE": ev.paired(vectors["V1"]["MBRU"], vectors["BASE"]["MBRU"]),
    }
    # Diagnostic only, and never fed back into anything.  If the pipeline fails, this
    # separates the two candidate reasons: a model that cannot represent the residual
    # would be flat across every penalty, whereas a selection rule that cannot certify
    # one would show a penalty that *would* have helped on the held context but that
    # the inner leave-one-training-context-out loop declined to choose.  With three
    # training contexts that inner loop only ever sees two, which is exactly the
    # structural limit this control exists to measure.
    sweep = {}
    for penalty in fc.CONTEXT_PENALTIES:
        probe = v2.fit_residual(rows["z_pred"], rows["h"], rows["target"],
                                fc.INTERACTION_RANK, float(penalty),
                                projection=model.projection)
        probe_arms = {"BASE": pack.base, "V1": arms["V1"],
                      "PROBE": v2.held_scores(pack, v1, probe, basis_projection)}
        probe_block = ev.evaluate_arms(probe_arms, pack.utility, pack.ids,
                                       pack.self_positions, np.ones(pack.ids.size, dtype=bool))
        probe_vectors = probe_block.pop("_vectors")
        probe_recovery = recovery(probe, v1, pack, basis_projection, truth)
        sweep[str(penalty)] = {
            "V2_minus_V1": ev.paired(probe_vectors["PROBE"]["MBRU"],
                                     probe_vectors["V1"]["MBRU"])["median"],
            "residual_cosine": probe_recovery["residual_cosine"]["median"],
            "modulation_weight_norm": float(np.linalg.norm(probe.weights)),
            "inner_selection_score": chosen["per_penalty_mean_cosine"][str(penalty)],
        }
    best_penalty = max(fc.CONTEXT_PENALTIES, key=lambda pn: sweep[str(pn)]["V2_minus_V1"])

    return {"penalty_selection": chosen,
            "held_context_lambda_sweep_diagnostic_only": sweep,
            "best_penalty_in_hindsight": float(best_penalty),
            "best_V2_minus_V1_in_hindsight": sweep[str(best_penalty)]["V2_minus_V1"],
            "selection_rule_found_the_hindsight_best": bool(
                float(best_penalty) == float(chosen["selected_penalty"])),
            "modulation_weight_norm": float(np.linalg.norm(model.weights)),
            "v1_held_out_cosine": v1.held_out_cosine, "v1_rows": v1.rows,
            "arms": block["arms"], "queries": block["queries"], "pool": block["candidates"],
            "paired": paired,
            "recovery": recovery(model, v1, pack, basis_projection, truth)}


def pass_criteria(curve: dict) -> dict:
    """Phase 8 of the positive control, evaluated mechanically.

    A pipeline that cannot recover a residual it was handed on a plate cannot be read
    as evidence about biology when it later finds none.
    """
    def at(share: str, path: tuple):
        block = curve.get(share, {}).get("pooled", {})
        for key in path:
            block = block.get(key, {}) if isinstance(block, dict) else {}
        return block if not isinstance(block, dict) else block.get("median", 0.0)

    null_off = (abs(at("0.0", ("paired", "V2_minus_V1"))) < 1e-6
                and curve.get("0.0", {}).get("modulation_shut_off", False))
    moderate = at("0.25", ("recovery", "residual_cosine"))
    strong = at("0.5", ("recovery", "residual_cosine"))
    weak = at("0.1", ("recovery", "residual_cosine"))
    monotone = weak <= moderate + 1e-9 <= strong + 1e-9

    beats_v1 = all(at(s, ("paired", "V2_minus_V1")) > 0 for s in ("0.25", "0.5"))
    beats_blind = all(at(s, ("paired", "V2_minus_PARAM_MATCHED_BLIND")) > 0
                      for s in ("0.25", "0.5"))
    recovers = moderate > 0.1 and strong > 0.1

    verdict = ("V2_IDENTIFIABLE_WITH_THREE_TRAINING_CONTEXTS"
               if (null_off and recovers and monotone and beats_v1 and beats_blind)
               else "V2_NOT_IDENTIFIABLE_WITH_THREE_TRAINING_CONTEXTS")
    return {
        "null_modulation_shut_off_and_V2_equals_V1": bool(null_off),
        "residual_recovered_at_moderate_and_strong": bool(recovers),
        "recovery_improves_with_injected_strength": bool(monotone),
        "V2_beats_V1_on_held_design_at_moderate_and_strong": bool(beats_v1),
        "V2_beats_parameter_matched_blind": bool(beats_blind),
        "residual_cosine": {"weak": weak, "moderate": moderate, "strong": strong},
        "verdict": verdict,
        "reading": ("a NO here means a real V2 - V1 <= 0 cannot be read as biology "
                    "declining context conditioning; it would mean three training "
                    "contexts do not identify this model"),
    }


def _pool_seeds(per_seed: list) -> dict:
    """Mean across the pre-registered repeats, with the spread kept visible."""
    output: dict = {}
    for group in ("paired", "recovery"):
        output[group] = {}
        keys = per_seed[0][group].keys()
        for key in keys:
            values = []
            for entry in per_seed:
                item = entry[group][key]
                values.append(item["median"] if isinstance(item, dict) and "median" in item
                              else float(item))
            array = np.asarray(values, dtype=np.float64)
            output[group][key] = {"median": float(array.mean()), "min": float(array.min()),
                                  "max": float(array.max()), "seeds": int(array.size)}
    return output


def run(args: argparse.Namespace) -> dict:
    real = load_real_contexts(args)
    print(f"shared gene axis {real['genes'].size}, candidates in every real context "
          f"{real['candidates'].size}, training contexts {real['training']}")
    print(f"candidate feature check {json.dumps(real['candidate_feature_check'])}")

    def build(share, seed, base_noise):
        packs, _, _, _ = build_world(real, share, seed, base_noise, "MATCHED_BASIS")
        return packs

    calibration = calibrate_base(build, held_name(real))
    base_noise = calibration["selected_noise"]
    print(f"base calibration: noise {base_noise:g} gives BASE MBRU "
          f"{calibration['achieved_base_MBRU']:+.4f} against a target of {BASE_MBRU_TARGET}")

    record: dict = {
        "schema": "VCDESIGN_V2_CONTEXT_IDENTIFIABILITY_POSITIVE_CONTROL",
        "purpose": ("whether the frozen V2 pipeline can recover a residual of its own "
                    "functional form from three training contexts, not whether context "
                    "conditioning helps biologically"),
        "held_context": held_name(real),
        "held_context_is_real": real["held_is_real"],
        "held_context_note": (
            "real control-derived source states, with the held context's measured "
            "responses never read" if real["held_is_real"] else
            "a deterministic synthetic source state at the real between-context scale; "
            "the same test is repeated with real Jurkat control cells once they land, "
            "before any Jurkat perturbation result is read"),
        "training_contexts": list(real["training"]),
        "shared_gene_axis": int(real["genes"].size),
        "candidates": int(real["candidates"].size),
        "share_curve": list(SHARE_CURVE), "seeds": list(REPEAT_SEEDS),
        "frozen": {"context_dim": fc.CONTEXT_DIM, "interaction_rank": fc.INTERACTION_RANK,
                   "basis_rank": ced.BASIS_RANK, "penalties": list(fc.CONTEXT_PENALTIES),
                   "penalty_selection": fc.CONTEXT_PENALTY_SELECTION},
        "base_calibration": calibration,
        "measurement_reliability": real_measurement_reliability(args),
        "context_geometry": build_world(real, 0.0, REPEAT_SEEDS[0], base_noise)[3],
        "planting_modes": list(PLANTING_MODES),
        "planting_mode_note": (
            "MATCHED_BASIS puts the truth inside the model's family, so a failure is an "
            "estimation failure; FREE_BASIS also lets the truth fall outside it, so the "
            "gap between the two is the representability cost of fixing A and B"),
        "measurement_noise_share": MEASUREMENT_NOISE_SHARE,
        "read_anchor_G_check": False,
        "uses_held_context_perturbation_outcomes": False,
        "curve": {},
    }

    for mode in PLANTING_MODES:
      record.setdefault("curves", {})[mode] = {}
      print(f"\n================ planting mode {mode} ================")
      for share in SHARE_CURVE:
        per_seed = []
        for seed in REPEAT_SEEDS:
            packs, truth, basis_projection, geometry = build_world(real, share, seed,
                                                                   base_noise, mode)
            fold = run_fold(packs, truth, real, basis_projection)
            fold["achieved_share"] = truth.achieved_share
            fold["gamma"] = truth.gamma
            per_seed.append(fold)
        pooled = _pool_seeds(per_seed)
        shut_off = all(entry["modulation_weight_norm"] < 1e-8
                       or abs(entry["paired"]["V2_minus_V1"]["median"]) < 1e-6
                       for entry in per_seed)
        record["curves"][mode][str(share)] = {
            "requested_share": share,
            "achieved_share": float(np.mean([e["achieved_share"] for e in per_seed])),
            "selected_penalty": [e["penalty_selection"]["selected_penalty"] for e in per_seed],
            "modulation_weight_norm": [e["modulation_weight_norm"] for e in per_seed],
            "modulation_shut_off": bool(shut_off),
            "v1_held_out_cosine": [e["v1_held_out_cosine"] for e in per_seed],
            "pooled": pooled,
            "per_seed": per_seed,
        }
        block = record["curves"][mode][str(share)]
        print(f"\nshare {share:.2f} (achieved {block['achieved_share']:.4f}): "
              f"penalty {block['selected_penalty']}")
        print(f"  residual recovery cosine {pooled['recovery']['residual_cosine']['median']:+.4f} "
              f"[{pooled['recovery']['residual_cosine']['min']:+.4f},"
              f"{pooled['recovery']['residual_cosine']['max']:+.4f}]")
        print(f"  total effect cosine      {pooled['recovery']['total_cosine']['median']:+.4f}")
        print(f"  coefficient correlation  {pooled['recovery']['coefficient_space_correlation']['median']:+.4f}")
        print(f"  V2 - V1                  {pooled['paired']['V2_minus_V1']['median']:+.5f}")
        print(f"  V2 - blind               {pooled['paired']['V2_minus_PARAM_MATCHED_BLIND']['median']:+.5f}")
        print(f"  blind - V1               {pooled['paired']['PARAM_MATCHED_BLIND_minus_V1']['median']:+.5f}")
        hindsight = [e["best_V2_minus_V1_in_hindsight"] for e in per_seed]
        print(f"  best V2 - V1 in hindsight {float(np.mean(hindsight)):+.5f} at penalty "
              f"{[e['best_penalty_in_hindsight'] for e in per_seed]} (diagnostic only)")

    record["curve"] = record["curves"]["MATCHED_BASIS"]        # the identifiability question
    record["null_control"] = ("NULL_CONTEXT_RESIDUAL_CONTROL = PASS"
                              if all(record["curves"][m]["0.0"]["modulation_shut_off"]
                                     for m in PLANTING_MODES)
                              else "NULL_CONTEXT_RESIDUAL_CONTROL = FAIL")
    record["pass_criteria"] = pass_criteria(record["curves"]["MATCHED_BASIS"])
    record["pass_criteria_free_basis_only"] = pass_criteria(record["curves"]["FREE_BASIS"])
    record["verdict_is_taken_from"] = "MATCHED_BASIS"

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record["runtime"] = {"timestamp": datetime.now(timezone.utc).isoformat(),
                         "python": platform.python_version(), "numpy": np.__version__}
    output.write_text(json.dumps(record, indent=2, sort_keys=True, default=float) + "\n")
    print(f"\n{record['null_control']}")
    print(json.dumps(record["pass_criteria"], indent=2, sort_keys=True, default=float))
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="V2 context identifiability positive control")
    parser.add_argument("--context", action="append", required=True,
                        help="NAME=build_directory, a training context")
    parser.add_argument("--held-context", dest="held_context", default=None,
                        help="NAME=build_directory for a real held context; only its "
                             "control-derived source states are read, never its responses")
    parser.add_argument("--config", required=True)
    parser.add_argument("--anchor-genes", dest="anchor_genes", required=True)
    parser.add_argument("--anchor-sources", dest="anchor_sources", required=True)
    parser.add_argument("--eligibility", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
