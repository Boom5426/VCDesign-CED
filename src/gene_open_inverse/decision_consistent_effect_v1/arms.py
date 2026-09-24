"""Two predictors per regime, four scoring arms, one goal-blind control.

The arms are built so that every paired contrast changes exactly one thing.  A0 and A3
read one raw-target prediction and differ only in whether it is normalized before the
inner product; A1 and A2 read one unit-target prediction with the same relation.  No
scoring arm fits a parameter, so a difference between two arms that share a predictor
cannot come from anything but the normalization.
"""
from __future__ import annotations

import numpy as np

from ..four_context_v1 import evaluate as ev
from ..four_context_v1 import predictors as pd
from . import contract as dc

ZERO_NORM = 1e-12


def unit_target_pool(pool: dict) -> dict:
    """The training pool with each consensus response replaced by its own direction.

    ``u_c = r_c / ||r_c||`` on the same rows, with the same features, presence and keys.
    A present row whose consensus response is exactly zero has no direction to learn,
    and silently dividing by a floor would hand the ridge a zero target that the raw arm
    does not see as special; so it is refused rather than floored.
    """
    responses = np.asarray(pool["responses"], dtype=np.float64)
    norms = np.linalg.norm(responses, axis=1)
    present = np.asarray(pool["present"], dtype=bool)
    if (norms[present] <= ZERO_NORM).any():
        raise ValueError(f"{int((norms[present] <= ZERO_NORM).sum())} present training rows "
                         "have a zero consensus response and no direction")
    unit = responses / np.maximum(norms, ZERO_NORM)[:, None]
    return {**pool, "responses": unit}


def fit_both(pool: dict, training: tuple[str, ...]) -> tuple:
    """The raw-target predictor of record and its unit-target counterpart.

    Both go through ``four_context_v1.predictors.fit`` unchanged: the same basis rank,
    oversampling, power iterations and seed, the same penalty grid and the same held-out
    gene-space cosine criterion.  The unit arm re-selects its penalty through that
    procedure because its target changed; nothing else about the recipe does.
    """
    raw = pd.fit(pool["features"], pool["responses"], pool["present"], pool["keys"], training)
    unit_pool = unit_target_pool(pool)
    unit = pd.fit(unit_pool["features"], unit_pool["responses"], unit_pool["present"],
                  unit_pool["keys"], training)
    return raw, unit


def predictor_record(model) -> dict:
    return {"rows": int(model.rows), "identities": int(model.identities),
            "penalty": float(model.penalty),
            "held_out_gene_space_cosine": float(model.held_out_cosine),
            "explained_energy": float(model.basis.explained),
            "intercept_norm": float(np.linalg.norm(
                model.basis.reconstruct(model.path.target_mean[None, :])[0]))}


def cosine_scores(pack, effect: np.ndarray) -> list:
    """``cos(u_q, e_c)``, the frozen effect score; a candidate with no effect scores zero."""
    return [ev.effect_cosine(pack.query_responses, effect, view) for view in (0, 1)]


def dot_scores(pack, effect: np.ndarray) -> list:
    """``<u_q, e_c> = ||e_c|| cos(u_q, e_c)``.

    The query is normalized exactly as in the frozen cosine, so the two scores of one
    predictor differ by the per-candidate factor ``||e_c||`` and by nothing else.  A
    candidate with no legal modality has an all-zero effect and scores exactly zero,
    the same neutral value the cosine arm gives it.
    """
    effect = np.asarray(effect, dtype=np.float64)
    return [ev.unit(np.asarray(pack.query_responses[view], dtype=np.float64)) @ effect.T
            for view in (0, 1)]


def norm_scores(pack, effect: np.ndarray) -> list:
    """``||e_c||`` for every query: the D3 goal-blind control, never a designer."""
    norms = np.linalg.norm(np.asarray(effect, dtype=np.float64), axis=1)
    queries = int(np.asarray(pack.query_rows).size)
    return [np.repeat(norms[None, :], queries, axis=0) for _ in (0, 1)]


def arm_scores(pack, raw_effect: np.ndarray, unit_effect: np.ndarray) -> dict:
    """``{mode: {arm: [view0, view1]}}`` for the four arms and the norm control."""
    effect_only = {
        dc.A0: cosine_scores(pack, raw_effect),
        dc.A1: cosine_scores(pack, unit_effect),
        dc.A2: dot_scores(pack, unit_effect),
        dc.A3: dot_scores(pack, raw_effect),
        dc.NORM_ONLY: norm_scores(pack, unit_effect),
    }
    fused = {name: [ev.fuse(pack.base[view], scores[view]) for view in (0, 1)]
             for name, scores in effect_only.items()}
    return {"effect_only": effect_only, "fused": fused}


def norm_homogeneity(effects: list, rows: np.ndarray) -> dict:
    """Can these fold predictors' norms share one score row.

    Each predictor scores the same candidate rows, so a difference in their median norm
    is a property of the fold's fit and not of any candidate.
    """
    medians = [float(np.median(np.linalg.norm(np.asarray(effect, dtype=np.float64)[rows], axis=1)))
               for effect in effects]
    ratio = max(medians) / max(min(medians), 1e-30)
    return {"median_norm_by_fold": medians, "ratio": float(ratio),
            "tolerance": dc.NORM_HOMOGENEITY_TOLERANCE,
            "passes": bool(ratio <= dc.NORM_HOMOGENEITY_TOLERANCE)}
