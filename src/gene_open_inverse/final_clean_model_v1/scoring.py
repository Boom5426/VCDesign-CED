"""Shared scoring, the frozen effect predictor, and the one explicit G_check accessor.

The effect predictor is recomputed from ``G_fit`` rather than loaded, and then
checked against the artifact the distillation phase saved.  Recomputing makes the
locked evaluation self-contained; checking makes it provably the same predictor.

``locked_check_arrays`` is the only way anything in this repository reaches
``G_check``.  ``GlobalAttributionAssets.role_arrays`` still refuses that role for
every other caller, and this function requires the caller to pass an explicit
acknowledgement, so it cannot be reached by accident or by a default argument.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from ..candidate_effect_distillation_v1 import basis as bs
from ..candidate_effect_distillation_v1 import contract as ced
from ..candidate_effect_distillation_v1 import predictor as pr
from ..model.global_objective_knowledge_attribution_v1.assets import AttributionRoleArrays
from . import contract as fc


ACKNOWLEDGEMENT = "I_AM_PERFORMING_THE_ONE_LOCKED_FINAL_EVALUATION"


def strict_legal_role(role: AttributionRoleArrays) -> AttributionRoleArrays:
    """Drop TEXT before any model can see it.

    ``GlobalAttributionAssets`` only admits the STRING or STRING+MAPKG+TEXT
    configurations, which is a phase-specific arm enumeration rather than an
    information rule, and it is frozen.  So the loader is used exactly as frozen and
    the restriction happens here instead: the returned arrays carry only the strict
    legal modalities, the model is constructed over those dimensions, and TEXT never
    reaches a tensor.  Nothing frozen is edited.
    """
    missing = set(fc.PRIMARY_MODALITIES) - set(role.capabilities)
    if missing:
        raise RuntimeError(f"the loader did not supply the strict legal modalities: {sorted(missing)}")
    restricted = AttributionRoleArrays(
        base=role.base, source=role.source, goal=role.goal,
        capabilities={name: role.capabilities[name] for name in fc.PRIMARY_MODALITIES},
        capability_present={name: role.capability_present[name] for name in fc.PRIMARY_MODALITIES})
    if "TEXT" in restricted.capabilities or "TEXT" in restricted.capability_present:
        raise RuntimeError("TEXT survived the strict legal restriction")
    return restricted


def strict_legal_dims(assets) -> dict[str, int]:
    return {name: int(assets.capability_dims[name]) for name in fc.PRIMARY_MODALITIES}


def locked_check_arrays(assets, acknowledgement: str) -> AttributionRoleArrays:
    """Assemble the ``G_check`` role arrays.  Deliberately awkward to call."""
    if acknowledgement != ACKNOWLEDGEMENT:
        raise PermissionError(
            "G_check is sealed; this accessor exists only for the single locked final evaluation "
            f"and requires the explicit acknowledgement {ACKNOWLEDGEMENT!r}")
    role = "G_check"
    base = assets.base.role_arrays(role)
    indices = assets.base.role_indices[role]
    source = np.asarray(assets.source[:, indices, :], dtype=np.float32)
    goal = np.asarray(assets.goal[:, indices, :], dtype=np.float32)
    if not np.allclose(goal - source, base.transitions, atol=1e-5, rtol=1e-5):
        raise RuntimeError("source/goal delta invariant failed for the locked evaluation role")
    return AttributionRoleArrays(
        base=base, source=source, goal=goal,
        capabilities={name: assets.values[name][indices].copy() for name in assets.modalities},
        capability_present={name: assets.present[name][indices].copy() for name in assets.modalities})


class FrozenEffectPredictor:
    """The distillation predictor, rebuilt deterministically and then verified."""

    def __init__(self, fit_role, penalty: float):
        self.penalty = float(penalty)
        consensus = bs.consensus_response(np.asarray(fit_role.base.transitions, dtype=np.float64))
        self.basis = bs.randomized_basis(consensus)
        targets = self.basis.project(consensus)
        features = pr.static_features(fit_role, ced.PRIMARY_MODALITIES)
        self.keep = features.fit_rows()
        self.path = pr.RidgePath(features.values[self.keep], targets[self.keep])

    def effect(self, role) -> np.ndarray:
        features = pr.static_features(role, ced.PRIMARY_MODALITIES)
        predicted = self.basis.reconstruct(self.path.predict(features.values, self.penalty))
        predicted[~features.present_any] = 0.0
        return predicted

    def verify_against(self, saved: np.ndarray, role) -> float:
        return float(np.abs(self.effect(role) - np.asarray(saved, dtype=np.float64)).max())


def unit_rows(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-30)


def effect_scores(role, effect: np.ndarray) -> list[np.ndarray]:
    """``cos(d_q, r_hat_c)`` per view, with the declared neutral for all-missing candidates."""
    transitions = np.asarray(role.base.transitions, dtype=np.float64)
    missing = np.linalg.norm(effect, axis=1) <= 0.0
    unit = unit_rows(effect)
    output = []
    for view in (0, 1):
        block = unit_rows(transitions[view]) @ unit.T
        block[:, missing] = fc.ALL_MISSING_CONTRIBUTION
        output.append(block)
    return output


def row_z(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    return (values - values.mean(axis=1, keepdims=True)) / np.maximum(
        values.std(axis=1, keepdims=True), 1e-12)


def base_scores(model, role, device, batch: int = 64) -> list[np.ndarray]:
    model.eval()
    source = torch.from_numpy(role.source).to(device)
    goal = torch.from_numpy(role.goal).to(device)
    descriptors = {name: torch.from_numpy(value).to(device) for name, value in role.capabilities.items()}
    masks = {name: torch.from_numpy(value).to(device) for name, value in role.capability_present.items()}
    output = []
    with torch.no_grad():
        candidates = model.encode_candidates(descriptors, masks)
        for view in (0, 1):
            rows = []
            for start in range(0, source.shape[1], batch):
                query = model.encode_query(source[view, start:start + batch], goal[view, start:start + batch])
                rows.append(model.score_embeddings(query, candidates).float().cpu())
            output.append(torch.cat(rows).numpy().astype(np.float64))
    return output


def fuse(base: list[np.ndarray], effect: list[np.ndarray]) -> list[np.ndarray]:
    """The frozen fusion: z-score additive, beta = 1, no fitted coefficient."""
    return [row_z(base[view]) + fc.FINAL_BETA * row_z(effect[view]) for view in (0, 1)]
