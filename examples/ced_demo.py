#!/usr/bin/env python3
"""Small synthetic VCDesign-CED smoke example; no biological result is reported."""
from __future__ import annotations

import numpy as np

from gene_open_inverse.candidate_effect_distillation_v1 import basis, contract, predictor
from gene_open_inverse.final_clean_model_v1.scoring import row_z


class Role:
    def __init__(self, capabilities: dict[str, np.ndarray], present: dict[str, np.ndarray]):
        self.capabilities = capabilities
        self.capability_present = present


def main() -> None:
    rng = np.random.default_rng(7)
    count, train_count, genes = 80, 64, 32
    latent = rng.normal(size=(count, 5))
    responses = latent @ rng.normal(size=(5, genes))
    capabilities = {
        "STRING": latent @ rng.normal(size=(5, 12)),
        "MAPKG": latent @ rng.normal(size=(5, 10)),
    }
    present = {name: np.ones(count, dtype=bool) for name in capabilities}
    for flag in present.values():
        flag[-2:] = False
    features = predictor.static_features(Role(capabilities, present), contract.PRIMARY_MODALITIES)

    fitted_basis = basis.randomized_basis(responses[:train_count], rank=8)
    targets = fitted_basis.project(responses[:train_count])
    path = predictor.RidgePath(features.values[:train_count], targets)
    effect = fitted_basis.reconstruct(path.predict(features.values[train_count:], penalty=1.0))
    effect[~features.present_any[train_count:]] = 0.0

    goal = responses[train_count] / np.linalg.norm(responses[train_count])
    norm = np.maximum(np.linalg.norm(effect, axis=1), 1e-30)
    effect_score = (effect @ goal) / norm
    base_score = rng.normal(size=effect_score.size)  # illustrative scorer output
    fused = row_z(base_score[None, :]) + row_z(effect_score[None, :])
    order = np.lexsort((np.arange(effect_score.size), -fused[0]))
    print("synthetic demo only; ranked candidate indices:", order[:5].tolist())
    print("all-missing raw effect scores:", effect_score[-2:].tolist())


if __name__ == "__main__":
    main()
