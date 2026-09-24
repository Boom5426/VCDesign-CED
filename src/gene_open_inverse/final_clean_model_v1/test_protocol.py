"""Protocol tests for the final locked model pair."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from gene_open_inverse.final_clean_model_v1 import contract as fc
from gene_open_inverse.final_clean_model_v1.model import CleanAttributionModel, CleanCapabilityEncoder
from gene_open_inverse.final_clean_model_v1.scoring import ACKNOWLEDGEMENT, fuse, locked_check_arrays, row_z


DIMS = {"STRING": 512, "MAPKG": 1024}


def test_01_the_clean_model_has_no_learnable_unknown_token():
    """The incumbent's shared token put 92 information-free genes at the top of every prior."""
    model = CleanAttributionModel(DIMS)
    assert not any(name.endswith("unknown_token") for name, _ in model.named_parameters())
    assert not hasattr(model.capability_encoder, "unknown_token")
    assert isinstance(model.capability_encoder, CleanCapabilityEncoder)


def test_02_all_missing_candidates_score_exactly_the_neutral_value_for_every_query():
    torch.manual_seed(0)
    model = CleanAttributionModel(DIMS).eval()
    count = 6
    descriptors = {"STRING": torch.randn(count, 512), "MAPKG": torch.randn(count, 1024)}
    masks = {"STRING": torch.tensor([1, 1, 0, 1, 0, 1]).bool(),
             "MAPKG": torch.tensor([1, 0, 0, 1, 0, 1]).bool()}
    with torch.no_grad():
        encoding = model.encode_candidates(descriptors, masks)
        query = model.encode_query(torch.randn(4, 8248), torch.randn(4, 8248))
        scores = model.score_embeddings(query, encoding)
    absent = [2, 4]
    assert encoding.embedding[absent].abs().max().item() == 0.0
    assert torch.allclose(scores[:, absent], torch.zeros_like(scores[:, absent]))
    assert float(scores[:, absent].abs().max()) == fc.ALL_MISSING_CONTRIBUTION
    present = [0, 1, 3, 5]
    assert torch.allclose(encoding.embedding[present].norm(dim=1), torch.ones(len(present)), atol=1e-5)


def test_03_a_candidate_with_one_modality_is_not_treated_as_missing():
    torch.manual_seed(1)
    model = CleanAttributionModel(DIMS).eval()
    descriptors = {"STRING": torch.randn(3, 512), "MAPKG": torch.randn(3, 1024)}
    masks = {"STRING": torch.tensor([1, 0, 0]).bool(), "MAPKG": torch.tensor([0, 1, 0]).bool()}
    with torch.no_grad():
        embedding = model.encode_candidates(descriptors, masks).embedding
    assert embedding[0].norm().item() > 0.5 and embedding[1].norm().item() > 0.5
    assert embedding[2].abs().max().item() == 0.0


def test_04_G_check_cannot_be_reached_without_the_explicit_acknowledgement():
    class _Assets:
        pass

    for wrong in ("", "yes", "G_check", ACKNOWLEDGEMENT.lower()):
        with pytest.raises(PermissionError):
            locked_check_arrays(_Assets(), wrong)


def test_05_the_frozen_loader_still_refuses_G_check_for_every_other_caller():
    from gene_open_inverse.model.global_objective_knowledge_attribution_v1.assets import (
        GlobalAttributionAssets,
    )
    import inspect

    source = inspect.getsource(GlobalAttributionAssets.role_arrays)
    assert 'role not in ("G_fit", "G_select")' in source
    assert "G_check is inaccessible" in source


def test_06_the_fusion_is_the_frozen_one_and_has_no_fitted_coefficient():
    assert fc.FINAL_FUSION == "z_score_additive" and fc.FINAL_BETA == 1.0
    assert fc.FINAL_FUSION_SELECTED_BEFORE_G_CHECK is True
    assert "Not selected on performance" in fc.FINAL_FUSION_SELECTION_REASON
    generator = np.random.default_rng(0)
    base = [generator.normal(size=(5, 11)) for _ in (0, 1)]
    effect = [generator.normal(size=(5, 11)) for _ in (0, 1)]
    fused = fuse(base, effect)
    for view in (0, 1):
        assert np.abs(fused[view] - (row_z(base[view]) + row_z(effect[view]))).max() < 1e-12


def test_07_row_standardisation_preserves_each_query_ordering():
    generator = np.random.default_rng(2)
    values = generator.normal(size=(6, 17)) * np.arange(1, 7)[:, None]
    standardised = row_z(values)
    assert np.array_equal(np.argsort(-standardised, axis=1), np.argsort(-values, axis=1))
    assert np.abs(standardised.mean(axis=1)).max() < 1e-9


def test_08_TEXT_is_excluded_from_the_final_primary_track():
    assert fc.PRIMARY_MODALITIES == ("STRING", "MAPKG")
    assert "TEXT" not in fc.PRIMARY_MODALITIES
    assert fc.TEXT_STATUS == "EXCLUDED_FROM_FINAL_PRIMARY_TRACK"
    assert fc.LEGACY_COMPARATOR_IS_NOT_PRIMARY is True
    assert fc.PRIMARY_COMPARISON == ("CLEAN_EFFECT", "CLEAN_BASE")


def test_09_the_split_is_named_honestly():
    assert fc.EVALUATION_SPLIT_NAME == "locked final evaluation split"
    assert fc.EVALUATION_SPLIT_MUST_NOT_BE_CALLED == "untouched blind test"
    assert fc.NO_CONFIG_CHANGE_ALLOWED_AFTER_G_CHECK_ACCESS is True


def test_10_strict_legal_restriction_drops_TEXT_and_keeps_the_rest():
    from gene_open_inverse.final_clean_model_v1.scoring import strict_legal_role
    from gene_open_inverse.model.global_objective_knowledge_attribution_v1.assets import (
        AttributionRoleArrays,
    )

    role = AttributionRoleArrays(
        base=object(), source=np.zeros((2, 3, 4)), goal=np.zeros((2, 3, 4)),
        capabilities={"STRING": np.ones((3, 2)), "MAPKG": np.ones((3, 5)), "TEXT": np.ones((3, 7))},
        capability_present={name: np.ones(3, dtype=bool) for name in ("STRING", "MAPKG", "TEXT")})
    restricted = strict_legal_role(role)
    assert set(restricted.capabilities) == {"STRING", "MAPKG"}
    assert set(restricted.capability_present) == {"STRING", "MAPKG"}
    assert restricted.capabilities["MAPKG"].shape == (3, 5)


def test_11_a_role_missing_a_strict_legal_modality_is_an_error():
    from gene_open_inverse.final_clean_model_v1.scoring import strict_legal_role
    from gene_open_inverse.model.global_objective_knowledge_attribution_v1.assets import (
        AttributionRoleArrays,
    )

    role = AttributionRoleArrays(
        base=object(), source=np.zeros((2, 3, 4)), goal=np.zeros((2, 3, 4)),
        capabilities={"STRING": np.ones((3, 2))},
        capability_present={"STRING": np.ones(3, dtype=bool)})
    with pytest.raises(RuntimeError):
        strict_legal_role(role)


def test_12_the_modality_restricted_bank_is_undefined_without_an_unknown_token():
    model = CleanAttributionModel(DIMS)
    with pytest.raises(NotImplementedError):
        model.encode_modality_candidates({}, {})
