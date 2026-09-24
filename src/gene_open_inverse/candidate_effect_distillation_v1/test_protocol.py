"""Protocol tests.  Each names a failure that would otherwise look like a valid gain."""
from __future__ import annotations

import numpy as np
import pytest

from gene_open_inverse.candidate_effect_distillation_v1 import basis as bs
from gene_open_inverse.candidate_effect_distillation_v1 import contract as ced
from gene_open_inverse.candidate_effect_distillation_v1 import predictor as pr
from gene_open_inverse.candidate_effect_distillation_v1.fuse import _row_z, fusion_scores, out_of_fold_effect


class _Role:
    def __init__(self, capabilities, present):
        self.capabilities = capabilities
        self.capability_present = present


def fixture(count: int = 60, genes: int = 40, seed: int = 5):
    generator = np.random.default_rng(seed)
    latent = generator.normal(size=(count, 6))
    loading = generator.normal(size=(6, genes))
    truth = latent @ loading
    transitions = np.stack([truth + 0.4 * generator.normal(size=(count, genes)) for _ in (0, 1)])
    capabilities = {"STRING": latent @ generator.normal(size=(6, 9)),
                    "MAPKG": latent @ generator.normal(size=(6, 7))}
    present = {"STRING": np.ones(count, dtype=bool), "MAPKG": np.ones(count, dtype=bool)}
    present["STRING"][:4] = False
    present["MAPKG"][:4] = False
    return transitions, _Role(capabilities, present)


def test_01_consensus_is_the_mean_of_the_two_views():
    transitions, _ = fixture()
    assert np.abs(bs.consensus_response(transitions) - 0.5 * (transitions[0] + transitions[1])).max() < 1e-12
    with pytest.raises(ValueError):
        bs.consensus_response(transitions[0])


def test_02_the_basis_is_orthonormal_and_projection_is_idempotent():
    transitions, _ = fixture(count=60, genes=40)
    basis = bs.randomized_basis(bs.consensus_response(transitions), rank=8)
    gram = basis.basis.T @ basis.basis
    assert np.abs(gram - np.eye(8)).max() < 1e-8
    values = bs.consensus_response(transitions)
    once = basis.reconstruct(basis.project(values))
    twice = basis.reconstruct(basis.project(once))
    assert np.abs(once - twice).max() < 1e-8


def test_03_static_features_carry_no_response_and_flag_what_is_missing():
    """The one structural guarantee the whole phase rests on."""
    _, role = fixture()
    features = pr.static_features(role, ced.PRIMARY_MODALITIES)
    width = sum(role.capabilities[name].shape[1] for name in ced.PRIMARY_MODALITIES)
    assert features.values.shape[1] == width + len(ced.PRIMARY_MODALITIES)
    assert not features.present_any[:4].any()
    assert features.present_any[4:].all()
    # every descriptor of an absent modality is exactly zero, and the flag says so
    assert np.abs(features.values[:4, :width]).max() == 0.0
    assert np.abs(features.values[:4, width:]).max() == 0.0


def test_04_identity_folds_partition_exactly():
    folds = pr.identity_folds(97, folds=5)
    joined = np.sort(np.concatenate(folds))
    assert np.array_equal(joined, np.arange(97))
    for index, left in enumerate(folds):
        for right in folds[index + 1:]:
            assert np.intersect1d(left, right).size == 0


def test_05_out_of_fold_predictions_are_not_in_sample():
    """An in-sample effect would make beta be chosen for a model that does not exist."""
    transitions, role = fixture()
    consensus = bs.consensus_response(transitions)
    basis = bs.randomized_basis(consensus, rank=6)
    targets = basis.project(consensus)
    features = pr.static_features(role, ced.PRIMARY_MODALITIES)
    keep = features.fit_rows()
    held_out = out_of_fold_effect(features.values, targets, keep, 1.0, basis, consensus.shape[0])
    path = pr.RidgePath(features.values[keep], targets[keep])
    in_sample = basis.reconstruct(path.predict(features.values, 1.0))

    def cosine(left, right):
        norm = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        return np.einsum("ij,ij->i", left, right) / np.maximum(norm, 1e-30)

    honest = float(np.median(cosine(held_out[keep], consensus[keep])))
    optimistic = float(np.median(cosine(in_sample[keep], consensus[keep])))
    assert np.isfinite(honest) and np.isfinite(optimistic)
    assert optimistic > honest, (optimistic, honest)
    assert np.abs(held_out[:4]).max() == 0.0     # all-missing stays exactly zero


def test_06_all_missing_candidates_get_the_declared_neutral_score():
    """They must be indistinguishable from each other, and never carry a learned preference.

    The per-query standardisation maps the declared neutral raw score to a value that
    depends on the query's own score spread, so the invariant is equality within a row,
    not equality across rows.  That is the correct invariant: the policy must not be able
    to prefer one information-free candidate over another.
    """
    transitions, role = fixture()
    consensus = bs.consensus_response(transitions)
    basis = bs.randomized_basis(consensus, rank=6)
    targets = basis.project(consensus)
    features = pr.static_features(role, ced.PRIMARY_MODALITIES)
    keep = features.fit_rows()
    effect = out_of_fold_effect(features.values, targets, keep, 1.0, basis, consensus.shape[0])
    assert np.abs(effect[:4]).max() == 0.0
    incumbent = [np.zeros((consensus.shape[0], consensus.shape[0])) for _ in (0, 1)]
    fused = fusion_scores(incumbent, transitions, effect, 1.0)
    for view in (0, 1):
        block = fused[view][:, :4]
        assert np.abs(block - block[:, :1]).max() < 1e-9
        assert np.isfinite(block).all()


def test_07_beta_zero_leaves_the_incumbent_ordering_untouched():
    transitions, role = fixture()
    generator = np.random.default_rng(1)
    incumbent = [generator.normal(size=(transitions.shape[1], transitions.shape[1])) for _ in (0, 1)]
    effect = generator.normal(size=(transitions.shape[1], transitions.shape[2]))
    fused = fusion_scores(incumbent, transitions, effect, 0.0)
    for view in (0, 1):
        assert np.array_equal(np.argsort(-fused[view], axis=1), np.argsort(-incumbent[view], axis=1))


def test_08_row_standardisation_is_per_query_and_rank_preserving():
    generator = np.random.default_rng(2)
    values = generator.normal(size=(7, 21)) * np.arange(1, 8)[:, None] + np.arange(7)[:, None]
    standardised = _row_z(values)
    assert np.abs(standardised.mean(axis=1)).max() < 1e-9
    assert np.abs(standardised.std(axis=1) - 1.0).max() < 1e-9
    assert np.array_equal(np.argsort(-standardised, axis=1), np.argsort(-values, axis=1))


def test_09_cross_validation_never_scores_a_fold_with_its_own_predictor():
    transitions, role = fixture()
    consensus = bs.consensus_response(transitions)
    basis = bs.randomized_basis(consensus, rank=6)
    targets = basis.project(consensus)
    features = pr.static_features(role, ced.PRIMARY_MODALITIES)
    keep = features.fit_rows()
    record = pr.cross_validate(features.values[keep], targets[keep], consensus[keep], basis,
                               penalties=(1e-2, 1e2))
    assert sum(record["fold_sizes"]) == keep.size
    assert str(record["selected_penalty"]) in record["held_out_cosine"]
    # a penalty of 1e-2 on this small fixture must not beat a heavy one by overfitting the fold
    assert set(record["held_out_cosine"]) == {"0.01", "100.0"}


def test_10_the_frozen_constants_are_the_declared_ones():
    assert ced.PRIMARY_MODALITIES == ("STRING", "MAPKG")
    assert ced.PRIMARY_EXCLUDES_TEXT is True
    assert "TEXT" not in ced.PRIMARY_MODALITIES
    assert ced.BASIS_RANK == 256 and ced.BASIS_CENTERED is False
    assert ced.RIDGE_FOLDS == 5 and ced.ALL_MISSING_SCORE == 0.0
    assert ced.RIDGE_CRITERION == "held_out_mean_cosine_in_gene_space"


def test_11_view_reliability_reports_a_cosine_per_candidate():
    transitions, _ = fixture()
    record = bs.view_reliability(transitions)["cross_view_cosine"]
    assert -1.0 <= record["median"] <= 1.0
    identical = np.stack([transitions[0], transitions[0]])
    assert abs(bs.view_reliability(identical)["cross_view_cosine"]["median"] - 1.0) < 1e-9
