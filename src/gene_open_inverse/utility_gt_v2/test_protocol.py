"""Each test targets a specific way this phase could report a valid-looking lie."""
from __future__ import annotations

import numpy as np
import pytest

from gene_open_inverse.model.decision_alignment_v1.contract import (
    BUDGETS, mbru_from_nru, normalized_ranked_utility, query_references,
)
from gene_open_inverse.utility_gt_v2 import contract as gt
from gene_open_inverse.utility_gt_v2 import validate as val


def _truth(count: int, features: int, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.normal(size=(count, features))


def test_01_latent_value_is_unbiased_while_a_same_view_form_is_not():
    """The estimator must converge to the latent value, not to value minus noise energy.

    A same-view form pays ``-||eta||^2`` on the candidate self term, which biases
    every candidate downward by roughly ``features * sigma^2`` and therefore
    changes which candidates clear zero.
    """
    truth = _truth(40, 300, 11)
    sigma, repeats = 0.7, 400
    exact = 2.0 * (truth @ truth.T) - np.diag(truth @ truth.T)[None, :]
    generator = np.random.default_rng(12)
    cross = np.zeros_like(exact)
    naive = np.zeros_like(exact)
    for _ in range(repeats):
        first = truth + generator.normal(scale=sigma, size=truth.shape)
        second = truth + generator.normal(scale=sigma, size=truth.shape)
        cross += gt.latent_value(first, second)
        gram_same = first @ first.T
        naive += 2.0 * gram_same - np.diag(gram_same)[None, :]
    cross /= repeats
    naive /= repeats
    expected_bias = truth.shape[1] * sigma ** 2
    off_diagonal = ~np.eye(truth.shape[0], dtype=bool)
    assert np.abs(cross - exact)[off_diagonal].mean() < 0.12 * expected_bias
    assert np.abs(np.mean((naive - exact)[off_diagonal]) + expected_bias) < 0.05 * expected_bias


def test_02_value_is_energy_minus_squared_endpoint_distance():
    truth = _truth(25, 60, 21)
    other = _truth(25, 60, 22)
    value = gt.latent_value(truth, other)
    gram = gt.cross_view_gram(truth, other)
    energy = np.diag(gram)
    distance = gram[:, :].diagonal()[None, :] + energy[:, None] - gram.T - gram
    assert np.allclose(value, energy[:, None] - distance)


def test_03_self_value_equals_the_transition_energy():
    truth = _truth(18, 40, 31)
    other = _truth(18, 40, 32)
    value = gt.latent_value(truth, other)
    assert np.allclose(np.diag(value), np.diag(gt.cross_view_gram(truth, other)))


def test_04_nru_is_invariant_to_a_positive_affine_transform_of_the_utility_row():
    """This is why dropping the no-op denominator is metric-neutral.

    The old normalization divided by a per-query constant, so it never changed a
    within-query ranking or an nRU; its only effect was to invert the row when
    that constant came out negative.  If this invariance did not hold, the new
    value form would not be comparable to any historical number at all.
    """
    generator = np.random.default_rng(41)
    utility = generator.normal(size=200)
    ids = np.asarray([f"g{index:04d}" for index in range(200)])
    order = np.argsort(-utility, kind="stable")
    order = order[order != 0]
    for scale, shift in ((2.5, -7.0), (0.5, 3.25), (13.0, 0.0)):
        base = mbru_from_nru(normalized_ranked_utility(utility[order], query_references(utility, ids, 0)))
        moved = scale * utility + shift
        transformed = mbru_from_nru(normalized_ranked_utility(moved[order], query_references(moved, ids, 0)))
        assert abs(base - transformed) < 1e-9


def test_05_the_retired_no_op_normalization_inverts_a_row_with_negative_energy():
    """The exact failure the new form removes by construction, stated as a test."""
    utility = np.asarray([3.0, 1.0, -2.0])
    for energy, expect_preserved in ((2.0, True), (-2.0, False)):
        normalized = utility / energy
        preserved = np.array_equal(np.argsort(-normalized), np.argsort(-utility))
        assert preserved is expect_preserved


def test_06_permutation_p_value_matches_a_brute_force_loop_and_has_a_resolution_floor():
    gram = _truth(30, 30, 51) @ _truth(30, 30, 52).T
    fast = gt.permutation_p_value(gram)
    slow = np.empty(30)
    for row in range(30):
        others = [gram[row, column] for column in range(30) if column != row]
        slow[row] = (1 + sum(1 for value in others if value >= gram[row, row])) / 30.0
    assert np.allclose(fast, slow)
    assert fast.min() >= 1.0 / 30.0


def test_07_benjamini_hochberg_matches_a_direct_step_up_reference():
    generator = np.random.default_rng(61)
    p_value = np.clip(generator.random(500) ** 2, 1e-6, 1.0)
    q_value = gt.benjamini_hochberg(p_value)
    for level in (0.01, 0.05, 0.10, 0.20):
        ranked = np.sort(p_value)
        admissible = np.flatnonzero(ranked <= np.arange(1, 501) * level / 500)
        expected = 0 if admissible.size == 0 else int(admissible[-1]) + 1
        assert int((q_value <= level).sum()) == expected


def test_08_eligibility_needs_both_significance_and_a_positive_energy():
    gram = np.diag(np.asarray([5.0, -5.0, 0.01]))
    gram[0, 1] = gram[0, 2] = -9.0
    gram[1, 0] = gram[1, 2] = -9.0
    gram[2, 0] = gram[2, 1] = -9.0
    decided = gt.eligibility(gram, level=1.0)
    assert decided.passes_fdr.all()
    assert decided.eligible.tolist() == [True, False, True]
    strict = gt.eligibility(gram, level=0.0)
    assert not strict.eligible.any()


def test_09_bh_cutoff_agrees_with_the_q_value_decision():
    generator = np.random.default_rng(71)
    p_value = np.clip(generator.random(400) ** 3, 1e-5, 1.0)
    for level in (0.01, 0.05, 0.20):
        cutoff = gt.bh_p_cutoff(p_value, level)
        selected = gt.benjamini_hochberg(p_value) <= level
        assert np.array_equal(selected, p_value <= cutoff) or (cutoff == 0.0 and not selected.any())


def test_10_orderings_and_grading_never_contain_the_query_itself():
    """Self-exclusion is by position, so a tie at the top cannot restore the target."""
    value = np.zeros((6, 6))
    value[np.arange(6), np.arange(6)] = 100.0
    ids = np.asarray([f"g{index}" for index in range(6)])
    self_position = np.arange(6)
    order = val.legal_orders(value, ids, self_position)
    assert order.shape == (6, 5)
    for row in range(6):
        assert row not in order[row].tolist()


def test_11_goal_independent_prior_is_one_ordering_shared_by_every_query():
    generator = np.random.default_rng(81)
    value = generator.normal(size=(12, 12))
    ids = np.asarray([f"g{index:02d}" for index in range(12)])
    order = val.goal_independent_order(value, ids, np.arange(12))
    reference = [index for index in order[0].tolist()]
    for row in range(1, 12):
        assert [index for index in order[row].tolist() if index != 0] == [
            index for index in reference if index != row
        ]


def test_12_a_perfect_replicate_scores_one_and_an_unrelated_one_scores_near_zero():
    generator = np.random.default_rng(91)
    value = generator.normal(size=(60, 60))
    ids = np.asarray([f"g{index:03d}" for index in range(60)])
    self_position = np.arange(60)
    order = val.legal_orders(value, ids, self_position)
    identical = val.grade(order, value, ids, self_position)
    assert np.allclose(identical.mbru, 1.0)
    unrelated = generator.normal(size=(60, 60))
    shuffled = val.grade(val.legal_orders(unrelated, ids, self_position), value, ids, self_position)
    assert abs(float(np.mean(shuffled.mbru))) < 0.15


def test_13_cluster_bootstrap_brackets_a_known_shift():
    generator = np.random.default_rng(101)
    difference = generator.normal(loc=0.30, scale=0.5, size=800)
    record = val.cluster_bootstrap(difference, replicates=2000, seed=7)
    low, high = record["mean"]["ci95"]
    assert low < 0.30 < high
    assert low > 0.0
    centred = val.cluster_bootstrap(generator.normal(loc=0.0, scale=0.5, size=800), replicates=2000, seed=7)
    assert centred["mean"]["ci95"][0] < 0.0 < centred["mean"]["ci95"][1]


def test_14_boundary_flip_is_a_half_when_the_grading_view_is_pure_noise():
    generator = np.random.default_rng(111)
    proposing = generator.normal(size=(400, 200))
    grading = generator.normal(size=(400, 200))
    ids = np.asarray([f"g{index:03d}" for index in range(200)])
    order = val.legal_orders(proposing, ids, -np.ones(400, dtype=np.int64))
    record = val.boundary_flip(order, proposing, grading)
    for budget in BUDGETS:
        assert abs(record[str(budget)]["independent_view_reverses"] - 0.5) < 0.07


def test_15_physical_bound_violation_is_exactly_value_above_energy():
    value = np.asarray([[1.0, 3.0], [0.5, -0.5]])
    energy = np.asarray([2.0, 0.0])
    assert gt.physical_bound_violation(value, energy).tolist() == [[False, True], [True, False]]


def test_16_the_module_declares_no_gate_selected_from_the_validation_pool():
    assert gt.GATE_SELECTED_FROM_G_SELECT is False
    assert gt.CALIBRATION_ROLE == "G_fit"
    assert gt.PRIMARY_FDR == 0.05
    assert gt.METRIC_SPACE == "IDENTITY"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
