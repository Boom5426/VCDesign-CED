"""Data-free invariant tests for CANDIDATE_KNOWLEDGE_V1."""
import numpy as np

from .common import exact_softndcg10_rows, ridge_path, shuffled_donor_index


def test_ridge_path_matches_augmented_closed_form():
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=(31, 7)), rng.normal(size=(31, 3))
    alpha = 2.5
    (w, b) = ridge_path(x, y, [alpha])[0][alpha]
    xc, yc = x - x.mean(0), y - y.mean(0)
    expected = np.linalg.solve(xc.T @ xc + alpha * np.eye(x.shape[1]), xc.T @ yc)
    assert np.max(np.abs(w - expected)) < 1e-11
    assert np.max(np.abs(b - (y.mean(0) - x.mean(0) @ expected))) < 1e-11


def test_metric_origin_deletion_and_pessimistic_ties():
    r = np.asarray([[0, 1, .5], [1, 0, .2], [.5, .2, 0]], np.float64)
    s = np.zeros_like(r)
    got = exact_softndcg10_rows(r, s)
    # With equal scores, lower relevance ranks first; the diagonal is always excluded.
    d = 1 / np.log2(np.arange(2, 4, dtype=np.float64))
    expected0 = (.5 * d[0] + 1 * d[1]) / (1 * d[0] + .5 * d[1])
    assert abs(got[0] - expected0) < 1e-12


def test_shuffle_uses_only_fit_donors_and_deranges_fit():
    universe = np.asarray(["A", "B", "C", "D", "E", "F"])
    fit = ["A", "C", "E", "F"]
    donor = shuffled_donor_index(universe, fit, 9)
    pos = np.asarray([0, 2, 4, 5])
    assert set(donor.tolist()) <= set(pos.tolist())
    assert np.all(donor[pos] != pos)
    assert np.array_equal(donor, shuffled_donor_index(universe, fit, 9))
