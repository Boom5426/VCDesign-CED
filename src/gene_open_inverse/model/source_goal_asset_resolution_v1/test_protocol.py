"""Data-free protocol tests for SOURCE_GOAL_ASSET_RESOLUTION_V1 helpers."""
from __future__ import annotations

import unittest

import numpy as np

from .audit_pairing_semantics import _cache_comparison, _source_summary
from .materialize_source_goal import _add_grouped, _source_weights


class AssetResolutionHelperTests(unittest.TestCase):
    def test_grouped_accumulation(self) -> None:
        values = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        groups = np.asarray([1, 0, 1], dtype=np.int64)
        dest = np.zeros((2, 2), dtype=np.float32)
        count = np.zeros(2, dtype=np.int64)
        _add_grouped(values, groups, dest, count)
        np.testing.assert_allclose(dest, [[3.0, 4.0], [6.0, 8.0]])
        np.testing.assert_array_equal(count, [1, 2])

    def test_source_weights_are_view_specific_and_normalized(self) -> None:
        incidence = np.asarray([[2, 1], [1, 3]], dtype=np.int32)
        a, b = _source_weights(incidence, np.asarray([0, 1], dtype=np.int8))
        np.testing.assert_allclose(a, [[1.0, 0.0], [1.0, 0.0]])
        np.testing.assert_allclose(b, [[0.0, 1.0], [0.0, 1.0]])

    def test_cache_comparison_and_full_rank_source_summary(self) -> None:
        source = np.asarray([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])
        goals = source + 0.25
        cache = np.full((2, 2, 2), 0.25, dtype=np.float32)
        comparison = _cache_comparison(goals, [source, source], cache, atol=1e-7, rtol=1e-7)
        self.assertTrue(all(x["all_close"] for method in comparison.values() for x in method.values()))
        summary = _source_summary(np.eye(2), np.eye(2), 2, 2, pair_seed=1, pair_count=4)
        self.assertEqual(summary["distinct_source_expression_profiles_float32"], 2)
        self.assertTrue(summary["source_profile_identity_proven"])


if __name__ == "__main__":
    unittest.main()
