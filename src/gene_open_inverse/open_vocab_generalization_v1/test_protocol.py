#!/usr/bin/env python3
"""Every test here names a specific failure that would otherwise look like a valid result.

The defect classes this mission is exposed to are not generic coding errors.  They are:
a masked candidate whose rows survive in one of the three training contexts, a leverage
that silently drops its null-space term and so declares every candidate well supported,
a matched comparison whose two pools end up different sizes, a cached effect cosine that
is not the cosine of the effect it claims to be, and a support diagnostic leaking into a
model as a feature.  Each of those produces numbers that look entirely reasonable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gene_open_inverse.four_context_v1 import contract as fc            # noqa: E402
from gene_open_inverse.four_context_v1 import evaluate as ev            # noqa: E402
from gene_open_inverse.four_context_v1 import predictors as pd          # noqa: E402
from gene_open_inverse.open_vocab_generalization_v1 import contract as ov      # noqa: E402
from gene_open_inverse.open_vocab_generalization_v1 import masking as mk       # noqa: E402
from gene_open_inverse.open_vocab_generalization_v1 import mixture as mx       # noqa: E402
from gene_open_inverse.open_vocab_generalization_v1 import natural as nt       # noqa: E402
from gene_open_inverse.open_vocab_generalization_v1 import support as sp       # noqa: E402


RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(condition), detail))


def _features(count: int, width: int = ov.FEATURE_WIDTH, seed: int = 0) -> np.ndarray:
    generator = np.random.default_rng(seed)
    values = generator.standard_normal((count, width))
    values[:, ov.STRING_FLAG_COLUMN] = (generator.random(count) < 0.8).astype(float)
    values[:, ov.MAPKG_FLAG_COLUMN] = (generator.random(count) < 0.6).astype(float)
    return values


# --- 01 to 04  the masking unit ---------------------------------------------
def test_01_fold_assignment_is_identity_only() -> None:
    ids = np.asarray([f"GENE{i}" for i in range(500)])
    first = mk.fold_assignment(ids)
    shuffled = ids[::-1]
    second = mk.fold_assignment(shuffled)[::-1]
    check("01 fold assignment depends on the symbol alone",
          np.array_equal(first, second) and set(first.tolist()) == set(range(ov.MASK_FOLDS)),
          f"sizes {np.bincount(first, minlength=ov.MASK_FOLDS).tolist()}")


def test_02_fold_assignment_is_stable_across_contexts() -> None:
    shared = np.asarray(["AARS1", "TP53", "MYC"])
    check("02 the same identity is masked in the same fold everywhere",
          np.array_equal(mk.fold_assignment(shared),
                         mk.fold_assignment(np.asarray(["TP53", "MYC", "AARS1"]))[[2, 0, 1]]))


class _Pack:
    def __init__(self, ids, features, responses, present, fit_mask):
        self.ids, self.features, self.responses = ids, features, responses
        self.present, self.fit_mask = present, fit_mask


def _pack(name: str, ids: list[str], seed: int, genes: int = 12) -> _Pack:
    generator = np.random.default_rng(seed)
    ids = np.asarray(ids, dtype=str)
    return _Pack(ids, _features(ids.size, seed=seed),
                 generator.standard_normal((2, ids.size, genes)),
                 np.ones(ids.size, dtype=bool), np.ones(ids.size, dtype=bool))


def test_03_masking_removes_every_context_row() -> None:
    packs = {"A": _pack("A", ["g1", "g2", "g3"], 1), "B": _pack("B", ["g2", "g3", "g4"], 2)}
    pool = mk.masked_training_pool(packs, ("A", "B"), np.asarray(["g2"]))
    identities = [key.split("|", 1)[1] for key in pool["keys"].tolist()]
    check("03 a masked identity keeps no training row in any context",
          "g2" not in identities and len(identities) == 4, f"{identities}")


def test_04_masking_leaves_the_other_rows_untouched() -> None:
    packs = {"A": _pack("A", ["g1", "g2", "g3"], 1), "B": _pack("B", ["g2", "g3", "g4"], 2)}
    from gene_open_inverse.four_context_v1 import harness as hn
    full = hn.training_pool(packs, ("A", "B"))
    masked = mk.masked_training_pool(packs, ("A", "B"), np.asarray(["g2"]))
    keep = np.asarray([key.split("|", 1)[1] != "g2" for key in full["keys"].tolist()])
    check("04 masking changes nothing about the rows it keeps",
          np.array_equal(full["features"][keep], masked["features"])
          and np.array_equal(full["responses"][keep], masked["responses"])
          and np.array_equal(full["keys"][keep], masked["keys"]))


# --- 05 to 08  leverage -------------------------------------------------------
def test_05_leverage_matches_the_explicit_inverse() -> None:
    generator = np.random.default_rng(5)
    features = generator.standard_normal((300, 20))
    targets = generator.standard_normal((300, 4))
    path = pd.WeightedRidgePath(features, targets)
    query = generator.standard_normal((7, 20))
    penalty = 3.0
    scaled = (features - path.feature_mean) / path.feature_scale
    centred = (query - path.feature_mean) / path.feature_scale
    inverse = np.linalg.inv(scaled.T @ scaled + penalty * np.eye(20))
    expected = np.einsum("ij,jk,ik->i", centred, inverse, centred)
    got = sp.ridge_leverage(path, query, penalty)
    check("05 leverage equals the explicit ridge inverse form",
          np.allclose(got, expected, rtol=1e-9, atol=1e-12),
          f"max abs error {np.abs(got - expected).max():.3e}")


def test_06_leverage_keeps_the_null_space_term() -> None:
    generator = np.random.default_rng(6)
    features = generator.standard_normal((12, 40))          # far fewer rows than features
    targets = generator.standard_normal((12, 3))
    path = pd.WeightedRidgePath(features, targets)
    query = generator.standard_normal((5, 40))
    penalty = 2.0
    scaled = (features - path.feature_mean) / path.feature_scale
    centred = (query - path.feature_mean) / path.feature_scale
    inverse = np.linalg.inv(scaled.T @ scaled + penalty * np.eye(40))
    expected = np.einsum("ij,jk,ik->i", centred, inverse, centred)
    got = sp.ridge_leverage(path, query, penalty)
    spanned_only = ((centred @ path.right.T) ** 2 / (path.singular ** 2 + penalty)).sum(axis=1)
    check("06 leverage keeps the unspanned term when p exceeds n",
          np.allclose(got, expected, rtol=1e-8, atol=1e-10)
          and not np.allclose(spanned_only, expected, rtol=1e-3),
          f"with null space {np.abs(got - expected).max():.3e}, "
          f"without {np.abs(spanned_only - expected).max():.3e}")


def test_07_leverage_grows_away_from_the_training_span() -> None:
    generator = np.random.default_rng(7)
    base = generator.standard_normal((200, 3))
    features = np.concatenate([base, np.zeros((200, 2))], axis=1) + 1e-3 * generator.standard_normal((200, 5))
    path = pd.WeightedRidgePath(features, generator.standard_normal((200, 2)))
    inside = np.concatenate([generator.standard_normal((1, 3)), np.zeros((1, 2))], axis=1)
    outside = np.concatenate([np.zeros((1, 3)), generator.standard_normal((1, 2))], axis=1)
    values = sp.ridge_leverage(path, np.concatenate([inside, outside]), 1.0)
    check("07 an unspanned direction reads as higher leverage", values[1] > values[0],
          f"spanned {values[0]:.4f} unspanned {values[1]:.4f}")


def test_08_modality_flags_round_trip() -> None:
    features = _features(50, seed=8)
    string, mapkg = sp.modality_flags(features)
    check("08 present flags are read back from the frozen layout, not re-derived",
          np.array_equal(string, features[:, ov.STRING_FLAG_COLUMN] > 0.5)
          and np.array_equal(mapkg, features[:, ov.MAPKG_FLAG_COLUMN] > 0.5)
          and set(sp.coverage_pattern(string, mapkg).tolist()) <= set(ov.COVERAGE_PATTERNS))


# --- 09 to 11  graph and neighbourhood support --------------------------------
def test_09_string_neighbourhood_counts_both_endpoints() -> None:
    adjacency = {"genes": np.asarray(["a", "b", "c", "d"]),
                 "degree": np.asarray([2, 1, 1, 0]),
                 "edge_row": np.asarray([0, 0]), "edge_col": np.asarray([1, 2]),
                 "edge_w": np.asarray([0.5, 0.25])}
    record = sp.string_neighbourhood(np.asarray(["a", "b", "c", "d", "zz"]), adjacency,
                                     np.asarray(["b"]))
    check("09 undirected edges are counted from both ends",
          record["training_neighbours"].tolist() == [1, 0, 0, 0, 0]
          and abs(record["training_neighbour_weight"][0] - 0.5) < 1e-12
          and record["string_in_graph"].tolist() == [True, True, True, True, False],
          f"{record['training_neighbours'].tolist()}")


def test_10_neighbourhood_uses_unique_identities() -> None:
    features = np.asarray([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]])
    keys = np.asarray(["A|g1", "B|g1", "A|g2"])
    reduced, identities = sp.unique_training(features, keys)
    check("10 a candidate measured twice supplies one neighbour, not two",
          reduced.shape[0] == 2 and identities.tolist() == ["g1", "g2"])


def test_11_d10_is_a_mean_over_the_ten_nearest() -> None:
    reference = np.arange(60, dtype=float)[:, None]
    record = sp.neighbourhood(np.asarray([[0.0]]), reference, k=10, sensitivity=(1, 50))
    check("11 d10 is the mean of the ten smallest distances",
          abs(record["d10"][0] - np.arange(10).mean()) < 1e-12
          and abs(record["d1"][0] - 0.0) < 1e-12,
          f"d10 {record['d10'][0]:.4f}")


# --- 12 to 14  stratification and matching ------------------------------------
def test_12_strata_are_equal_and_ascending() -> None:
    values = np.random.default_rng(12).random(1003)
    masks = nt.quantile_strata(values, ov.SUPPORT_STRATA, ov.STRATUM_NAMES)
    sizes = [int(mask.sum()) for mask in masks.values()]
    medians = [float(np.median(values[mask])) for mask in masks.values()]
    check("12 support strata are equal in size and ordered best first",
          sum(sizes) == values.size and max(sizes) - min(sizes) <= 1
          and medians == sorted(medians), f"{sizes} {['%.3f' % m for m in medians]}")


class _Table:
    def __init__(self, coverage, leverage, density, degree):
        self.coverage = np.asarray(coverage, dtype="<U12")
        self.leverage_percentile = np.asarray(leverage, dtype=float)
        self.d10_percentile = np.asarray(density, dtype=float)
        self.string_degree = np.asarray(degree, dtype=float)


def test_13_matching_is_one_to_one_and_respects_the_caliper() -> None:
    table = _Table(["BOTH"] * 4 + ["NEITHER"] * 2,
                   [0.10, 0.20, 0.11, 0.80, 0.30, 0.31],
                   [0.10, 0.20, 0.11, 0.80, 0.30, 0.31],
                   [1, 2, 3, 4, 5, 6])
    treated = np.asarray([True, True, False, False, True, False])
    control = np.asarray([False, False, True, True, False, True])
    matched = nt.match_support(table, treated, control, caliper=0.05)
    pairs = set(zip(matched["treated_rows"].tolist(), matched["control_rows"].tolist()))
    check("13 matching is one to one, inside the caliper, exact on coverage",
          matched["matched"] == 2 and pairs == {(0, 2), (4, 5)}
          and len(set(matched["treated_rows"].tolist())) == 2
          and len(set(matched["control_rows"].tolist())) == 2, f"{sorted(pairs)}")


def test_14_matching_never_crosses_a_coverage_pattern() -> None:
    table = _Table(["BOTH", "NEITHER"], [0.5, 0.5], [0.5, 0.5], [1, 1])
    matched = nt.match_support(table, np.asarray([True, False]), np.asarray([False, True]),
                               caliper=0.5)
    check("14 a candidate with no STRING is never matched to one that has it",
          matched.get("matched", 0) == 0)


# --- 15 to 18  the mixture and its cached cosines -----------------------------
def test_15_shares_nest() -> None:
    ids = np.asarray([f"G{i}" for i in range(4000)])
    values = mx.share_value(ids)
    sets = [set(np.flatnonzero(values < share).tolist()) for share in ov.MIXTURE_SHARES]
    check("15 whatever is unseen at a smaller share stays unseen at a larger one",
          sets[0] < sets[1] < sets[2]
          and all(abs(len(part) / ids.size - share) < 0.03
                  for part, share in zip(sets, ov.MIXTURE_SHARES)),
          f"{[len(part) / ids.size for part in sets]}")


def test_16_mixing_cosines_equals_mixing_effects() -> None:
    generator = np.random.default_rng(16)
    responses = generator.standard_normal((2, 6, 9))
    seen = generator.standard_normal((5, 9))
    masked = generator.standard_normal((5, 9))
    unseen = np.asarray([True, False, True, False, True])
    mixed_effect = np.where(unseen[:, None], masked, seen)
    direct = ev.effect_cosine(responses, mixed_effect, 0)
    assembled = np.where(unseen[None, :], ev.effect_cosine(responses, masked, 0),
                         ev.effect_cosine(responses, seen, 0))
    check("16 the cached cosine columns are the cosine of the mixed effect",
          np.allclose(direct, assembled, rtol=0, atol=0),
          f"max abs error {np.abs(direct - assembled).max():.3e}")


def test_17_entrant_value_credits_against_what_was_displaced() -> None:
    ids = np.asarray(["a", "b", "c", "d"])
    base = np.asarray([[4.0, 3.0, 2.0, 1.0]])
    mixed = np.asarray([[4.0, 1.0, 2.0, 3.0]])           # d replaces b in the top two
    utility = np.asarray([[0.9, 0.5, 0.1, 0.7]])
    record = mx.entrant_value(base, mixed, utility, ids, np.asarray([-1]), budget=2)
    check("17 an entrant is credited with its utility minus what it displaced",
          record["count"].tolist() == [0, 0, 0, 1]
          and abs(record["total"][3] - (0.7 - 0.5)) < 1e-12,
          f"{record['total'].tolist()}")


def test_18_composition_fractions_partition_the_budget() -> None:
    generator = np.random.default_rng(18)
    ids = np.asarray([f"c{i}" for i in range(120)])
    scores = generator.standard_normal((4, 120))
    utility = generator.standard_normal((4, 120))
    unseen = generator.random(120) < 0.4
    record = mx.composition(scores, utility, ids, np.full(4, -1), {"seen": ~unseen, "unseen": unseen})
    check("18 the seen and unseen fractions of a budget sum to one",
          all(abs(entry["fraction"]["seen"] + entry["fraction"]["unseen"] - 1.0) < 1e-12
              for entry in record.values()), f"{list(record)}")


# --- 19 to 22  the firewall ----------------------------------------------------
def test_19_no_module_here_opens_a_sealed_asset() -> None:
    """A mention is not a read.  Every phase record *declares* ``read_anchor_G_check``,
    so grepping for the name alone would fail on the declaration itself and would pass
    on a module that opened the raw cache under a variable.  What must never appear is
    a load or an open whose argument names a sealed asset."""
    directory = Path(__file__).resolve().parent
    sealed = ("responses_ab.npy", "x_goal_ab.npy", "G_check_eligible", "G_check.npy")
    opener = ("np.load", "open(", "memmap", "read_text", "read_bytes")
    offenders, declared = [], []
    for path in sorted(directory.glob("*.py")):
        if path.name == "test_protocol.py":
            continue
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if any(name in line for name in sealed) and any(call in line for call in opener):
                offenders.append(f"{path.name}:{number}")
            if "read_anchor_G_check" in line and "False" in line:
                declared.append(path.name)
    check("19 no module in this mission opens a sealed anchor asset, and each declares it",
          not offenders and len(declared) >= 3, f"{offenders} declared by {declared}")


def test_20_support_is_never_a_feature() -> None:
    directory = Path(__file__).resolve().parent
    offenders = []
    for path in sorted(directory.glob("*.py")):
        if path.name in ("support.py", "test_protocol.py"):
            continue
        text = path.read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or "pd.fit(" not in stripped:
                continue
            if any(word in stripped for word in ("leverage", "d10", "degree", "coverage")):
                offenders.append(f"{path.name}: {stripped}")
    check("20 no support diagnostic is ever passed into a predictor fit",
          not offenders, str(offenders))


def test_21_homogeneity_rejects_disagreeing_intercepts() -> None:
    class _Basis:
        def __init__(self, direction):
            self.direction = np.asarray(direction, dtype=float)

        def reconstruct(self, values):
            return np.asarray(values, dtype=float) @ np.eye(self.direction.size) * 0 + self.direction

    class _Path:
        def __init__(self, mean):
            self.target_mean = np.asarray(mean, dtype=float)

    class _Fitted:
        def __init__(self, direction):
            self.basis = _Basis(direction)
            self.path = _Path(np.zeros(3))
            self.penalty = 1e4

    same = mk.homogeneity([_Fitted([1.0, 0.0, 0.0]), _Fitted([1.0, 0.0, 0.0])])
    different = mk.homogeneity([_Fitted([1.0, 0.0, 0.0]), _Fitted([0.0, 1.0, 0.0])])
    check("21 the cross-fit gate passes identical intercepts and rejects orthogonal ones",
          same["passes"] and not different["passes"],
          f"{same['minimum_pairwise_cosine']:.3f} vs {different['minimum_pairwise_cosine']:.3f}")


def test_25_crossfit_keeps_the_seen_effect_outside_the_manipulation() -> None:
    """Zero-filling the untouched candidates is the defect this pins.

    The frozen fusion standardizes the effect row over the whole pool before the
    evaluation restricts it.  If the masked arm's row were mostly exact zeros while the
    seen arm's stayed dense, the two arms would enter the fusion at different effective
    weights and the contrast would be a fusion artifact, not an exposure result.
    """
    seen = np.arange(24, dtype=float).reshape(6, 4)
    effects = [np.full((6, 4), float(-1 - i)) for i in range(ov.MASK_FOLDS)]
    eligible = np.asarray([True, True, False, True, False, True])
    folds = np.asarray([0, 1, 2, 3, 4, 0])
    out = mk._crossfit(seen, effects, eligible, folds, offset=0)
    outside = ~eligible
    took_own = all(np.allclose(out[row], effects[folds[row]][row])
                   for row in np.flatnonzero(eligible))
    check("25 crossfit keeps the seen effect outside the eligible set",
          np.allclose(out[outside], seen[outside]) and took_own
          and not np.allclose(out[outside], 0.0))


def test_26_size_control_takes_the_next_fold() -> None:
    seen = np.zeros((5, 3))
    effects = [np.full((5, 3), float(i + 1)) for i in range(ov.MASK_FOLDS)]
    eligible = np.ones(5, dtype=bool)
    folds = np.arange(5)
    control = mk._crossfit(seen, effects, eligible, folds, offset=1)
    expected = np.stack([effects[(f + 1) % ov.MASK_FOLDS][row]
                         for row, f in enumerate(folds.tolist())])
    check("26 the size control is the next fold's predictor, wrapping at the last",
          np.allclose(control, expected))


def test_27_query_cluster_bootstrap_keeps_views_together() -> None:
    """Both views of one query must move together, or the interval is the frozen one."""
    half = 200
    difference = np.concatenate([np.full(half, 1.0), np.full(half, -1.0)])
    record = mk.query_cluster_bootstrap(difference, replicates=200, seed=1)
    low, high = record["median_ci95"]
    check("27 the query-identity interval draws both views of a query together",
          record["clusters"] == half and abs(low) < 1e-9 and abs(high) < 1e-9,
          f"clusters {record['clusters']} ci [{low:+.3f},{high:+.3f}]")


def test_28_frozen_bootstrap_would_not_keep_them_together() -> None:
    """The comparison that makes test 27 evidence rather than a tautology."""
    from gene_open_inverse.utility_gt_v2.validate import cluster_bootstrap
    half = 200
    difference = np.concatenate([np.full(half, 1.0), np.full(half, -1.0)])
    low, high = cluster_bootstrap(difference, replicates=200, seed=1)["median"]["ci95"]
    check("28 the frozen bootstrap does not, which is why it is reported beside",
          not (abs(low) < 1e-9 and abs(high) < 1e-9), f"frozen ci [{low:+.3f},{high:+.3f}]")


def test_29_mixture_fuses_over_the_full_pool() -> None:
    """Fusing after the restriction would run the effect branch at another weight."""
    generator = np.random.default_rng(29)
    base = generator.standard_normal((3, 40))
    cosine = generator.standard_normal((3, 40))
    columns = np.asarray([2, 5, 9, 14, 30])
    full_then_slice = ev.fuse(base, cosine)[:, columns]
    slice_then_fuse = ev.fuse(base[:, columns], cosine[:, columns])
    check("29 fusing before and after a restriction are different scores",
          not np.allclose(full_then_slice, slice_then_fuse)
          and "full context pool" in mx.run_context.__doc__,
          f"max abs difference {np.abs(full_then_slice - slice_then_fuse).max():.3f}")


def test_30_a_candidate_is_not_its_own_neighbour() -> None:
    """The static feature vector is a function of the symbol, so a measured candidate
    finds a bitwise identical copy of itself in the training atlas at distance zero.
    Leaving it in hands every atlas-measured candidate a free zero on exactly the axis
    the seen-against-unseen matching matches on."""
    reference = np.arange(60, dtype=float)[:, None]
    query = np.asarray([[7.0]])
    without = sp.neighbourhood(query, reference, k=10, sensitivity=(1, 50))
    with_self = sp.neighbourhood(query, reference, k=10, sensitivity=(1, 50),
                                 exclude=np.asarray([7]))
    check("30 a candidate in the training atlas is excluded from its own neighbourhood",
          abs(without["d1"][0]) < 1e-12 and abs(with_self["d1"][0] - 1.0) < 1e-12
          and with_self["d10"][0] > without["d10"][0],
          f"d1 {without['d1'][0]:.1f} -> {with_self['d1'][0]:.1f}, "
          f"d10 {without['d10'][0]:.2f} -> {with_self['d10'][0]:.2f}")


def test_31_self_columns_maps_only_measured_identities() -> None:
    columns = sp.self_columns(np.asarray(["b", "zz", "a"]), np.asarray(["a", "b", "c"]))
    check("31 only identities present in the training atlas are excluded",
          columns.tolist() == [1, -1, 0], f"{columns.tolist()}")


class _Fold:
    def __init__(self, direction, penalty):
        self.penalty = penalty
        self._direction = np.asarray(direction, dtype=float)

        class _B:
            def __init__(self, d):
                self.d = d

            def reconstruct(self, values):
                return np.repeat(self.d[None, :], np.asarray(values).shape[0], axis=0)

        class _P:
            def __init__(self, n):
                self.target_mean = np.zeros(n)

        self.basis = _B(self._direction)
        self.path = _P(3)


def test_32_homogeneity_fails_on_disagreeing_penalties() -> None:
    same = [_Fold([1.0, 0.0, 0.0], 1e4) for _ in range(3)]
    mixed = [_Fold([1.0, 0.0, 0.0], 1e4), _Fold([1.0, 0.0, 0.0], 1e3),
             _Fold([1.0, 0.0, 0.0], 1e4)]
    check("32 the gate fails when the mixed predictors shrank by different penalties",
          mk.homogeneity(same)["passes"] and not mk.homogeneity(mixed)["passes"],
          f"{mk.homogeneity(mixed)['selected_penalties']}")


def test_33_homogeneity_fails_on_disagreeing_shrinkage() -> None:
    """The defect this replaces: a direction-only gate passes two predictors whose
    candidate-specific content differs by a factor, which is what actually makes their
    per-candidate columns incomparable inside one score row."""
    folds = [_Fold([1.0, 0.0, 0.0], 1e4) for _ in range(2)]
    rows = np.arange(4)
    tight = np.repeat(np.asarray([[1.0, 0.01, 0.0]]), 4, axis=0)
    loose = np.repeat(np.asarray([[1.0, 0.50, 0.0]]), 4, axis=0)
    direction_only = mk.homogeneity(folds)
    with_scale = mk.homogeneity(folds, [tight, loose], rows)
    check("33 a direction-only gate passes disagreeing shrinkage and the real one does not",
          direction_only["passes"] and not with_scale["passes"]
          and with_scale["candidate_specific_share_ratio"] > ov.CROSSFIT_SCALE_TOLERANCE,
          f"ratio {with_scale['candidate_specific_share_ratio']:.1f}")


def test_22_contract_names_only_the_four_legal_verdicts() -> None:
    check("22 the verdict set is fixed in advance and holds four entries",
          len(ov.VERDICTS) == 4 and "OPEN_VOCABULARY_FAILURE_UNRESOLVED" in ov.VERDICTS)


def test_23_primary_floor_is_four_budgets_wide() -> None:
    from gene_open_inverse.model.decision_alignment_v1.contract import MAX_BUDGET
    check("23 the primary pool floor is four times the largest budget",
          ov.MIN_PRIMARY_POOL == 4 * MAX_BUDGET, f"{ov.MIN_PRIMARY_POOL} vs {MAX_BUDGET}")


def test_24_every_context_is_covered_by_the_masking_plan() -> None:
    covered = set(ov.MASK_PRIMARY_CONTEXTS) | set(ov.MASK_SECONDARY_CONTEXTS)
    check("24 the masking plan names every context exactly once",
          covered == set(fc.CONTEXTS)
          and not (set(ov.MASK_PRIMARY_CONTEXTS) & set(ov.MASK_SECONDARY_CONTEXTS)))


def main() -> int:
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            try:
                function()
            except Exception as error:                       # a crash is a failure, not a skip
                check(name, False, f"raised {type(error).__name__}: {error}")
    width = max(len(name) for name, _, _ in RESULTS)
    failed = 0
    for name, passed, detail in RESULTS:
        failed += not passed
        print(f"{'PASS' if passed else 'FAIL'}  {name:<{width}}  {detail}")
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
