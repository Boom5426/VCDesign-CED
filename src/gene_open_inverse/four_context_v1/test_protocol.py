"""Protocol tests for the four-context programme.

Each test targets a defect that would otherwise look like a valid result: a split
that is not a refinement, a weighted solver that quietly disagrees with the frozen
one, a self-exclusion that deletes the wrong candidate, a stratum whose two halves
overlap, a fusion that reorders a row it should not.
"""
from __future__ import annotations

import numpy as np
import pytest

from gene_open_inverse.candidate_effect_distillation_v1 import predictor as pr
from gene_open_inverse.four_context_v1 import contract as fc
from gene_open_inverse.four_context_v1 import evaluate as ev
from gene_open_inverse.four_context_v1 import genes as gn
from gene_open_inverse.four_context_v1 import overlap as ov
from gene_open_inverse.four_context_v1 import predictors as pd
from gene_open_inverse.four_context_v1.build import quarter_of, view_side
from gene_open_inverse.four_context_v1.contexts import Context, _restrict, common_gene_axis


NAMES = [f"lane_{i}" for i in range(200)]


def test_01_quarter_refines_the_side_and_is_deterministic():
    for name in NAMES:
        assert quarter_of(name) // 2 == view_side(name)
        assert quarter_of(name) == quarter_of(name)
    sides = np.asarray([view_side(n) for n in NAMES])
    assert 0.3 < sides.mean() < 0.7, "the A/B hash is lopsided on a 200-name axis"
    quarters = np.asarray([quarter_of(n) for n in NAMES])
    assert set(quarters.tolist()) == {0, 1, 2, 3}


def test_02_only_the_anchor_restricts_which_identities_may_train_or_be_queried():
    # Outside the anchor the held context supplies no training row in its own fold,
    # so splitting it would only discard queries.  If a future edit reintroduced a
    # split for the other three, this asserts it was a deliberate contract change.
    assert fc.TRAIN_RULE["K562"] != fc.QUERY_RULE["K562"]
    for name in ("RPE1", "HepG2", "Jurkat"):
        assert fc.TRAIN_RULE[name] == "every usable identity"
        assert "every usable and eligible identity" == fc.QUERY_RULE[name]


def test_03_weighted_ridge_reduces_to_the_frozen_path_at_equal_weights():
    generator = np.random.default_rng(7)
    features = generator.standard_normal((60, 9))
    targets = generator.standard_normal((60, 4))
    frozen = pr.RidgePath(features, targets)
    weighted = pd.WeightedRidgePath(features, targets)
    probe = generator.standard_normal((11, 9))
    for penalty in (1e-2, 1.0, 1e3):
        a = frozen.predict(probe, penalty)
        b = weighted.predict(probe, penalty)
        # The two solve the same normal equations; they differ only by the row scaling
        # 1/sqrt(n), which cancels on both sides, so the residual is float64 round-off
        # of the fitted magnitude rather than a modelling difference.
        assert np.abs(a - b).max() < 1e-8 * max(np.abs(a).max(), 1.0)


def test_04_candidate_balanced_weights_give_each_identity_one_unit():
    keys = np.asarray(["K562|A", "RPE1|A", "HepG2|A", "K562|B", "RPE1|C"])
    weights = pd.candidate_balanced_weights(keys)
    assert weights.sum() == pytest.approx(3.0)
    assert weights[:3] == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert weights[3] == pytest.approx(1.0)


def test_05_self_exclusion_never_deletes_the_last_candidate_when_the_query_is_absent():
    # The frozen stable_order reads excluded_index=-1 as the final candidate, so a
    # stratum that does not contain its query must pass None.  If that regressed, the
    # highest-utility candidate below would silently vanish from every ordering.
    ids = np.asarray([f"c{i:03d}" for i in range(60)])   # must exceed the largest budget
    scores = np.zeros((1, ids.size))
    scores[0, -1] = 10.0
    utility = np.zeros((1, ids.size))
    utility[0, -1] = 1.0
    absent = ev.stratum_metrics(scores, utility, ids, np.asarray([-1]))
    assert absent["Mean@10"][0] > 0.0, "the last candidate was deleted by a -1 exclusion"
    present = ev.stratum_metrics(scores, utility, ids, np.asarray([ids.size - 1]))
    assert present["Mean@10"][0] == pytest.approx(0.0)


def test_06_oracle_beats_a_constant_ranker_on_the_same_stratum():
    generator = np.random.default_rng(11)
    queries, candidates = 6, 80
    utility = generator.standard_normal((queries, candidates))
    ids = np.asarray([f"c{i}" for i in range(candidates)])
    self_positions = np.full(queries, -1)
    oracle = ev.stratum_metrics(ev.oracle_scores(utility), utility, ids, self_positions)
    flat = ev.stratum_metrics(np.zeros_like(utility), utility, ids, self_positions)
    assert oracle["MBRU"].mean() > flat["MBRU"].mean()
    assert oracle["MBRU"].mean() == pytest.approx(1.0, abs=1e-9)


def test_07_strata_are_a_partition_and_common4_is_a_subset_of_seen_or_unseen():
    candidates = np.asarray(["A", "B", "C", "D"])
    masks = ov.strata_masks(candidates, seen=np.asarray(["A", "B"]), common4=np.asarray(["B", "D"]))
    assert np.array_equal(masks["SEEN_CANDIDATE"] | masks["UNSEEN_CANDIDATE"], masks["ALL"])
    assert not (masks["SEEN_CANDIDATE"] & masks["UNSEEN_CANDIDATE"]).any()
    assert masks["COMMON4"].tolist() == [False, True, False, True]


def test_08_row_z_is_rank_preserving_within_a_row():
    generator = np.random.default_rng(3)
    matrix = generator.standard_normal((5, 40)) * generator.uniform(0.1, 9.0, size=(5, 1))
    for row in range(matrix.shape[0]):
        assert np.array_equal(np.argsort(matrix[row]), np.argsort(ev.row_z(matrix)[row]))


def test_09_effect_cosine_gives_an_absent_candidate_exactly_the_neutral_score():
    responses = np.zeros((2, 3, 5))
    responses[0] = np.asarray([[1.0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0], [0, 0, 1.0, 0, 0]])
    effect = np.asarray([[1.0, 0, 0, 0, 0], [0.0, 0, 0, 0, 0], [0, 0, 1.0, 0, 0]])
    cosine = ev.effect_cosine(responses, effect, view=0)
    assert cosine[0, 0] == pytest.approx(1.0)
    assert np.array_equal(cosine[:, 1], np.zeros(3))


def test_10_energy_retained_is_exact_on_a_constructed_case():
    responses = np.zeros((2, 2, 4))
    responses[:, 0] = [3.0, 4.0, 0.0, 0.0]
    responses[:, 1] = [0.0, 0.0, 6.0, 8.0]
    usable = np.ones(2, dtype=bool)
    record = gn.energy_retained(responses, usable, np.asarray([0, 1]))
    # kept 2*(9+16) = 50 of total 2*(25+100) = 250
    assert record["pooled"] == pytest.approx(50.0 / 250.0)
    assert record["per_identity_median"] == pytest.approx(0.5)


def test_11_gene_axis_restriction_reorders_columns_and_refuses_unmeasured_genes():
    context = Context(name="X", ids=np.asarray(["a"]), genes=np.asarray(["g1", "g2", "g3"]),
                      responses=np.arange(6, dtype=np.float32).reshape(2, 1, 3),
                      sources=np.zeros((2, 1, 3), dtype=np.float32),
                      usable=np.ones(1, dtype=bool), eligible=np.ones(1, dtype=bool),
                      train_ok=np.ones(1, dtype=bool), query_ok=np.ones(1, dtype=bool))
    reduced = _restrict(context, np.asarray(["g3", "g1"]))
    assert reduced.responses[0, 0].tolist() == [2.0, 0.0]
    with pytest.raises(KeyError):
        _restrict(context, np.asarray(["g1", "g9"]))


def test_12_common_gene_axis_is_the_intersection_and_is_sorted():
    def make(genes):
        return Context(name="c", ids=np.asarray(["a"]), genes=np.asarray(genes),
                       responses=np.zeros((2, 1, len(genes)), dtype=np.float32),
                       sources=np.zeros((2, 1, len(genes)), dtype=np.float32),
                       usable=np.ones(1, dtype=bool), eligible=np.ones(1, dtype=bool),
                       train_ok=np.ones(1, dtype=bool), query_ok=np.ones(1, dtype=bool))
    contexts = {"a": make(["g3", "g1", "g2"]), "b": make(["g2", "g3"]), "c": make(["g3", "g2", "g7"])}
    axis = common_gene_axis(contexts)
    assert axis.tolist() == ["g2", "g3"]


def test_13_fit_refuses_a_pool_too_small_for_the_declared_rank():
    generator = np.random.default_rng(5)
    features = generator.standard_normal((20, 6))
    responses = generator.standard_normal((20, 30))
    present = np.ones(20, dtype=bool)
    keys = np.asarray([f"K562|g{i}" for i in range(20)])
    with pytest.raises(ValueError):
        pd.fit(features, responses, present, keys, ("K562",), rank=64)


def test_14_a_fitted_predictor_recovers_a_linear_map_it_was_shown():
    generator = np.random.default_rng(13)
    rows, width, genes, rank = 400, 12, 60, 8
    truth = generator.standard_normal((width, genes))
    features = generator.standard_normal((rows, width))
    responses = features @ truth + 0.01 * generator.standard_normal((rows, genes))
    keys = np.asarray([f"K562|g{i}" for i in range(rows)])
    model = pd.fit(features, responses, np.ones(rows, dtype=bool), keys, ("K562",), rank=rank)
    predicted = model.effect(features, np.ones(rows, dtype=bool))
    cosine = np.einsum("ij,ij->i", ev.unit(predicted), ev.unit(responses))
    assert float(np.median(cosine)) > 0.9
    assert model.rows == rows and model.identities == rows


def test_15_an_all_missing_candidate_gets_exactly_zero_effect():
    generator = np.random.default_rng(17)
    rows = 300
    features = generator.standard_normal((rows, 6))
    responses = generator.standard_normal((rows, 40))
    keys = np.asarray([f"K562|g{i}" for i in range(rows)])
    model = pd.fit(features, responses, np.ones(rows, dtype=bool), keys, ("K562",), rank=8)
    present = np.ones(rows, dtype=bool)
    present[:5] = False
    effect = model.effect(features, present)
    assert np.abs(effect[:5]).max() == 0.0
    assert np.abs(effect[5:]).max() > 0.0


def test_16_v2_cosine_matches_a_brute_force_evaluation():
    # The closed-form expansion is the only reason a query-dependent effect is
    # affordable at all.  If it drifted from the definition every V2 number would be
    # wrong in a way no downstream check could see.
    from gene_open_inverse.four_context_v1 import v2
    generator = np.random.default_rng(23)
    queries, candidates, genes, rank = 7, 11, 19, 4
    effect = generator.standard_normal((candidates, genes))
    u = generator.standard_normal((candidates, rank))
    h = generator.standard_normal((queries, rank))
    g = generator.standard_normal((rank, genes))
    directions = generator.standard_normal((queries, genes))
    fast = v2.v2_cosine(effect, u, h, directions, g)
    slow = np.empty((queries, candidates))
    for q in range(queries):
        unit_d = directions[q] / np.linalg.norm(directions[q])
        for c in range(candidates):
            vector = effect[c] + sum(u[c, k] * h[q, k] * g[k] for k in range(rank))
            slow[q, c] = float(unit_d @ vector / np.linalg.norm(vector))
    assert np.abs(fast - slow).max() < 1e-9


def test_17_v2_degenerates_to_v1_as_the_penalty_grows():
    from gene_open_inverse.four_context_v1 import v2
    generator = np.random.default_rng(29)
    rows, width, rank = 300, 24, 4
    z = generator.standard_normal((rows, width))
    h = generator.standard_normal((rows, 6))
    target = generator.standard_normal((rows, width))
    with pytest.raises(ValueError):
        v2.fit_residual(z, h, target, 9, 1e1)   # rank above the context dimension
    small = v2.fit_residual(z, h, target, rank, 1e1)
    large = v2.fit_residual(z, h, target, rank, 1e12)
    assert np.abs(large.delta(z, h)).max() < 1e-6 * max(np.abs(small.delta(z, h)).max(), 1e-12)


def test_18_the_context_basis_recovers_a_planted_low_dimensional_structure():
    from gene_open_inverse.four_context_v1 import v2
    generator = np.random.default_rng(31)
    loadings = generator.standard_normal((400, 3))
    directions = generator.standard_normal((3, 50))
    sources = loadings @ directions + 1e-6 * generator.standard_normal((400, 50))
    basis = v2.context_basis(sources, dimension=3)
    assert basis["explained"] > 0.999
    projected = v2.project_context(sources, basis)
    # The projection is a rotation of the planted loadings, so a linear map recovers them.
    fitted = np.linalg.lstsq(projected, loadings - loadings.mean(axis=0), rcond=None)[0]
    residual = (loadings - loadings.mean(axis=0)) - projected @ fitted
    assert np.abs(residual).max() < 1e-4


def test_19_the_context_prior_control_modulates_every_candidate_identically():
    from gene_open_inverse.four_context_v1 import v2
    generator = np.random.default_rng(37)
    rows, width, rank = 200, 12, 4
    z = generator.standard_normal((rows, width))
    h = generator.standard_normal((rows, 5))
    target = generator.standard_normal((rows, width))
    prior = v2.fit_residual(z, h, target, rank, 1e2, candidate_blind=True)
    term = prior.candidate_term(z)
    assert np.abs(term - term[0]).max() < 1e-9
    normal = v2.fit_residual(z, h, target, rank, 1e2)
    assert np.abs(normal.candidate_term(z) - normal.candidate_term(z)[0]).max() > 1e-6


def test_20_the_blind_control_keeps_the_marginals_and_destroys_the_context_signal():
    from gene_open_inverse.four_context_v1 import v2
    generator = np.random.default_rng(41)
    h = np.concatenate([generator.normal(-3.0, 1.0, size=(400, 4)),
                        generator.normal(+3.0, 1.0, size=(400, 4))])
    blind = v2._blind_h(h, seed=5)
    assert np.abs(blind.mean(axis=0) - h.mean(axis=0)).max() < 0.3
    assert np.abs(blind.std(axis=0) - h.std(axis=0)).max() < 0.4
    label = np.concatenate([np.zeros(400), np.ones(400)])
    true_gap = abs(h[label == 0, 0].mean() - h[label == 1, 0].mean())
    blind_gap = abs(blind[label == 0, 0].mean() - blind[label == 1, 0].mean())
    assert blind_gap < 0.2 * true_gap


def test_21_stratified_evaluation_is_not_confounded_by_a_moving_candidate_pool():
    # Two strata of different sizes must not be compared on absolute MBRU.  The
    # evaluator therefore renormalises inside each stratum; this asserts it does.
    generator = np.random.default_rng(43)
    queries, candidates = 4, 300
    utility = generator.standard_normal((queries, candidates))
    ids = np.asarray([f"c{i:04d}" for i in range(candidates)])
    self_positions = np.full(queries, -1)
    whole = ev.stratum_metrics(ev.oracle_scores(utility), utility, ids, self_positions)
    half = np.zeros(candidates, dtype=bool)
    half[:150] = True
    part = ev.stratum_metrics(ev.oracle_scores(utility[:, half]), utility[:, half],
                              ids[half], self_positions)
    assert whole["MBRU"].mean() == pytest.approx(1.0, abs=1e-9)
    assert part["MBRU"].mean() == pytest.approx(1.0, abs=1e-9)
    assert part["Mean@10"].mean() < whole["Mean@10"].mean()


# --- the frozen response-asset verifier contract ---------------------------

def test_22_storage_contract_accepts_a_one_ulp_float32_difference():
    from gene_open_inverse.four_context_v1 import storage as st
    generator = np.random.default_rng(101)
    expected = generator.uniform(1.0, 8.0, size=(50, 400))
    stored = np.nextafter(expected.astype(np.float32), np.float32(np.inf))
    record = st.compare(stored, expected, "one_ulp")
    assert record["passed"], record
    assert record["max_ulp_distance_where_ulp_binds"] <= 1


def test_23_storage_contract_rejects_a_realistic_magnitude_error():
    # This is the test that makes the tolerance a verification rather than a formality.
    from gene_open_inverse.four_context_v1 import storage as st
    generator = np.random.default_rng(103)
    expected = generator.uniform(-8.0, 8.0, size=(50, 400))
    stored = st.corrupt_for_witness(expected.astype(np.float32), magnitude=1e-3)
    record = st.compare(stored, expected, "corrupted")
    assert not record["passed"]
    assert record["violations"] == 1


def test_24_storage_contract_survives_subtractive_cancellation_near_zero():
    # A response is a difference of two order-one quantities, so where they cancel the
    # result is tiny while its absolute error is not.  A purely relative or purely ULP
    # bound would fail here even though nothing is wrong; the max(.,1) floor is why.
    from gene_open_inverse.four_context_v1 import storage as st
    expected = np.asarray([[1e-12, -3e-11, 5e-13, 0.0]], dtype=np.float64)
    stored = np.asarray([[2e-12, -1e-11, -4e-13, 0.0]], dtype=np.float32)
    assert st.compare(stored, expected, "cancellation")["passed"]
    assert int(st.ulp_distance(stored, expected).max()) > 1000


def test_25_ulp_distance_is_monotonic_across_zero_and_symmetric():
    from gene_open_inverse.four_context_v1 import storage as st
    def gap(left, right):
        return int(st.ulp_distance(np.asarray([left], dtype=np.float32),
                                   np.asarray([right], dtype=np.float32))[0])

    assert gap(0.0, -0.0) == 0
    assert gap(1.0, np.nextafter(np.float32(1.0), np.float32(np.inf))) == 1
    assert gap(-1.0, np.nextafter(np.float32(-1.0), np.float32(0.0))) == 1
    assert gap(-1e-45, 1e-45) == 2


def test_26_storage_contract_refuses_a_non_finite_value_instead_of_ignoring_it():
    from gene_open_inverse.four_context_v1 import storage as st
    expected = np.asarray([[1.0, np.nan]], dtype=np.float64)
    stored = np.asarray([[1.0, np.nan]], dtype=np.float32)
    assert not st.compare(stored, expected, "nan")["passed"]


def test_27_the_bound_is_the_declared_expression_and_not_a_fitted_number():
    from gene_open_inverse.four_context_v1 import storage as st
    assert st.TOLERANCE_ULPS == 2.0
    assert st.TOLERANCE_EXPRESSION == "abs(stored - expected32) <= 2 * eps32 * max(abs(expected32), 1)"
    # exactly on the bound passes, just above it fails, at a magnitude where the
    # relative half of the expression is the binding one
    value = np.float64(100.0)
    bound = st.TOLERANCE_ULPS * st.EPS32 * value
    assert st.compare(np.asarray([np.float32(value + bound)]), np.asarray([value]), "edge")["passed"]
    assert not st.compare(np.asarray([np.float32(value + 20 * bound)]),
                          np.asarray([value]), "over")["passed"]


# --- the dual-sgRNA identity gate ------------------------------------------

UNIVERSE = frozenset({"AARS1", "BRCA1", "TP53", "MYC", "EGFR"})


def test_28_same_gene_dual_guides_collapse_to_one_intervention():
    from gene_open_inverse.four_context_v1 import dualguide as dgm
    for label in ("AARS1_+_AARS1_+", "AARS1_+_AARS1_-", "TP53|TP53"):
        kind, identity = dgm.resolve_label(label, UNIVERSE)
        assert (kind, identity) == (dgm.SINGLE_GENE, label.split("_")[0].split("|")[0])


def test_29_gene_plus_non_targeting_is_a_single_gene_intervention():
    from gene_open_inverse.four_context_v1 import dualguide as dgm
    for label in ("MYC_+_non-targeting_00123", "non-targeting_7_+_MYC_-"):
        kind, identity = dgm.resolve_label(label, UNIVERSE)
        assert (kind, identity) == (dgm.SINGLE_GENE, "MYC")


def test_30_cross_gene_pairs_are_never_relabelled_as_one_of_their_genes():
    # This is the defect the gate exists for.  Calling "BRCA1 + TP53" an intervention
    # on BRCA1 would put TP53's response into BRCA1's distillation target and every
    # downstream number would inherit it.
    from gene_open_inverse.four_context_v1 import dualguide as dgm
    kind, identity = dgm.resolve_label("BRCA1_+_TP53_+", UNIVERSE)
    assert kind == dgm.CROSS_GENE
    assert identity == ""


def test_31_both_guides_non_targeting_is_a_control_and_unknown_symbols_are_unparsed():
    from gene_open_inverse.four_context_v1 import dualguide as dgm
    assert dgm.resolve_label("non-targeting_1_+_non-targeting_2", UNIVERSE)[0] == dgm.NON_TARGETING
    assert dgm.resolve_label("ZZZ9_+_ZZZ9_+", UNIVERSE)[0] == dgm.UNPARSED
    assert dgm.resolve_label("AARS1_+_ZZZ9_+", UNIVERSE)[0] == dgm.UNPARSED


def test_32_resolution_summary_counts_every_cell_exactly_once():
    from gene_open_inverse.four_context_v1 import dualguide as dgm
    labels = np.asarray(["AARS1_+_AARS1_+"] * 5 + ["MYC_+_non-targeting_1"] * 3
                        + ["BRCA1_+_TP53_+"] * 4 + ["non-targeting_1_+_non-targeting_2"] * 6
                        + ["ZZZ9_+_ZZZ9_+"] * 2)
    resolution = dgm.resolve(labels, UNIVERSE)
    summary = dgm.summarise(resolution, labels)
    assert sum(summary["cell_count"].values()) == labels.size
    assert summary["cell_count"][dgm.SINGLE_GENE] == 8
    assert summary["cell_count"][dgm.CROSS_GENE] == 4
    assert summary["cell_count"][dgm.NON_TARGETING] == 6
    assert summary["cell_count"][dgm.UNPARSED] == 2
    assert summary["single_gene_identities"] == 2


def test_33_pair_shape_separates_the_three_protocol_cases():
    from gene_open_inverse.four_context_v1 import dualguide as dgm
    labels = np.asarray(["AARS1_+_AARS1_+", "MYC_+_non-targeting_1", "BRCA1_+_TP53_+",
                         "non-targeting_1_+_non-targeting_2", "ZZZ9_+_ZZZ9_+"])
    shape = dgm.pair_shape(dgm.resolve(labels, UNIVERSE), UNIVERSE)
    counts = shape["label_counts"]
    assert counts["same_gene_pair"] == 1
    assert counts["gene_plus_ntc"] == 1
    assert counts["geneA_plus_geneB"] == 1
    assert counts["ntc_plus_ntc"] == 1
    assert counts["unparsed"] == 1


def test_34_a_context_with_too_few_single_gene_identities_is_downgraded():
    from gene_open_inverse.four_context_v1 import dualguide as dgm
    labels = np.asarray(["AARS1_+_AARS1_+", "MYC_+_MYC_+", "non-targeting_1_+_non-targeting_2"])
    resolution = dgm.resolve(labels, UNIVERSE)
    verdict = dgm.case_verdict(dgm.summarise(resolution, labels),
                               dgm.pair_shape(resolution, UNIVERSE), fc.MIN_USABLE_IDENTITIES)
    assert verdict["case"] == "A_SAME_GENE_DUAL_GUIDES"
    assert verdict["single_gene_unit_constructible"] is False
    assert verdict["single_gene_identities_after_exclusion"] == 2


def test_35_a_cross_gene_dominated_context_is_classified_as_case_C():
    from gene_open_inverse.four_context_v1 import dualguide as dgm
    labels = np.asarray([f"AARS1_+_TP53_+", "BRCA1_+_MYC_+", "EGFR_+_TP53_+",
                         "AARS1_+_AARS1_+"])
    resolution = dgm.resolve(labels, UNIVERSE)
    verdict = dgm.case_verdict(dgm.summarise(resolution, labels),
                               dgm.pair_shape(resolution, UNIVERSE), 1)
    assert verdict["case"] == "C_CROSS_GENE_PAIRS"
    assert verdict["cross_gene_labels_excluded"] == 3
    assert verdict["single_gene_identities_after_exclusion"] == 1


# --- the G_check seal -------------------------------------------------------

def test_36_the_anchor_loader_drops_every_sealed_row(tmp_path):
    from types import SimpleNamespace

    from gene_open_inverse.four_context_v1 import contexts as ctx

    ids = np.asarray([f"G{i:03d}" for i in range(12)])
    cache = np.arange(2 * 12 * 4, dtype=np.float32).reshape(2, 12, 4)
    role_indices = {"G_fit": np.arange(0, 6), "G_select": np.arange(6, 9),
                    "G_check": np.arange(9, 12)}
    base = SimpleNamespace(cache_ids=ids, cache=cache, role_indices=role_indices)
    assets = SimpleNamespace(base=base)

    np.save(tmp_path / "sources.npy", np.zeros_like(cache))
    np.save(tmp_path / "genes.npy", np.asarray(["g0", "g1", "g2", "g3"]))
    np.save(tmp_path / "G_fit_eligible.npy", np.ones(6, dtype=bool))
    np.save(tmp_path / "G_select_eligible.npy", np.ones(3, dtype=bool))

    context = ctx.load_anchor(assets, tmp_path / "sources.npy", tmp_path / "genes.npy", tmp_path)
    assert context.ids.size == 9
    sealed = set(ids[role_indices["G_check"]].tolist())
    assert not (set(context.ids.tolist()) & sealed)
    assert context.responses.shape[1] == 9
    # training may only use G_fit, queries only G_select
    assert set(context.ids[context.train_ok].tolist()) == set(ids[role_indices["G_fit"]].tolist())
    assert set(context.ids[context.query_ok].tolist()) == set(ids[role_indices["G_select"]].tolist())
    assert not (context.train_ok & context.query_ok).any()


def test_37_no_module_reads_the_raw_anchor_response_cache_directly():
    # The sealed split lives inside responses_ab.npy alongside the readable ones, so a
    # module that opens that file by path has bypassed the loader that drops G_check.
    # An aggregate over the sealed rows is still a read of them.
    import pathlib

    package = pathlib.Path(__file__).parent
    offenders = []
    for path in sorted(package.glob("*.py")):
        if path.name in ("test_protocol.py",):
            continue
        text = path.read_text()
        if "responses_ab.npy" in text or "eligible_ids.npy" in text:
            offenders.append(path.name)
    assert offenders == [], f"these modules reference the raw anchor cache: {offenders}"


def test_38_the_corruption_helper_works_on_a_non_contiguous_block():
    # The regression test for the round-four witness failure.  The verifier hands this
    # helper a fancy-indexed slice, which is not C-contiguous, and the earlier
    # implementation silently returned it unchanged.
    from gene_open_inverse.four_context_v1 import storage as st
    generator = np.random.default_rng(107)
    full = generator.standard_normal((2, 40, 25)).astype(np.float32)
    block = full[:, np.asarray([3, 9, 17, 22])]
    assert not block.flags["C_CONTIGUOUS"], "the fixture no longer reproduces the condition"
    corrupted = st.corrupt_for_witness(block, magnitude=1e-3)
    difference = np.abs(corrupted.astype(np.float64) - block.astype(np.float64))
    assert int((difference > 0).sum()) == 1
    assert float(difference.max()) == pytest.approx(1e-3, rel=1e-3)
    assert not st.compare(corrupted, block.astype(np.float64), "witness")["passed"]


def test_39_the_corruption_helper_refuses_to_be_a_silent_no_op():
    from gene_open_inverse.four_context_v1 import storage as st
    # A magnitude below the float32 spacing of the chosen element cannot change it, and
    # the helper must say so rather than return an uncorrupted array.
    huge = np.full((4, 4), 1e30, dtype=np.float32)
    with pytest.raises(RuntimeError):
        st.corrupt_for_witness(huge, magnitude=1e-3)


def test_40_the_real_gse264667_label_format_parses_to_one_gene():
    # The regression test for the first HepG2 audit.  The pair separator is "|" and
    # each half reads GENE_strand_coordinate, so "_+_" occurs *inside* a guide name.
    # With "_+_" tried first the label shredded and the whole dataset came back
    # unparsed, which looked like a disqualified context and was a parser fault.
    from gene_open_inverse.four_context_v1 import dualguide as dgm
    universe = frozenset({"AAAS", "AAMP", "AAR2", "AARS2", "TP53", "TP53BP1"})
    cases = {
        "AAAS_-_53715438.23-P1P2|AAAS_+_53715355.23-P1P2": "AAAS",
        "AAMP_+_219134851.23-P1P2|AAMP_+_219134841.23-P1P2": "AAMP",
        "AAR2_-_34824434.23-P1P2|AAR2_+_34824488.23-P1P2": "AAR2",
        "AARS2_+_44281027.23-P1P2|AARS2_+_44281044.23-P1P2": "AARS2",
    }
    for label, expected in cases.items():
        assert dgm.resolve_label(label, universe) == (dgm.SINGLE_GENE, expected), label
    assert dgm.resolve_label("non-targeting_00039|non-targeting_01771",
                             universe)[0] == dgm.NON_TARGETING
    # a genuine cross-gene pair in the same format is still refused a single identity
    assert dgm.resolve_label("AAAS_-_1.23-P1P2|TP53_+_2.23-P1P2", universe)[0] == dgm.CROSS_GENE


def test_41_the_prefix_reader_prefers_the_longest_known_symbol():
    # TP53BP1 must not be read as TP53 just because TP53 is also a known symbol.
    from gene_open_inverse.four_context_v1 import dualguide as dgm
    universe = frozenset({"TP53", "TP53BP1"})
    label = "TP53BP1_+_1.23-P1P2|TP53BP1_-_2.23-P1P2"
    assert dgm.resolve_label(label, universe) == (dgm.SINGLE_GENE, "TP53BP1")


def test_42_a_separator_that_shreds_the_label_is_rejected_not_accepted():
    from gene_open_inverse.four_context_v1 import dualguide as dgm
    universe = frozenset({"AAAS"})
    label = "AAAS_-_53715438.23-P1P2|AAAS_+_53715355.23-P1P2"
    # "_+_" is present in the label but splitting on it leaves a part that is not a
    # gene, so it must lose to "|".
    assert "_+_" in label
    assert dgm.split_pair(label, universe) == (
        "AAAS_-_53715438.23-P1P2", "AAAS_+_53715355.23-P1P2")


def test_43_an_unknown_prefix_stays_unparsed_rather_than_becoming_a_gene():
    from gene_open_inverse.four_context_v1 import dualguide as dgm
    universe = frozenset({"AAAS"})
    assert dgm.resolve_label("ZZZ9_+_1.23-P1P2|ZZZ9_-_2.23-P1P2", universe)[0] == dgm.UNPARSED


def test_44_unparsed_labels_are_checked_for_hidden_cross_gene_pairs():
    # "No cross-gene pairs" must not be a claim about only the labels that parsed.
    # An unresolved symbol is exactly where a combination could hide.
    from gene_open_inverse.four_context_v1 import dualguide as dgm
    universe = frozenset({"AAAS"})
    labels = np.asarray([
        "AAAS_-_1.23-P1P2|AAAS_+_2.23-P1P2",     # parses
        "ASUN_-_1.23-P1P2|ASUN_+_2.23-P1P2",     # unparsed alias, same gene
        "ZZZ1_-_1.23-P1P2|ZZZ2_+_2.23-P1P2",     # unparsed, and the halves differ
    ])
    resolution = dgm.resolve(labels, universe)
    unparsed = dgm.unparsed_pair_shape(resolution)
    assert unparsed["unparsed_labels"] == 2
    assert unparsed["identical_leading_token"] == 1
    assert unparsed["differing_leading_token"] == 1
    verdict = dgm.case_verdict(dgm.summarise(resolution, labels),
                               dgm.pair_shape(resolution, universe), 1, unparsed)
    assert verdict["labels_that_could_be_cross_gene"] == 1
    assert verdict["no_cross_gene_pair_anywhere"] is False


def test_45_no_cross_gene_anywhere_requires_the_unparsed_set_to_be_clean_too():
    from gene_open_inverse.four_context_v1 import dualguide as dgm
    universe = frozenset({"AAAS"})
    labels = np.asarray(["AAAS_-_1.23-P1P2|AAAS_+_2.23-P1P2",
                         "ASUN_-_1.23-P1P2|ASUN_+_2.23-P1P2"])
    resolution = dgm.resolve(labels, universe)
    unparsed = dgm.unparsed_pair_shape(resolution)
    verdict = dgm.case_verdict(dgm.summarise(resolution, labels),
                               dgm.pair_shape(resolution, universe), 1, unparsed)
    assert verdict["no_cross_gene_pair_anywhere"] is True


# --- the V2 identifiability positive control --------------------------------

def test_46_planted_residual_hits_the_requested_energy_share_exactly():
    # gamma is solved, not searched.  If this drifted, the identifiability curve would
    # be a curve over an unknown quantity.
    from gene_open_inverse.four_context_v1 import identifiability as idt
    generator = np.random.default_rng(51)
    candidates, genes, dim = 120, 90, fc.CONTEXT_DIM
    features = generator.standard_normal((candidates, 16))
    h = {name: generator.standard_normal((2, candidates, dim))
         for name in ("A", "B", "C", "H")}
    present = {name: np.ones(candidates, dtype=bool) for name in h}
    for share in (0.0, 0.10, 0.25, 0.50):
        truth = idt.plant(features, h, present, genes, share)
        assert truth.achieved_share == pytest.approx(share, abs=1e-9), share


def test_47_the_planted_residual_is_centred_across_contexts():
    # A component shared by every context is not a residual: V1 would absorb it and the
    # control would be measuring the wrong thing.
    from gene_open_inverse.four_context_v1 import identifiability as idt
    generator = np.random.default_rng(53)
    candidates, genes = 80, 60
    features = generator.standard_normal((candidates, 12))
    names = ("A", "B", "C", "H")
    h = {name: generator.standard_normal((2, candidates, fc.CONTEXT_DIM)) for name in names}
    present = {name: np.ones(candidates, dtype=bool) for name in names}
    truth = idt.plant(features, h, present, genes, 0.25)
    stacked = np.stack([truth.residual_z[name] for name in names])
    assert np.abs(stacked.mean(axis=0)).max() < 1e-8


def test_48_shared_and_residual_parts_are_energy_orthogonal():
    # This is what makes the share solvable in closed form rather than by search.
    from gene_open_inverse.four_context_v1 import identifiability as idt
    generator = np.random.default_rng(57)
    candidates, genes = 90, 70
    features = generator.standard_normal((candidates, 10))
    names = ("A", "B", "C")
    h = {name: generator.standard_normal((2, candidates, fc.CONTEXT_DIM)) for name in names}
    present = {name: np.ones(candidates, dtype=bool) for name in names}
    truth = idt.plant(features, h, present, genes, 0.5)
    shared = truth.shared_z @ truth.basis
    cross = sum(float((shared[None] * (truth.residual_z[name] @ truth.basis)).sum())
                for name in names)
    scale = float(np.abs(shared).sum())
    assert abs(cross) < 1e-6 * max(scale, 1.0)


def test_49_a_zero_share_plants_nothing_at_all():
    from gene_open_inverse.four_context_v1 import identifiability as idt
    generator = np.random.default_rng(59)
    features = generator.standard_normal((60, 8))
    names = ("A", "B", "C")
    h = {name: generator.standard_normal((2, 60, fc.CONTEXT_DIM)) for name in names}
    present = {name: np.ones(60, dtype=bool) for name in names}
    truth = idt.plant(features, h, present, 50, 0.0)
    assert truth.gamma == 0.0
    for name in names:
        assert np.abs(truth.residual_response(name)).max() == 0.0
    # every context then carries the identical context-blind effect
    assert np.abs(truth.response["A"] - truth.response["B"]).max() < 1e-12


def test_50_the_pass_rule_refuses_a_pipeline_that_recovers_nothing():
    from gene_open_inverse.four_context_v1 import identifiability as idt
    dead = {s: {"pooled": {"paired": {"V2_minus_V1": {"median": 0.0},
                                      "V2_minus_PARAM_MATCHED_BLIND": {"median": 0.0}},
                           "recovery": {"residual_cosine": {"median": 0.0}}},
                "modulation_shut_off": True}
            for s in ("0.0", "0.1", "0.25", "0.5")}
    assert idt.pass_criteria(dead)["verdict"] == "V2_NOT_IDENTIFIABLE_WITH_THREE_TRAINING_CONTEXTS"
    alive = {
        "0.0": {"pooled": {"paired": {"V2_minus_V1": {"median": 0.0},
                                      "V2_minus_PARAM_MATCHED_BLIND": {"median": 0.0}},
                           "recovery": {"residual_cosine": {"median": 0.0}}},
                "modulation_shut_off": True},
        "0.1": {"pooled": {"paired": {"V2_minus_V1": {"median": 0.001},
                                      "V2_minus_PARAM_MATCHED_BLIND": {"median": 0.001}},
                           "recovery": {"residual_cosine": {"median": 0.15}}},
                "modulation_shut_off": False},
        "0.25": {"pooled": {"paired": {"V2_minus_V1": {"median": 0.01},
                                       "V2_minus_PARAM_MATCHED_BLIND": {"median": 0.01}},
                            "recovery": {"residual_cosine": {"median": 0.40}}},
                 "modulation_shut_off": False},
        "0.5": {"pooled": {"paired": {"V2_minus_V1": {"median": 0.03},
                                      "V2_minus_PARAM_MATCHED_BLIND": {"median": 0.02}},
                           "recovery": {"residual_cosine": {"median": 0.70}}},
                "modulation_shut_off": False},
    }
    assert idt.pass_criteria(alive)["verdict"] == "V2_IDENTIFIABLE_WITH_THREE_TRAINING_CONTEXTS"


def test_51_the_null_level_must_shut_the_modulation_off_to_pass():
    from gene_open_inverse.four_context_v1 import identifiability as idt
    curve = {
        "0.0": {"pooled": {"paired": {"V2_minus_V1": {"median": 0.02},
                                      "V2_minus_PARAM_MATCHED_BLIND": {"median": 0.02}},
                           "recovery": {"residual_cosine": {"median": 0.0}}},
                "modulation_shut_off": False},
        "0.1": {"pooled": {"paired": {"V2_minus_V1": {"median": 0.01},
                                      "V2_minus_PARAM_MATCHED_BLIND": {"median": 0.01}},
                           "recovery": {"residual_cosine": {"median": 0.2}}},
                "modulation_shut_off": False},
        "0.25": {"pooled": {"paired": {"V2_minus_V1": {"median": 0.02},
                                       "V2_minus_PARAM_MATCHED_BLIND": {"median": 0.02}},
                            "recovery": {"residual_cosine": {"median": 0.5}}},
                 "modulation_shut_off": False},
        "0.5": {"pooled": {"paired": {"V2_minus_V1": {"median": 0.03},
                                      "V2_minus_PARAM_MATCHED_BLIND": {"median": 0.03}},
                           "recovery": {"residual_cosine": {"median": 0.8}}},
                "modulation_shut_off": False},
    }
    # recovery looks perfect, but a modulation that fires on an absent residual makes
    # the whole curve uninterpretable, so the verdict must still be NO.
    assert idt.pass_criteria(curve)["verdict"] == "V2_NOT_IDENTIFIABLE_WITH_THREE_TRAINING_CONTEXTS"


def test_52_the_planted_response_carries_exactly_two_measurement_views():
    # The regression test for the first positive-control run, which stacked an already
    # two-view array again and produced a four-dimensional response.
    from gene_open_inverse.four_context_v1 import identifiability as idt
    generator = np.random.default_rng(61)
    candidates, genes = 40, 30
    features = generator.standard_normal((candidates, 8))
    names = ("A", "B", "C")
    h = {name: generator.standard_normal((2, candidates, fc.CONTEXT_DIM)) for name in names}
    present = {name: np.ones(candidates, dtype=bool) for name in names}
    truth = idt.plant(features, h, present, genes, 0.25)
    for name in names:
        assert truth.response[name].shape == (2, candidates, genes)
    pack = idt._pack("A", np.asarray([f"g{i}" for i in range(candidates)]),
                     truth.response["A"], generator.standard_normal((2, candidates, genes)),
                     features, np.ones(candidates, dtype=bool), 0.25, 1.0, 5)
    assert pack.responses.shape == (2, candidates, genes)
    assert pack.utility[0].shape == (candidates, candidates)


def test_53_recovery_compares_against_a_two_view_planted_effect():
    # Second instance of the same slip: truth.response already carries both views, so
    # re-stacking it gave a four-dimensional array.  The guard makes a repeat loud.
    from gene_open_inverse.four_context_v1 import identifiability as idt
    generator = np.random.default_rng(67)
    candidates, genes = 50, 40
    features = generator.standard_normal((candidates, 8))
    names = ("A", "B", "C")
    h = {name: generator.standard_normal((2, candidates, fc.CONTEXT_DIM)) for name in names}
    present = {name: np.ones(candidates, dtype=bool) for name in names}
    truth = idt.plant(features, h, present, genes, 0.25)
    assert truth.residual_response("A").shape == (2, candidates, genes)
    assert truth.response["A"].shape == (2, candidates, genes)


def test_54_the_synthetic_held_context_is_distinguishable_from_the_training_ones():
    # A random direction in gene space projects to nearly nothing in a context basis
    # fitted on three contexts, so the held context would collapse onto the training
    # centre and a successful recovery would mean nothing.  The offset is therefore
    # built inside the span the basis can see.
    from gene_open_inverse.four_context_v1 import identifiability as idt
    from gene_open_inverse.four_context_v1 import v2 as v2m
    generator = np.random.default_rng(71)
    genes, candidates = 400, 60
    centres = generator.standard_normal((3, genes)) * 3.0
    training = {name: (centres[i][None, None, :]
                       + 0.4 * generator.standard_normal((2, candidates, genes))).astype(np.float32)
                for i, name in enumerate(("A", "B", "C"))}
    held = idt.synthetic_held_sources(training, candidates, genes)
    assert held.shape == (2, candidates, genes)

    pooled = np.concatenate([np.asarray(training[n][v], dtype=np.float64)
                             for n in training for v in (0, 1)])
    projection = v2m.context_basis(pooled, dimension=8)
    h = {n: np.stack([v2m.project_context(training[n][v], projection) for v in (0, 1)])
         for n in training}
    h["H"] = np.stack([v2m.project_context(held[v], projection) for v in (0, 1)])
    geometry = idt.context_geometry(h, ("A", "B", "C"))
    entry = geometry["H_distance_to_training_centres"]
    assert entry["is_a_distinguishable_context"], geometry
    # and it is not absurdly far either: a fourth cell line, not a different planet
    assert entry["ratio_to_between_training_median"] < 3.0, geometry


def test_55_a_random_gene_space_direction_would_have_collapsed():
    # The negative counterpart, showing the first implementation's flaw rather than
    # asserting it: an offset drawn isotropically in gene space lands far closer to the
    # training centre, in the projected space, than one built inside the span.
    from gene_open_inverse.four_context_v1 import identifiability as idt
    from gene_open_inverse.four_context_v1 import v2 as v2m
    generator = np.random.default_rng(73)
    genes, candidates = 400, 60
    centres = generator.standard_normal((3, genes)) * 3.0
    training = {name: (centres[i][None, None, :]
                       + 0.4 * generator.standard_normal((2, candidates, genes))).astype(np.float32)
                for i, name in enumerate(("A", "B", "C"))}
    pooled = np.concatenate([np.asarray(training[n][v], dtype=np.float64)
                             for n in training for v in (0, 1)])
    projection = v2m.context_basis(pooled, dimension=8)
    centre = pooled.mean(axis=0)
    spread = float(np.median([np.linalg.norm(c - centre) for c in centres]))

    isotropic = generator.standard_normal(genes)
    isotropic /= np.linalg.norm(isotropic)
    naive = (centre + spread * isotropic)[None, None, :] * np.ones((2, candidates, 1))
    inside = idt.synthetic_held_sources(training, candidates, genes)

    def displacement(block):
        h = np.stack([v2m.project_context(np.asarray(block[v], dtype=np.float64), projection)
                      for v in (0, 1)])
        return float(np.linalg.norm(h.reshape(-1, h.shape[-1]).mean(axis=0)))

    assert displacement(naive) < 0.5 * displacement(inside)


def test_56_the_hindsight_sweep_never_enters_the_verdict():
    # The held-context penalty sweep exists to attribute a failure between the model and
    # its selection rule.  If it ever leaked into the PASS rule the control would be
    # selecting on the answer, so this pins that it does not.
    import inspect

    from gene_open_inverse.four_context_v1 import identifiability as idt
    source = inspect.getsource(idt.pass_criteria)
    for forbidden in ("hindsight", "sweep", "lambda_sweep"):
        assert forbidden not in source, f"pass_criteria reads {forbidden}"


def test_57_penalty_selection_keys_survive_a_json_round_trip():
    # The record is read back from disk by the documents and by the identifiability
    # control.  A float-keyed dict is stringified on the way out, so an in-memory reader
    # and a from-disk reader would disagree about how to index it.
    import json

    from gene_open_inverse.four_context_v1 import v2 as v2m
    fake = {str(p): float(i) for i, p in enumerate(fc.CONTEXT_PENALTIES)}
    assert set(json.loads(json.dumps(fake))) == set(fake)
    for penalty in fc.CONTEXT_PENALTIES:
        assert str(penalty) in fake
    assert v2m.fc.CONTEXT_PENALTIES == fc.CONTEXT_PENALTIES


def test_58_matched_basis_planting_is_inside_the_models_own_family():
    # The point of MATCHED_BASIS: the model does not learn A and B, it fixes them, so a
    # residual planted with independently drawn ones may be unrepresentable and a
    # failure would say nothing about estimation.  Here the planted A and B are the ones
    # the model derives, so only W_m separates truth from model.
    from gene_open_inverse.four_context_v1 import identifiability as idt
    generator = np.random.default_rng(79)
    candidates, genes = 200, 120
    features = generator.standard_normal((candidates, 16))
    names = ("A", "B", "C", "H")
    h = {n: generator.standard_normal((2, candidates, fc.CONTEXT_DIM)) for n in names}
    present = {n: np.ones(candidates, dtype=bool) for n in names}
    training = ("A", "B", "C")

    matched = idt.plant(features, h, present, genes, 0.25, mode="MATCHED_BASIS",
                        training=training)
    free = idt.plant(features, h, present, genes, 0.25, mode="FREE_BASIS", training=training)
    assert matched.mode == "MATCHED_BASIS" and free.mode == "FREE_BASIS"
    assert matched.achieved_share == pytest.approx(0.25, abs=1e-9)
    assert free.achieved_share == pytest.approx(0.25, abs=1e-9)
    # the two modes really do plant different residuals
    assert not np.allclose(matched.residual_z["H"], free.residual_z["H"])
    # and the matched one uses the leading directions of the true z, reproducibly
    a = idt._leading_directions(matched.shared_z, fc.INTERACTION_RANK)
    assert a.shape == (idt.TRUE_LATENT, fc.INTERACTION_RANK)


def test_59_leading_directions_are_orthonormal_and_ordered():
    from gene_open_inverse.four_context_v1 import identifiability as idt
    generator = np.random.default_rng(83)
    values = generator.standard_normal((300, 40)) @ np.diag(np.linspace(10, 1, 40))
    directions = idt._leading_directions(values, 8)
    assert directions.shape == (40, 8)
    assert np.abs(directions.T @ directions - np.eye(8)).max() < 1e-8


def test_60_a_real_held_context_contributes_sources_but_never_responses():
    # The firewall for the Jurkat repeat.  The held context supplies control-derived
    # source states, and its effects are planted, so its measured perturbation responses
    # are never read even in principle.  Verified on real data at a correlation of
    # -0.0015; this pins the property without needing the data.
    from gene_open_inverse.four_context_v1 import identifiability as idt
    generator = np.random.default_rng(89)
    candidates, genes, width = 60, 80, 10
    names = ("A", "B", "C", "HELD")
    sources = {n: (generator.standard_normal((2, candidates, genes)) + i).astype(np.float32)
               for i, n in enumerate(names)}
    real = {"genes": np.asarray([f"g{i}" for i in range(genes)]),
            "candidates": np.asarray([f"c{i}" for i in range(candidates)]),
            "sources": sources, "features": generator.standard_normal((candidates, width)),
            "present": np.ones(candidates, dtype=bool),
            "candidate_feature_check": {}, "training": ("A", "B", "C"),
            "held_name": "HELD", "held_is_real": True}
    packs, truth, _, geometry = idt.build_world(real, 0.25, 11, 1.0, "MATCHED_BASIS")

    assert idt.held_name(real) == "HELD"
    held = packs["HELD"]
    # its sources are the real ones it was handed
    assert np.allclose(held.sources, sources["HELD"])
    # its responses come from the planted truth, not from anything measured
    assert np.abs(held.responses - truth.response["HELD"]).max() > 0.0   # noise added
    planted = truth.response["HELD"]
    assert np.corrcoef(held.responses.ravel(), planted.ravel())[0, 1] > 0.9
    # and the held context never contributes a training row
    assert "HELD" not in real["training"]
    assert geometry["HELD_distance_to_training_centres"]["is_a_distinguishable_context"]
