"""Protocol tests for the external-context transfer."""
from __future__ import annotations

import numpy as np
import pytest

from gene_open_inverse.external_context_v1 import contract as ec
from gene_open_inverse.external_context_v1.build import quarter_of, role_of, view_side
from gene_open_inverse.external_context_v1.transfer import (
    cross_view_cosine, embed_into_anchor, gene_bridge,
)


def test_01_the_A_B_split_rule_is_the_anchors_own():
    """A different split rule would make 'two independent views' a different object here."""
    from gene_open_inverse.measurement_v2_quad.split import quarter_of as anchor_quarter
    from gene_open_inverse.measurement_v2_quad.split import view_side as anchor_side

    for name in ("1", "17", "gem_group_3", "batch-A", "zzz"):
        assert view_side(name) == anchor_side(name)
        assert quarter_of(name) == anchor_quarter(name)


def test_02_quarters_refine_sides():
    for name in [str(index) for index in range(200)]:
        assert quarter_of(name) // 2 == view_side(name)


def test_03_the_role_split_is_deterministic_and_covers_every_identity():
    names = [f"GENE{index}" for index in range(4000)]
    roles = [role_of(name) for name in names]
    assert set(roles) <= set(ec.SPLIT_FRACTIONS)
    assert [role_of(name) for name in names] == roles
    counts = {role: roles.count(role) for role in ec.SPLIT_FRACTIONS}
    for role, width in ec.SPLIT_FRACTIONS.items():
        assert abs(counts[role] / len(names) - width / 10.0) < 0.05
    assert sum(ec.SPLIT_FRACTIONS.values()) == 10


def test_04_the_gene_bridge_is_the_intersection_and_indexes_both_axes():
    external = np.asarray(["A", "B", "C", "D"])
    anchor = np.asarray(["C", "A", "E"])
    bridge = gene_bridge(external, anchor)
    assert list(bridge["common"]) == ["A", "C"]
    assert bridge["size"] == 2
    assert list(external[bridge["external_positions"]]) == ["A", "C"]
    assert list(anchor[bridge["anchor_positions"]]) == ["A", "C"]


def test_05_embedding_places_shared_genes_and_zeroes_the_rest():
    external = np.asarray(["A", "B", "C", "D"])
    anchor = np.asarray(["C", "A", "E"])
    bridge = gene_bridge(external, anchor)
    values = np.asarray([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
    embedded = embed_into_anchor(values, bridge, anchor.size)
    assert embedded.shape == (2, 3)
    assert embedded[0].tolist() == [3.0, 1.0, 0.0]      # C, A, E-absent
    assert embedded[1].tolist() == [7.0, 5.0, 0.0]


def test_06_the_evaluator_grades_a_view_with_the_other_view():
    generator = np.random.default_rng(0)
    responses = generator.normal(size=(2, 6, 5))
    for view in (0, 1):
        got = cross_view_cosine(responses, view)
        other = responses[1 - view]
        unit = other / np.linalg.norm(other, axis=1, keepdims=True)
        assert np.abs(got - unit @ unit.T).max() < 1e-9
        assert np.abs(np.diag(got) - 1.0).max() < 1e-9


def test_07_the_two_generalization_notions_are_named_and_kept_apart():
    assert ec.STRATA == ("seen_in_K562_distillation", "unseen_in_K562_distillation", "all_candidates")
    assert ec.PRIMARY_COMPARISON == ("ZS_EFFECT", "ZS_BASE")


def test_08_inherited_constants_are_not_re_chosen():
    from gene_open_inverse.measurement_v2_quad.split import (
        MIN_VIEW_CELLS, QUAD_NAMESPACE, VIEW_NAMESPACE, VIEW_SEED,
    )

    assert ec.VIEW_SEED == VIEW_SEED
    assert ec.VIEW_NAMESPACE == VIEW_NAMESPACE
    assert ec.QUAD_NAMESPACE == QUAD_NAMESPACE
    assert ec.MIN_VIEW_CELLS == MIN_VIEW_CELLS
    assert ec.ELIGIBILITY_FDR == 0.05


def test_09_the_out_of_fold_keys_separate_contexts():
    """A gene measured in both contexts must be held out only for the context being graded."""
    from gene_open_inverse.external_context_v1.joint import fit_predictor, honest_effect

    generator = np.random.default_rng(3)
    count, genes = 40, 12
    latent = generator.normal(size=(count, 4))
    responses = latent @ generator.normal(size=(4, genes))
    features = np.concatenate([latent @ generator.normal(size=(4, 6)), np.ones((count, 1))], axis=1)
    present = np.ones(count, dtype=bool)
    keys = np.asarray([f"K562|G{i}" if i < 20 else f"RPE1|G{i - 20}" for i in range(count)])
    model = fit_predictor(features, responses, present, keys, rank=3)

    evaluated = np.asarray([f"RPE1|G{i}" for i in range(20)])
    held = honest_effect(model, features[20:], present[20:], evaluated)
    full = honest_effect(model, features[20:], present[20:], None)
    assert np.abs(held - full).max() > 0.0          # the RPE1 rows really were held out
    assert held.shape == (20, genes)


def test_10_an_absent_candidate_keeps_the_declared_neutral_effect():
    from gene_open_inverse.external_context_v1.joint import fit_predictor, honest_effect

    generator = np.random.default_rng(5)
    count, genes = 30, 9
    responses = generator.normal(size=(count, genes))
    features = generator.normal(size=(count, 5))
    present = np.ones(count, dtype=bool)
    keys = np.asarray([f"RPE1|G{i}" for i in range(count)])
    model = fit_predictor(features, responses, present, keys, rank=3)
    evaluation_present = present.copy()
    evaluation_present[:3] = False
    effect = honest_effect(model, features, evaluation_present, None)
    assert np.abs(effect[:3]).max() == 0.0


def test_11_the_perturbation_type_matches_the_anchor():
    assert "CRISPRi" in ec.PERTURBATION_TYPE
    assert ec.PRIMARY_CONTEXT == "RPE1" and ec.ANCHOR_CONTEXT == "K562"
    assert "zero elsewhere" in ec.TRANSFER_CONVENTION
