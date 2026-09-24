"""The four-context contract, frozen before any HepG2 or Jurkat number exists.

The three questions
-------------------
Q1  Does a larger perturbation atlas improve ``static knowledge -> effect geometry``?
Q2  Does the improvement come from more training rows or from more biological contexts?
Q3  After controlling atlas size, does ``f(candidate, source context)`` beat
    ``f(candidate)`` on a cell context never seen in training?

Q2 is the reason this mission exists.  A pooled atlas has both more rows and more
contexts, so a pooled win is uninterpretable on its own.  Every scaling claim here
is therefore paired with a row-count-matched control in which the only thing that
changes is how many contexts the same number of rows came from.

What is inherited and never re-invented
---------------------------------------
The A/B batch hash, the quarter refinement, the minimum cell threshold, the
per-batch control subtraction, the identity-incidence-weighted control reference and
the ``QUERY_ELIGIBILITY_V1`` gate are taken from the frozen K562 lock and applied
unchanged to each new context.  Phase 2 of the protocol requires exactly this: the
measurement contract is reproduced, not redesigned around the new data.

The batch hash is applied to each context's own batch names.  Two contexts whose
lanes happen to carry the same names therefore receive the same A/B pattern.  That
is harmless because no quantity in this programme ever pairs a batch in one context
with a batch in another, and it is recorded rather than worked around.

What is new
-----------
The gene axis.  ``G_4`` is the intersection of the four measured axes, taken once,
before any model runs.  The historical K562 result keeps its 8248-gene axis and is
not recomputed; the four-context programme is a separate experiment on ``G_4``.

The base ranker
---------------
One fixed reference for every context and every fold: the frozen K562 ``CLEAN_BASE``
scorer, run unchanged by embedding a context's state vectors into the anchor's
8248-gene axis.  It is never retrained, never adapted and never told which context it
is looking at.  It is stronger on K562 than elsewhere because it was trained there,
which makes the K562 fold's ``effect minus base`` delta harder rather than easier;
that asymmetry is declared, not corrected.  For the same reason the K562 evaluation
population excludes the anchor's ``G_fit`` identities.  The anchor's ``G_check`` is
never opened.

Pre-registered numbers
----------------------
``CONTEXT_DIM``, ``INTERACTION_RANK``, ``CONTEXT_PENALTIES``, ``MATCH_SEEDS`` and the
row-budget rule are fixed in this file before the first model result.  None of them
may be changed after seeing a held-context number, and no grid over them is a search:
the penalty is chosen by leave-one-training-context-out validation inside the three
training contexts only.
"""
from __future__ import annotations

# --- contexts -------------------------------------------------------------
ANCHOR_CONTEXT = "K562"
CONTEXTS = ("K562", "RPE1", "HepG2", "Jurkat")
NEW_CONTEXTS = ("HepG2", "Jurkat")
ACCESSION = "GSE264667"
ACCESSION_TITLE = "Transcriptome-wide characterization of genetic perturbations"
NEW_CONTEXT_SOURCES = {
    "HepG2": "GSE264667_hepg2_raw_singlecell_01.h5ad",
    "Jurkat": "GSE264667_jurkat_raw_singlecell_01.h5ad",
}
PERTURBATION_TYPE = "gene-level CRISPRi knockdown, dual-sgRNA, day 7, 10x 3' v3"

# --- inherited measurement contract ---------------------------------------
LIBRARY_TARGET = 1.0e4
MIN_VIEW_CELLS = 25
VIEW_SEED = 20260914
VIEW_NAMESPACE = "REPLOGLE_BATCH_DISJOINT_V1"
QUAD_NAMESPACE = "REPLOGLE_BATCH_QUAD_REFINEMENT_V1"
ELIGIBILITY_FDR = 0.05
INHERITED_FROM = "the frozen K562 lock, VCDesign_FINAL_LOCK_2026-09-18.md"

# --- who may train on what, per context ------------------------------------
# There is deliberately no hash split inside a context.  In a leave-one-context-out
# fold the held context contributes no training row at all, so nothing it is graded
# on could have fitted the predictor, and splitting it further would only throw away
# half its queries.  The one exception is the anchor, where the shared base ranker
# was trained on G_fit: K562 is therefore queried on G_select alone, so the base is
# not evaluated in sample.  G_check is dropped entirely and never opened.
TRAIN_RULE = {
    "K562": "usable identities in the anchor G_fit role",
    "RPE1": "every usable identity",
    "HepG2": "every usable identity",
    "Jurkat": "every usable identity",
}
QUERY_RULE = {
    "K562": "usable and QUERY_ELIGIBILITY_V1 eligible identities in the anchor G_select role",
    "RPE1": "every usable and eligible identity",
    "HepG2": "every usable and eligible identity",
    "Jurkat": "every usable and eligible identity",
}
CANDIDATE_RULE = "every usable identity in the context, whatever its training role"

# --- hard qualification, Phase 1 -------------------------------------------
QUALIFICATION = (
    "gene_level_crispri_knockdown",
    "explicit_non_targeting_controls",
    "batch_or_gemgroup_structure",
    "every_retained_batch_has_controls",
    "min_view_cells_in_both_sides",
    "at_least_several_hundred_usable_identities",
    "sufficient_common_gene_axis",
    "no_candidate_response_in_inference_features",
)
MIN_USABLE_IDENTITIES = 300      # "several hundred", fixed before the audit runs

# --- candidate strata, Phase 5 ---------------------------------------------
STRATA = ("ALL", "SEEN_CANDIDATE", "UNSEEN_CANDIDATE", "COMMON4")
STRATA_NOTE = (
    "context generalization and candidate generalization are different questions and "
    "are never collapsed into one number"
)

# --- atlas scaling, Phases 7 and 8 -----------------------------------------
ARMS = ("BASE", "V1", "V2", "ORACLE")
PRIMARY_V2_COMPARISON = ("V2", "V1")
PRIMARY_SCALING_QUANTITY = "delta_MBRU_effect_minus_base"
ROW_BUDGET_RULE = (
    "the largest budget every comparable arm can meet: the minimum, over the single "
    "context training designs available in a fold, of that design's usable row count; "
    "computed from data properties alone and never from a model result"
)
MATCH_SEEDS = (20260918, 20260919, 20260920, 20260921, 20260922)   # five, fixed

# --- context representation, Phase 12 --------------------------------------
CONTEXT_DIM = 32                 # fixed before the first V2 number; no dimension search
CONTEXT_BASIS_SEED = 20260918
CONTEXT_BASIS_SOURCE = (
    "training-context source states only: the view-specific, perturbation-cell-incidence "
    "weighted mean of batch-specific normalized control pseudobulks.  This is the same "
    "object in all four contexts, it is what a deployment caller actually holds, and it "
    "gives one datapoint per identity and view rather than one per context.  No perturbed "
    "cell enters it."
)
# The modulation is the Hadamard product of two projections, so the number of
# interaction features equals the rank, and the rank cannot usefully exceed the
# context dimension because v_x = B h_x has at most CONTEXT_DIM degrees of freedom.
# Setting the two equal is therefore the largest value the architecture can use, not
# a tuned one.  Fixed before the first V2 number; there is no rank search.
INTERACTION_RANK = 32
# The grid reaches high enough that the modulation is effectively switched off at
# its top end, so V1 is inside the model class the selection can choose.
CONTEXT_PENALTIES = (1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7)
CONTEXT_PENALTY_SELECTION = (
    "leave-one-training-context-out inside the three training contexts; the held "
    "context never participates"
)
V2_IS_RESIDUAL = True
V2_RESIDUAL_NOTE = (
    "V2 = V1 + a low-rank context residual.  If context carries nothing the "
    "modulation weight goes to zero and V2 degenerates to V1 by construction."
)

# --- headroom gate, Phase 10 ------------------------------------------------
RESIDUAL_GATE_NOTE = (
    "V2 is built only if the context residual is both non-trivial in variance share "
    "and reproducible across the two independent measurement views"
)

# --- deployment firewall ----------------------------------------------------
DEPLOYMENT_INPUTS = ("candidate static knowledge", "source or control state")
FORBIDDEN_INPUTS = (
    "candidate perturbed state",
    "target perturbation outcome",
    "held-context candidate response",
)
ANCHOR_G_CHECK_IS_SEALED = True
