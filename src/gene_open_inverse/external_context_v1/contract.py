"""The external context contract, frozen before any model touches it.

The question
-----------
Does ``CANDIDATE_EFFECT_DISTILLATION_V1`` encode a transferable intervention
mechanism, or a K562-specific response map?

The primary external context is Replogle RPE1: the same laboratory, the same
platform, the same perturbation type (CRISPRi knockdown) and the same processing,
in a different biological context (retinal pigment epithelium, non-cancer, against
K562 erythroleukemia).  That combination is what makes it the controlled comparison
for this question: cell context varies and almost nothing else does.

What is frozen here
-------------------
The batch-disjoint split rule, the response recipe, the eligibility rule and the
identity split are all inherited from the K562 contracts and applied unchanged.
They are fixed before any transfer number exists and are never adjusted on a
method result.

The gene-space contract
-----------------------
``G_common`` is the intersection of the two measured gene axes, taken once.  Genes
are never selected on a result.  A K562 predicted effect is restricted to
``G_common``; an RPE1 desired transition is restricted to the same set; the score
is their cosine there.

Two kinds of generalization
---------------------------
They are different questions and are never pooled into one number.  A candidate
that appeared in K562 ``G_fit`` had its K562 response seen by the effect predictor,
so evaluating it in RPE1 tests *context* transfer.  A candidate absent from K562
``G_fit`` tests *candidate* transfer as well.  Results are reported on both strata
and on the union.
"""
from __future__ import annotations

PRIMARY_CONTEXT = "RPE1"
PRIMARY_SOURCE = "inputs/ReplogleWeissman2022_rpe1.h5ad"
ANCHOR_CONTEXT = "K562"
PERTURBATION_TYPE = "CRISPRi knockdown, identical to the anchor"

CONTROL_LABEL = "control"
LIBRARY_TARGET = 1.0e4
MIN_VIEW_CELLS = 25              # inherited from the frozen K562 lock, not re-chosen
VIEW_SEED = 20260914             # inherited
VIEW_NAMESPACE = "REPLOGLE_BATCH_DISJOINT_V1"   # inherited
QUAD_NAMESPACE = "REPLOGLE_BATCH_QUAD_REFINEMENT_V1"   # inherited

# The external identity split, fixed by hash before any model runs.
SPLIT_NAMESPACE = "RPE1_EXTERNAL_ROLE_SPLIT_V1"
SPLIT_SEED = 20260918
SPLIT_FRACTIONS = {"E_fit": 6, "E_select": 2, "E_check": 2}   # buckets out of ten

ELIGIBILITY_FDR = 0.05           # the frozen QUERY_ELIGIBILITY_V1 level, unchanged

STRATA = ("seen_in_K562_distillation", "unseen_in_K562_distillation", "all_candidates")
ARMS = ("ZS_BASE", "ZS_EFFECT")
PRIMARY_COMPARISON = ("ZS_EFFECT", "ZS_BASE")

TRANSFER_CONVENTION = (
    "an RPE1 state vector is embedded into the anchor's 8248-gene axis by placing the "
    "measured value at each G_common gene and zero elsewhere, so the frozen anchor model "
    "runs unchanged; the effect cosine is computed on G_common only"
)
ACTION_MODEL_NOTE = (
    "the perturbation type is identical to the anchor's, so the attenuation action model "
    "and its J guardrail remain applicable and are reported"
)
