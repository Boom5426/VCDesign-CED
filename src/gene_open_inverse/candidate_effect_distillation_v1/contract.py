"""Frozen before the first number.

The hypothesis
--------------
The problem with STRING and MAPKG is not only that they carry little information.
It is that they live in a space with no relationship to the desired transition, so
a ranker has to learn both what a candidate does *and* how to compare it, from a
supervision signal that is already weak.  Distillation removes the second half of
that problem: learn ``static knowledge -> signed gene-space effect`` on ``G_fit``,
where measured responses are legally available, and then a deployment candidate's
predicted effect lives in the same 8248-gene coordinate system as ``d_q``, where the
comparison is a cosine and needs no learning at all.

What is legal
-------------
Training may read ``G_fit`` static knowledge together with ``G_fit`` measured
responses.  Inference may read only a candidate's static knowledge.  A ``G_select``
candidate's own measured response is used for grading and never as an input, in any
view.  ``G_check`` is sealed.

TEXT
----
``GENE_INFORMATION_POLICY_v1`` bars literature-derived gene descriptions from the
primary model and `ROADMAP_V1_ASSESSMENT` reaffirmed it on 2026-09-09.  The live
``decision_alignment_v1_assets.json`` nevertheless carries TEXT.  This phase does not
resolve that conflict by argument; it resolves it by not depending on it.  The
primary track uses STRING and MAPKG only.  Anything carrying TEXT is reported as a
legacy-contract comparator and labelled.

All-missing candidates
----------------------
A candidate with no static modality present has no predicted effect.  It is given
an exactly neutral score rather than a learned one, because a shared learnable
token for such candidates is what put 92 information-free genes at the top of every
goal-blind ordering in the previous phase.  Neutral means zero: a cosine's own
indifference point, fixed here and not tuned.
"""
from __future__ import annotations

# --- contract -------------------------------------------------------------
PRIMARY_MODALITIES = ("STRING", "MAPKG")
LEGACY_MODALITIES = ("STRING", "MAPKG", "TEXT")
PRIMARY_EXCLUDES_TEXT = True
TEXT_EXCLUSION_REASON = "GENE_INFORMATION_POLICY_v1 literature-leakage bar, reaffirmed 2026-09-09"

# --- response target ------------------------------------------------------
CONSENSUS = "mean of the two batch-disjoint view responses"
RESPONSE_DIM = 8248

# --- basis ----------------------------------------------------------------
BASIS_RANK = 256              # one value, fixed in advance, no grid
BASIS_CENTERED = False        # uncentered SVD: the basis spans responses, not deviations
BASIS_SEED = 20260918
BASIS_OVERSAMPLE = 32
BASIS_POWER_ITERATIONS = 4

# --- predictor ------------------------------------------------------------
RIDGE_FOLDS = 5
RIDGE_FOLD_SEED = 20260918
RIDGE_PENALTIES = (1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5)
RIDGE_FIT_POPULATION = "G_fit candidates with at least one primary modality present"
RIDGE_HAS_INTERCEPT = True    # the common response direction is a constant in z-space
# The penalty is chosen by held-out mean cosine between the reconstructed effect and the
# measured consensus response, not by held-out squared error, because the deployment use
# is a direction comparison and squared error would spend the penalty on magnitude.  Both
# criteria are G_fit-only; the squared-error choice is reported as a sensitivity.
RIDGE_CRITERION = "held_out_mean_cosine_in_gene_space"
RIDGE_SENSITIVITY_CRITERION = "held_out_mean_squared_error_in_basis_space"

# --- scoring --------------------------------------------------------------
ALL_MISSING_SCORE = 0.0
SCORE_NAME = "PREDICTED_EFFECT_COSINE"
PREDICTED_EFFECT_IS_NOT_AN_INFERENCE_INPUT = (
    "the predicted effect is a function of static knowledge alone; no measured "
    "response of the scored candidate is read at inference"
)
