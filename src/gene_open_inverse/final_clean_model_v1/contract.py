"""The final contract.  Frozen before training, not revisited.

Method of record: ``CANDIDATE_EFFECT_DISTILLATION_V1``.

What is frozen and why
----------------------
The effect predictor keeps every hyperparameter it already had: rank-256 ``G_fit``
response basis, ridge penalty 1e4 chosen by ``G_fit`` five-fold identity CV,
STRING and MAPKG descriptors with present flags, consensus response target.  None
of it is re-selected here.

The fusion is z-score additive with ``beta = 1``.  That choice was made **before**
any ``G_check`` access and for one reason: it is the only fusion in the family with
no fitted coefficient at all, so it cannot have been tuned on anything.  It was not
chosen because it performed better on ``G_check``, which has not been read, and it
is not being re-compared against rank-average or ``beta = 0.25``.

TEXT is excluded from the final primary track because
``GENE_INFORMATION_POLICY_v1`` bars literature-derived gene descriptions from the
primary model, reaffirmed 2026-09-09.  The legacy incumbent that carries TEXT stays
in the tables as historical context and is **not** the final paired comparator.

All-missing candidates
----------------------
Both arms use one handling, and it is not learnable.  A candidate with no legal
modality present contributes exactly zero to the score, for every query, in both
the base ranker and the effect branch.  The incumbent instead gave such candidates a
shared learnable token, which is why 100% of its goal-blind Top-10 were
information-free genes.  The neutral value is zero because that is a cosine's own
indifference point; it was not chosen by looking at ``G_select``.
"""
from __future__ import annotations

FINAL_METHOD = "CANDIDATE_EFFECT_DISTILLATION_V1"
FINAL_FUSION = "z_score_additive"
FINAL_BETA = 1.0
FINAL_FUSION_SELECTED_BEFORE_G_CHECK = True
FINAL_FUSION_SELECTION_REASON = (
    "method simplicity: it is the only member of the fusion family with no fitted "
    "coefficient, so no G_select or G_check quantity enters it. Not selected on performance."
)

PRIMARY_MODALITIES = ("STRING", "MAPKG")
TEXT_STATUS = "EXCLUDED_FROM_FINAL_PRIMARY_TRACK"
TEXT_REASON = "GENE_INFORMATION_POLICY_v1 literature-leakage bar, reaffirmed 2026-09-09"

ALL_MISSING_CONTRIBUTION = 0.0
ALL_MISSING_RULE = (
    "no learnable shared unknown token; a candidate with no legal modality contributes "
    "exactly zero to the score for every query, identically in both arms; the present "
    "mask is retained"
)

ARMS = ("CLEAN_BASE", "CLEAN_EFFECT")
PRIMARY_COMPARISON = ("CLEAN_EFFECT", "CLEAN_BASE")
LEGACY_COMPARATOR_IS_NOT_PRIMARY = True

EVALUATION_SPLIT_NAME = "locked final evaluation split"
EVALUATION_SPLIT_MUST_NOT_BE_CALLED = "untouched blind test"
NO_CONFIG_CHANGE_ALLOWED_AFTER_G_CHECK_ACCESS = True
