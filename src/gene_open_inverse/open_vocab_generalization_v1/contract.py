"""The open-vocabulary contract, frozen before any support number or masked result exists.

The question
------------
Can historical perturbation measurements improve design for interventions whose
responses have never been measured?  Everything here is judged on budgeted design
utility and never on response reconstruction accuracy, because the project's object
is an experiment list of size 10, 20 or 50 and not a predicted expression vector.

Why the existing evidence does not settle it
--------------------------------------------
C-014 found ``delta MBRU = -0.0103`` on the K562 fold's 6255 training-atlas-unseen
candidates against ``+0.0095`` on the same fold's 1570 seen candidates.  That is a
natural experiment: the two candidate sets were not assigned, they differ by which
screen library measured them, and candidate novelty is entangled with context
novelty.  Two incompatible readings survive it.  H1 says unseen candidates fail
because their static knowledge is poorly supported.  H2 says static knowledge cannot
extrapolate an effect to an intervention with no response supervision, however well
supported it is.  This contract exists to separate them.

The factorial
-------------
Two generalization axes, never added together.

  regime                interpretation
  C_seen  x G_seen      interpolation reference
  C_seen  x G_unseen    pure candidate transfer
  C_unseen x G_seen     pure context transfer, already established
  C_unseen x G_unseen   joint open-vocabulary transfer

Which masking design is primary, and why it is not the one the protocol lists first
-----------------------------------------------------------------------------------
The protocol gives two controlled-masking designs.  The within-context design trains
on one context and evaluates masked candidates in that same context.  Its SEEN arm
therefore fits on the evaluated candidate's own measured response, and the graded
utility is a cosine between measured responses of that same candidate.  The SEEN arm
is thus partly memorizing the quantity it is graded on, and the amount it can memorize
is governed by the candidate's ridge leverage, which is the very axis this mission
stratifies by.  ``SEEN - MASKED`` in that design is an upper bound on the worth of
response exposure and its interaction with support is mechanically confounded.

The cross-context design has no such defect.  The held context supplies no training
row to either arm, so neither predictor has ever seen the graded response.  The two
arms differ only in whether the same candidate identities were measured in the three
*other* contexts.  That is the clean factorial, so it is pre-registered here as the
primary evidence for the worth of historical response exposure.  The within-context
design is still run and reported, labelled as a within-context reference carrying the
self-memorization caveat.  This choice is recorded before any masked number exists.

Pool sizing, recorded before the first fit
------------------------------------------
At the pre-registered ``MASK_FOLDS = 5`` the masked pools are K562 286, RPE1 254,
Jurkat 290 and HepG2 152 candidates.  Every one clears the protocol's hard floor of
"much larger than 50" and none reaches its target of 300.  The fold count is not moved
to reach the target, because a fold count chosen after reading the sizes is a chosen
masking fold.  Instead a primary-inclusion floor is fixed here at 200, which is four
times the largest budget, so that the largest budget never takes more than a quarter
of the pool it ranks.  A fold below it is diagnostic and does not enter the pooled
verdict.  On those numbers HepG2 is diagnostic-only in the per-fold contrast and still
enters the cross-fitted full-pool contrast, whose pool is 764.

What is inherited and never re-opened
--------------------------------------
The frozen CLEAN_BASE scorer, the frozen candidate feature contract, the Candidate
Effect Distillation V1 architecture, the basis rank, the ridge penalty grid and
selection criterion, the fusion weight, the query eligibility gate, the utility
definition and the budget set.  This mission adds no model.  It measures a capability
boundary of the model that already exists.

An inherited property that is recorded rather than fixed
---------------------------------------------------------
``four_context_v1.predictors._cross_validate`` folds by row index.  In a pooled atlas
a candidate measured in three contexts contributes three rows that can land in
different validation folds, so the penalty selection inside a pooled training set is
mildly optimistic.  That is a property of the frozen recipe and is left alone: changing
it would make every arm here a different method from the one whose boundary is being
measured.  It affects only the penalty choice, it is identical across arms, and in the
G_unseen arm the masked identities are absent from the pool entirely, so nothing about
them leaks through it.
"""
from __future__ import annotations

# --- what is being generalized over ----------------------------------------
GENERALIZATION_AXES = ("context", "candidate")
REGIMES = ("C_seen_G_seen", "C_seen_G_unseen", "C_unseen_G_seen", "C_unseen_G_unseen")
PRIMARY_QUANTITY = "delta_MBRU_effect_minus_base"
VERDICT_QUANTITY = "budgeted intervention design utility, never response reconstruction"

# --- Phase A, the natural unseen audit -------------------------------------
NATURAL_FOLD = "K562"              # the only fold whose unseen pool is not degenerate
NATURAL_TRAINING = ("RPE1", "HepG2", "Jurkat")
COVERAGE_PATTERNS = ("BOTH", "STRING_ONLY", "MAPKG_ONLY", "NEITHER")
# The static feature vector that actually enters CED, in its frozen layout.
STRING_WIDTH = 512
MAPKG_WIDTH = 1024
FEATURE_WIDTH = STRING_WIDTH + MAPKG_WIDTH + 2      # two modality present flags
STRING_FLAG_COLUMN = STRING_WIDTH + MAPKG_WIDTH
MAPKG_FLAG_COLUMN = STRING_WIDTH + MAPKG_WIDTH + 1

# --- neighbourhood support -------------------------------------------------
NEIGHBOUR_K = 10                   # d_10 is primary; k is fixed here, not chosen later
NEIGHBOUR_SENSITIVITY = (1, 50)    # reported beside it, never substituted for it
NEIGHBOUR_SPACE = (
    "the frozen standardized candidate feature vector that enters CED, Euclidean, "
    "against the unique identities of the training atlas"
)

# --- the primary support diagnostic ----------------------------------------
PRIMARY_SUPPORT = "ridge_leverage"
LEVERAGE_DEFINITION = (
    "L_c = k_c^T (X^T X + lambda I)^{-1} k_c with X the training feature matrix in the "
    "ridge's own standardization and lambda the penalty that fit actually selected; "
    "k_c is standardized by the same fitted mean and scale"
)
SUPPORT_SECONDARY = ("coverage_pattern", "d10", "string_degree")
NO_OOD_DETECTOR = "no uncertainty model is trained anywhere in this mission"

# --- Phase A stratification -------------------------------------------------
SUPPORT_STRATA = 5                 # equal-count quintiles of L_c, best supported first
STRATUM_NAMES = ("Q1_best_supported", "Q2", "Q3", "Q4", "Q5_weakest_supported")

# --- Phase A support matching ----------------------------------------------
MATCH_EXACT = ("coverage_pattern",)
MATCH_CALIPER_VARIABLES = ("leverage_percentile", "d10_percentile")
# A caliper of 0.05 on each of two percentile axes defines a box covering 1 percent of
# the joint support space.  With 5926 present unseen candidates to draw from and 1430
# present seen candidates to match, an evenly spread box would hold about 59 unseen
# candidates, which is ample for one-to-one matching without replacement.  Fixed here,
# with 0.02 reported as a sensitivity and never substituted for it.
MATCH_CALIPER = 0.05
MATCH_CALIPER_SENSITIVITY = 0.02
MATCH_BALANCE_SECONDARY = ("string_degree_percentile",)
MATCH_RULE = (
    "one to one without replacement, greedy in ascending order of the matched pair's "
    "joint caliper distance, ties broken by identity symbol; both pools therefore have "
    "exactly the same size and absolute MBRU is comparable between them"
)

# --- Phase B, controlled candidate masking ---------------------------------
MASK_NAMESPACE = "VCDESIGN_OPEN_VOCAB_CANDIDATE_MASK_V1"
MASK_FOLDS = 5
MASK_PRIMARY_CONTEXTS = ("RPE1", "HepG2", "Jurkat")
MASK_SECONDARY_CONTEXTS = ("K562",)
MASK_SECONDARY_REASON = (
    "the frozen base ranker was trained on K562 G_fit responses, so a K562 masked "
    "candidate can retain response exposure through the base even after the effect "
    "branch loses it"
)
MIN_PRIMARY_POOL = 200             # four times the largest budget
PRIMARY_MASKING_DESIGN = "cross_context"
WITHIN_CONTEXT_CAVEAT = (
    "its SEEN arm fits on the evaluated candidate's own graded response, so "
    "SEEN minus MASKED is an upper bound and its interaction with leverage is "
    "mechanically confounded"
)
CROSSFIT_HOMOGENEITY_CHECK = (
    "the five masked predictors of one context must agree on the common response "
    "direction before their per-candidate effects may be mixed into one score matrix; "
    "reported as the minimum pairwise cosine between their intercept reconstructions"
)
CROSSFIT_HOMOGENEITY_FLOOR = 0.99
# The direction test alone cannot fail, so the gate also requires that the mixed
# predictors selected the same ridge penalty and shrink by the same amount.  The
# tolerance on the candidate-specific share is a judgement fixed before the numbers, not
# a derivation.  It governs only the secondary SEEN minus MASKED contrast: the primary
# SIZE_CONTROL minus MASKED contrast builds both of its arms out of the same five
# predictors permuted by one fold, so any mixing distortion is common to both.
CROSSFIT_SCALE_TOLERANCE = 1.05
# Removing a candidate's rows also removes rows.  This programme established in C-013
# that atlas size, not context diversity, is the primary driver of effect distillation
# quality, so SEEN minus MASKED cannot separate "this candidate was never measured" from
# "the atlas is a fifth smaller".  The size control removes a different fold of the same
# construction instead, which is exactly the next fold's masked predictor, so it costs no
# extra fit and is fixed by the same hash rather than by a draw.
SIZE_CONTROL_RULE = "fold M is size controlled by the predictor that masked fold (M + 1) mod 5"
EXPOSURE_DECOMPOSITION = {
    "exposure_value": "SEEN minus MASKED, the upper bound, atlas shrinkage included",
    "exposure_value_size_controlled": "SIZE_CONTROL minus MASKED, candidate specific",
    "atlas_shrinkage_cost": "SEEN minus SIZE_CONTROL, what one fifth less atlas costs",
}
PRIMARY_EXPOSURE_QUANTITY = "exposure_value_size_controlled"

# An inherited defect, recorded rather than fixed.  ``evaluate_arms`` concatenates the two
# batch-disjoint query views into one vector of length 2Q, and the frozen
# ``cluster_bootstrap`` resamples that vector entry by entry, so the two views of one
# query enter as two independent clusters even though its own docstring names the query
# identity as the cluster unit.  Every interval in the four-context programme, C-013 and
# C-014 included, is therefore narrower than the stated cluster unit implies.  Changing
# the frozen function would make these numbers incomparable with those, so it is left
# alone and a correct query-identity interval is reported beside it as a sensitivity.
INHERITED_BOOTSTRAP_DEFECT = (
    "the frozen cluster bootstrap resamples query views, not query identities; a correct "
    "query-identity interval is reported beside every headline contrast"
)

# --- Phase B support interaction -------------------------------------------
MASK_SUPPORT_STRATA = 3
MASK_STRATUM_NAMES = ("high_support", "medium_support", "low_support")

# --- Phase C, open-vocabulary deployment mixture ---------------------------
MIXTURE_NAMESPACE = "VCDESIGN_OPEN_VOCAB_MIXTURE_V1"
MIXTURE_SHARES = (0.25, 0.50, 0.75)
MIXTURE_RULE = (
    "candidate identity and candidate pool are held fixed across shares; only which "
    "candidates take their predicted effect from the predictor that masked them changes"
)
ENTRANT_VALUE_DEFINITION = (
    "for a query q and budget B, a candidate entering the top B under the fused effect "
    "score but not under the base alone is credited with utility(q, c) minus the mean "
    "utility of the candidates it displaced; this is what the effect branch actually "
    "bought by promoting it"
)

# --- Section 21, support must predict harm before any gate is built --------
NO_GATE_IN_THIS_MISSION = (
    "a support-aware gate is not implemented here under any name; this mission only "
    "measures whether support predicts the sign of what the effect branch buys"
)
SUPPORT_HARM_PRIMARY = "spearman(leverage_percentile, mean_entrant_value)"

# --- Section 22, the only verdicts this mission may reach ------------------
VERDICTS = (
    "STATIC_KNOWLEDGE_SUPPORT_LIMITED",
    "MEASURED_CANDIDATE_COVERAGE_LIMITED",
    "SUPPORT_DEPENDENT_OPEN_VOCABULARY",
    "OPEN_VOCABULARY_FAILURE_UNRESOLVED",
)

# --- hard stops -------------------------------------------------------------
FROZEN_AND_NOT_REOPENED = (
    "K562 CLEAN_BASE", "K562 CLEAN_EFFECT", "K562 G_check", "G_4", "query eligibility",
    "design utility", "sampler", "scorer", "DTA", "context-conditioned V2",
    "Candidate Effect Distillation architecture", "ridge penalty recipe",
    "response rank", "fusion weight",
)
NO_NEW_CANDIDATE_EMBEDDING = (
    "the STRING adjacency is read for degree and neighbour counts as a diagnostic only; "
    "no diagnostic quantity in this mission is ever given to a model as a feature"
)
