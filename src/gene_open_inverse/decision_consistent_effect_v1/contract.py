"""DECISION_CONSISTENT_EFFECT_DISTILLATION_V1, frozen before the first number.

The single question
-------------------
Does the current Candidate Effect Distillation lose candidate-specific directional
predictability by unit normalizing its predicted effect a second time, and does that
loss limit intervention design?

Why the question is well posed
------------------------------
The graded utility is a cosine between unit-normalized measured responses,
``U(q, c) = <u_q, u_c>`` with ``u_c = r_c / ||r_c||``.  A candidate estimator that is
consistent with that decision predicts ``m_hat_c ~ E[u_c | k_c]`` and scores
``<u_q, m_hat_c>``.  The current method instead learns the raw response ``r_c`` and
scores ``<u_q, unit(r_hat_c)>``, which deletes ``||m_hat_c||``.  This mission measures
whether the deleted quantity has downstream design value.  It is called the
model-implied directional predictability, or concentration, and is not called a
posterior probability or a confidence unless the diagnostics below support that.

Arms, at most four
------------------
  A0  CURRENT_CED          raw consensus target, score cos(u_q, r_hat_c)
  A1  UNIT_TARGET_COSINE   unit consensus target, score cos(u_q, m_hat_c)
  A2  UNIT_TARGET_DOT      the A1 predictor unchanged, score <u_q, m_hat_c>
  A3  RAW_TARGET_DOT       the A0 predictor unchanged, score <u_q, r_hat_c>

A1 and A2 share one basis, one ridge, one set of predictions, one pool and one query
set; the only difference is whether the prediction is normalized before the inner
product.  A0 and A3 share the same relation.  No arm fits a parameter the others do not.
``NORM_ONLY`` ranks by ``||m_hat_c||`` alone.  It is the D3 goal-blind control and is
never a designer.

What is new and what is inherited
---------------------------------
The only new object is the unit-normalized training target.  The basis rank, the
oversampling, the power iterations, the basis seed, the ridge penalty grid, the ridge
fold rule and the held-out gene-space cosine selection criterion are inherited from
``four_context_v1.predictors.fit`` unchanged, and A1 re-selects its penalty through that
procedure because its target changed.  The selection criterion is a per-row cosine, so
it is scale invariant and reads the same geometry for both targets.  The inherited
row-index fold defect recorded in ``open_vocab_generalization_v1.contract`` applies to
every arm identically.

Regimes
-------
``SEEN``   leave one context out.  The held context supplies no training row, so this
           is ``C_seen x G_unseen`` for candidates measured elsewhere in the atlas.
``MASKED`` the frozen C-016 protocol.  Five SHA-256 identity folds, each fold's
           identities removed from all three training contexts, basis and ridge rebuilt
           inside the reduced pool, every eligible candidate scored by the predictor that
           masked it.  This is ``C_unseen x G_unseen``.
Both regimes are evaluated on the identical C-016 eligible pool: identities measured in
at least one training context and carrying a legal static modality.  The fusion
standardizes each query row over the whole context pool and the evaluation restricts
afterwards, as every earlier phase does.

Which regime carries which rule
------------------------------
Both regimes enter every branch; see the outcome rules.  An earlier draft decided B, C and
D from SEEN alone and claimed that could not change the outcome, which is true only for A;
pre-run review (amendment 1) caught it before any number existed.

The decision statistic
----------------------
The paired per-query difference in MBRU under the frozen integrated fusion, pooled over
RPE1, HepG2 and Jurkat, summarized by its median, with a 95 percent interval from a
cluster bootstrap whose unit is the query identity: the gene symbol of the query, carrying
both batch-disjoint views and every context in which that gene is a query.  The mission
asks for the true query identity; 1333 pooled query rows come from 976 genes and a gene's
paired differences are correlated across contexts, so treating each (context, gene) as
its own cluster would be narrower than the mission's unit.  The (context, gene) interval
and the frozen view-level interval are reported beside it and decide nothing.  Within one
context the gene and the query coincide.

The mission names "downstream design improvement", which is the integrated designer.
Effect-only rankings are reported for every arm so that an improvement of the estimator
that does not reach the designer is visible, but they do not enter the decision.

The MASKED regime and the D4 condition
--------------------------------------
Cross-fitting mixes five fold predictors into one score row.  Within a query row the dot
product is ``<u_q, intercept_f> + <u_q, candidate part>``, so a candidate scored by fold f
carries that fold's intercept offset and that fold's shrinkage.  Whether five fold
predictors may share a row is exactly the question the frozen C-016 gate answers, so it is
reused unchanged on the five unit-target fold predictors: intercept directions at least
0.99 alike, one selected penalty, and a candidate-specific share ratio of at most 1.05.
(An earlier draft gated on the total norm, which the shared intercept dilutes; amendment 1.)

The D4 condition of Outcome A holds only if all three of these hold:
  * the cross-fitted fused A2 - A1, pooled with query-identity clusters, is POSITIVE;
  * the cross-fitted effect-only A2 - A1 pooled median is above zero.  The fused score
    still carries the frozen base, which was trained on K562 responses of about two
    thirds of the masked candidates, so only the effect-only contrast is exposure free;
  * if the frozen gate fails in any primary context, the per-fold fused A2 - A1 pooled
    median is also above zero.  The per-fold design scores each fold's candidates with
    that fold's predictor alone, so nothing is mixed, but its pools are about 250
    candidates and the frozen primary floor of 200 excludes every HepG2 fold; it is a
    robustness condition, never the estimate of record.

Diagnostics, and only these
---------------------------
Each is computed against the regime's own common response direction ``a``, the unit
vector of the unit-target predictor's intercept (for MASKED, of the fold's predictor).
Pre-run review measured, on training data alone, that the unit-target norm tracks the
predicted projection on ``a`` at Spearman 0.94, so a norm can look informative about
direction accuracy only because both share ``a``.  The candidate-specific accuracy
removes ``a`` from both vectors first (amendment 1).

D1  Spearman of ``||m_hat_c||`` with ``cos(m_hat_c, r_bar_c)``, the direction accuracy
    against the held context's measured consensus response, and with the
    candidate-specific accuracy ``cos(P m_hat_c, P r_bar_c)``, ``P = I - a a^T``.
D2  Spearman of ``||m_hat_c||`` with the measured amplitude ``||r_bar_c||`` and with the
    measured common alignment ``cos(r_bar_c, a)``; the partial Spearman of the norm with
    the candidate-specific accuracy given both; Top-20 overlaps of the goal-blind norm
    ordering with the measured-amplitude and raw predicted-norm orderings (descriptive).
D3  ``NORM_ONLY`` through effect-only ranking and the frozen fusion, and the median
    per-query Top-20 overlap of the A2 and A1 effect-only orderings with it.
D4  Everything above in the MASKED regime.  The MASKED diagnostics of record are
    per fold: each fold's predictor on that fold's own masked candidates, which it never
    saw, with no second predictor in the computation, summarized by the median over the
    five folds.  The cross-fitted versions are reported beside them.

The same quantities are reported for the raw predictor as the mechanism reference.  The
held context's measured responses enter only the utility and these diagnostics.  No
diagnostic quantity is ever an input to any score.

Outcome rules, applied mechanically by ``decide.decide``
-------------------------------------------------------
Each pooled contrast has a state: POSITIVE when the lower end of its query-identity
interval on the median is above zero, NEGATIVE when the upper end is below zero, and
NULL otherwise.  ``S21`` is the pooled SEEN fused A2 - A1 state.

  1. S21 POSITIVE and the amplitude rule trips in at least two primary contexts
                                                    -> D  AMPLITUDE_PRIOR_REAPPEARS
  2. S21 POSITIVE, the D4 condition holds, at least two primary contexts with a positive
     SEEN median, no primary context whose SEEN interval lies wholly below zero, and
     A2 - A0 (SEEN) not NEGATIVE                     -> A  DECISION_CONSISTENT_SCORING_SUPPORTED
  3. S21 POSITIVE but rule 2 fails                  -> NOT_ADOPTED, failed conditions named
  4. S21 not POSITIVE but the cross-fitted fused A2 - A1 POSITIVE
                                                    -> NOT_ADOPTED, the regimes disagree
  5. A2 - A1 NULL in both regimes, and A1 - A0 POSITIVE in SEEN and in the cross-fitted
     MASKED regime                                  -> B  UNIT_DIRECTION_TARGET_ONLY
  6. otherwise, including any NEGATIVE A2 - A1      -> C  PREDICTED_NORM_NOT_USEFUL

The condition "A2 - A0 not NEGATIVE" is an addition to the mission's four conditions for
A, recorded as such: promoting a scoring core that is worse than the current one is not
an upgrade whatever its mechanism.

The amplitude rule for one context trips when any of these holds:
  proxy        in the MASKED per-fold diagnostics of record, the norm's Spearman with the
               measured amplitude or with the measured common alignment is at least
               AMPLITUDE_SPEARMAN_STRONG, and its partial Spearman with the
               candidate-specific accuracy given both is below PARTIAL_SPEARMAN_WEAK;
  replication  in SEEN or in the cross-fitted MASKED regime, A2 - A1 fused is positive and
               NORM_ONLY fused minus BASE is at least PRIOR_REPLICATION_SHARE of it: a
               goal-blind ranking by the norm delivers at least half of the increment the
               norm is credited with (amendment 1; the draft compared against A2 - BASE,
               about ten times larger, which made this half unreachable);
  degeneration in SEEN or in the MASKED per-fold median, at least DEGENERATION_OVERLAP of
               A2's effect-only Top-20 is, for the median query, the goal-blind NORM_ONLY
               Top-20: the known amplitude-prior failure pattern of a query-independent list.
The proxy requires both halves because a stronger or more generic responder is measured
with less noise, so some correlation with amplitude is expected under the reliability
reading too; only the partial term separates the two readings.

Thresholds are conventions fixed here, not derivations: 0.5 is the conventional line
for a strong rank correlation, 0.1 for a negligible one, "most" of a gain is read as at
least half of it, and half of a budget spent on one query-independent list is read as
degeneration.

Next action, fixed for every outcome
------------------------------------
A, C and D route to RELIABILITY_WEIGHTED_EFFECT_DISTILLATION_V1 as the mission states.
B is left open by the mission ("decide later"), and it is fixed here to the same action
because B means that aligning the training target with the decision geometry already
paid, which is the premise the reliability-weighted design builds on.  NOT_ADOPTED routes
there too, because a gain that does not survive masking is a representation-quality
result and not a reason to stop.  Recorded plainly: under these rules no outcome of this
experiment routes to stopping candidate-effect method expansion, so this experiment
cannot by itself falsify the next direction.  The stop option is reached only if the A0
provenance gate fails, in which case nothing downstream is interpretable and the defect
is fixed before anything else.
"""
from __future__ import annotations

MISSION = "DECISION_CONSISTENT_EFFECT_DISTILLATION_V1"
SCHEMA = "VCDESIGN_DECISION_CONSISTENT_EFFECT_DISTILLATION_V1"

# --- arms --------------------------------------------------------------------
A0 = "A0_CURRENT_CED"
A1 = "A1_UNIT_TARGET_COSINE"
A2 = "A2_UNIT_TARGET_DOT"
A3 = "A3_RAW_TARGET_DOT"
ARMS = (A0, A1, A2, A3)
NORM_ONLY = "NORM_ONLY"            # D3 control, goal-blind, never a designer
BASE = "BASE"
ARM_TARGET = {A0: "raw", A1: "unit", A2: "unit", A3: "raw"}
ARM_SCORE = {A0: "cosine", A1: "cosine", A2: "dot", A3: "dot"}
MAX_SCORING_ARMS = 4

# --- scoring modes -------------------------------------------------------------
MODES = ("effect_only", "fused")
DECISION_MODE = "fused"
FUSION = "row_z(base) + 1.0 * row_z(effect), the frozen four_context_v1.evaluate.fuse"

# --- contexts and regimes ------------------------------------------------------
PRIMARY_CONTEXTS = ("RPE1", "HepG2", "Jurkat")
STRESS_CONTEXT = "K562"
STRESS_STRATUM = "UNSEEN_CANDIDATE"
STRESS_ARMS = (A0, A1, A2)
REGIMES = ("SEEN", "MASKED")
PRIMARY_REGIME = "SEEN"
MASKING_REGIME = "MASKED"
PER_FOLD_REGIME = "MASKED_PER_FOLD"
ELIGIBLE_POOL = (
    "identities measured in at least one training context and carrying a legal static "
    "modality, exactly the C-016 eligible set"
)

# --- inherited, never re-opened ----------------------------------------------------
PER_FOLD_PRIMARY_FLOOR = 200       # the frozen C-016 MIN_PRIMARY_POOL
NORM_HOMOGENEITY_TOLERANCE = 1.05  # descriptive total-norm ratio only; the gate is masking.homogeneity
HOMOGENEITY_GATE = "open_vocab_generalization_v1.masking.homogeneity on the five unit-target fold predictors"

# --- the provenance gate -------------------------------------------------------
# A0 must rebuild the cached C-016 effect cosines element by element and the recorded
# medians to the last printed digit.  The computation is deterministic, so anything
# above round-off is a wiring defect and stops the run.
PROVENANCE_COSINE_TOLERANCE = 1e-9
PROVENANCE_MEDIAN_TOLERANCE = 1e-9
RECORDED_SEEN_MEDIAN = {            # C-016 crossfit paired_vs_base EFFECT_SEEN MBRU median
    "RPE1": 0.0593859607119853,
    "HepG2": 0.05235267443958541,
    "Jurkat": 0.03710604577533211,
}
RECORDED_MASKED_MEDIAN = {          # C-016 crossfit paired_vs_base EFFECT_MASKED MBRU median
    "RPE1": 0.04819530524804283,
    "HepG2": 0.05680124561174221,
    "Jurkat": 0.03009521202312604,
}
RECORDED_STRESS_MEDIAN = -0.010281249762853273   # C-014 K562 UNSEEN_CANDIDATE EFFECT - BASE
RECORDED_ELIGIBLE = {"RPE1": 1273, "HepG2": 764, "Jurkat": 1450}

# --- metrics -------------------------------------------------------------------------
HEADLINE_METRICS = ("MBRU", "Mean@10", "Mean@20", "Mean@50",
                    "HighValueCount@10", "HighValueCount@20", "HighValueCount@50")
COMPANION_METRICS = ("Best@10", "Best@20", "Best@50")
BEST_DEFINITION = "the largest graded utility among the top B of the same legal ordering"

# --- inference -------------------------------------------------------------------------
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260916          # the frozen programme seed
CLUSTER_UNIT = "query gene identity across contexts; both views and every context of one gene drawn together"
COMPANION_CLUSTER_UNIT = "(context, query) rows; reported beside the decision interval, never deciding"
DIAGNOSTIC_BOOTSTRAP_REPLICATES = 2000
DIAGNOSTIC_BOOTSTRAP_SEED = 20260918

# --- contrasts ---------------------------------------------------------------------------
CONTRASTS = {
    "A1_minus_A0": (A1, A0),       # does learning the decision direction beat raw-then-normalize
    "A2_minus_A1": (A2, A1),       # the primary mechanistic contrast: is the deleted norm worth anything
    "A3_minus_A0": (A3, A0),       # the raw-norm mechanism control
    "A2_minus_A0": (A2, A0),       # the proposed core against the current core
    "A0_minus_BASE": (A0, BASE),
    "A1_minus_BASE": (A1, BASE),
    "A2_minus_BASE": (A2, BASE),
    "A3_minus_BASE": (A3, BASE),
    "NORM_ONLY_minus_BASE": (NORM_ONLY, BASE),
}

# --- diagnostics -------------------------------------------------------------------------
DIRECTION_ACCURACY = "cos(predicted effect, held-context measured consensus response)"
AMPLITUDE = "||held-context measured consensus response||"
OVERLAP_BUDGET = 20
AMPLITUDE_SPEARMAN_STRONG = 0.5
PARTIAL_SPEARMAN_WEAK = 0.1
PRIOR_REPLICATION_SHARE = 0.5
DEGENERATION_OVERLAP = 0.5
CONTEXT_MAJORITY = 2               # of three primary contexts

# --- states and outcomes -----------------------------------------------------------------
POSITIVE, NULL, NEGATIVE = "POSITIVE", "NULL", "NEGATIVE"
OUTCOME_A = "DECISION_CONSISTENT_SCORING_SUPPORTED"
OUTCOME_B = "UNIT_DIRECTION_TARGET_ONLY"
OUTCOME_C = "PREDICTED_NORM_NOT_USEFUL"
OUTCOME_D = "AMPLITUDE_PRIOR_REAPPEARS"
OUTCOME_NOT_ADOPTED = "NOT_ADOPTED_OUTSIDE_A_TO_D"
NEXT_RWED = "RELIABILITY_WEIGHTED_EFFECT_DISTILLATION_V1"
NEXT_STOP = "STOP_CANDIDATE_EFFECT_METHOD_EXPANSION"
AMENDMENT_1 = (
    "2026-09-18, before any number of this mission existed: prompted by pre-run code review. "
    "Replication compares against A2 - A1; both regimes enter every branch; clusters are query "
    "genes; the D4 gate is the frozen C-016 homogeneity gate; D4 also needs the exposure-free "
    "effect-only sign; diagnostics remove the common response direction; the amplitude proxy "
    "reads per-fold masked diagnostics; a degeneration trip was added"
)
NEXT_ACTION = {OUTCOME_A: NEXT_RWED, OUTCOME_B: NEXT_RWED, OUTCOME_C: NEXT_RWED,
               OUTCOME_D: NEXT_RWED, OUTCOME_NOT_ADOPTED: NEXT_RWED}

# --- hard stops ------------------------------------------------------------------------------
FORBIDDEN_IN_THIS_MISSION = (
    "gamma sweep", "beta-versus-pool-size sweep", "query-adaptive gate",
    "per-candidate learned reliability network", "MLP confidence head",
    "new nonlinear predictor", "new ranking objective", "per-direction reliability penalty",
    "new candidate embedding", "support gate", "beta calibration", "additional context",
    "reliability-weighted shrinkage", "base retraining",
)
NOT_AUTHORIZED_BY_SUCCESS = (
    "an Outcome A does not authorize any item in FORBIDDEN_IN_THIS_MISSION"
)
