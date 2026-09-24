"""RELIABILITY_WEIGHTED_EFFECT_DISTILLATION_V1, frozen before the first number.

The single question
-------------------
After unit-direction effect distillation (C-018 A1) gained a small, stable amount, can the
cross-view reproducibility of perturbation measurements be used to suppress irreproducible
candidate-specific effect directions and so improve fully open-vocabulary design further?
This is not reliability scoring.  No candidate-specific reliability, norm, confidence or
support enters any score (C-017, C-018).

The one estimator, and why it is this one
-----------------------------------------
Start from A1 exactly: the frozen four-context recipe on unit consensus targets, rank-256
basis ``B``, one penalty chosen by the frozen cross validation.  Its targets in the basis
are ``Y = U B`` and its intercept is ``mean(Y)``.

1. Coordinates.  Rotate A1's own coefficient space into
   ``Q = [a, W]``: ``a`` is the unit vector of A1's intercept (the C-018 common-direction
   definition, so no new freedom), and ``W`` holds the principal axes of the targets with
   ``a`` removed, from an exact SVD of ``Y (I - a a^T)``.  ``Q`` is orthonormal and spans
   exactly A1's basis, so the rank budget is A1's: one common axis and 255 candidate-specific
   axes.  The centred residual has no component along ``a``, and ``a`` is proportional to
   the target mean, so centring does not change ``W``.

2. Reliability, training contexts only.  Each training row's two batch-disjoint views are
   unit normalized, projected on ``B`` and rotated by ``Q``.  Within each training context
   each view's coordinate is centred, so a context-level offset shared by both views cannot
   pass as reproducibility.  The per-view reliability of axis ``k`` is the Pearson
   correlation of the two views' coordinates across training rows, ``rho_k``, clipped to
   ``[RHO_FLOOR, 1]``.

3. Measurement model.  View coordinates are ``s + e_A`` and ``s + e_B`` with independent
   noise, so ``rho_k = var(s) / (var(s) + var(e))``.  The consensus target averages two
   parallel measurements; its reliability is Spearman-Brown, ``R_k = 2 rho_k / (1 + rho_k)``.

4. Empirical-Bayes penalty.  Give axis ``k`` a Gaussian prior on its regression weights
   whose variance is proportional to the reproducible variance ``gamma var(s_k)`` the
   features can explain, with the same explained fraction ``gamma`` on every axis.  The
   residual variance of the consensus target is ``(1 - gamma) var(s_k) + var(noise_k)``.
   The posterior-mean ridge penalty is noise over prior, which is proportional to
   ``(1 - gamma R_k) / (gamma R_k)``.  Static features explain little of the reproducible
   response here (A1's held-out gene-space cosine is 0.159 to 0.170), so to first order in
   ``gamma`` the penalty is proportional to ``1 / R_k``.  That first-order form is frozen; the
   exact form would add ``gamma`` as a free parameter and is not used.

5. The common axis.  A shared intercept direction is reproducible because every candidate
   carries it, not because any candidate's specific effect is.  Letting its reliability
   lower its penalty is exactly the generic-responder failure C-018 recorded.  So the common
   axis is taken out of the reliability map and given the reference multiplier 1, and the
   residual multipliers are normalized to geometric mean 1:
       m_common = 1,   m_k = Rbar / R_k,   Rbar = geometric mean of R_k over residual axes.
   Reliability can redistribute shrinkage among candidate-specific axes but cannot move the
   common axis relative to the average one.  Penalty of axis k: ``lambda * m_k``.

6. The global ``lambda`` is re-selected by the frozen procedure unchanged: the frozen grid,
   the frozen five row-index folds, and the held-out gene-space cosine against the unit
   targets.  With every ``R_k = 1`` the estimator is A1 exactly; the tests assert it.

Monotone and parameter free: ``m_k`` falls as ``rho_k`` rises; ``rho_k -> 1`` keeps an axis at
the average level or better, ``rho_k -> 0`` sends its penalty toward ``Rbar / R_floor`` and
its prediction to the intercept.  ``RHO_FLOOR`` only keeps the geometric mean finite; an
axis at the floor carries ``1 / R_floor``, about 500 times, the penalty of a perfectly
reproducible axis, which is indistinguishable from removing it.

Score and fusion are unchanged: ``cos(d_q, effect_hat_c)`` and the frozen
``row_z(base) + 1.0 * row_z(effect)``.

Regimes, arms and pools
-----------------------
Arms: A0 (provenance only, graded from C-018's saved raw predictions), A1 (recomputed and
required to equal C-018's saved unit predictions), RWED.  Primary contexts RPE1, HepG2,
Jurkat on the C-016/C-018 eligible pools.  MASKED (the frozen C-016 cross-fit) is the
decision regime; SEEN is a companion.  K562 6255 unseen candidates is a secondary stress
test and does not enter the decision.

Inference: paired per-query MBRU, pooled over the primary contexts, median, 95 percent
cluster bootstrap over query genes (every view and every context of a gene together).
Candidate-level diagnostics resample candidate genes the same way.

The decision, applied by ``decide.decide``
-----------------------------------------
RWED is adopted only if all seven hold:
  C1  pooled MASKED fused RWED - A1 is POSITIVE (lower interval end above zero);
  C2  its median is at least MIN_GAIN;
  C3  at least two primary contexts have a positive MASKED fused median;
  C4  no primary context's MASKED fused interval lies wholly below zero;
  C5  the pooled MASKED per-candidate change in candidate-specific reconstruction,
      cos(P m_hat, P r_bar) with the common axis removed, is POSITIVE;
  C6  the design gain survives removing the common axis: under the diagnostic score
      cos(d_q, P m_hat), applied identically to both estimators, the pooled MASKED fused
      RWED - A1 median is above zero;
  C7  the gain holds for fully masked candidates without the base's exposure: the pooled
      MASKED effect-only RWED - A1 median is above zero, and if the frozen C-016 cross-fit
      gate fails for either estimator in any primary context, the per-fold fused median
      (folds of at least 200 candidates) is also above zero.
Otherwise: RWED_NOT_WORTH_ADDED_COMPLEXITY, and every failed condition is named.
"""
from __future__ import annotations

MISSION = "RELIABILITY_WEIGHTED_EFFECT_DISTILLATION_V1"
SCHEMA = "VCDESIGN_RELIABILITY_WEIGHTED_EFFECT_DISTILLATION_V1"

A0 = "A0_CURRENT_CED"
A1 = "A1_UNIT_TARGET_ISOTROPIC"
RWED = "RWED_UNIT_TARGET_RELIABILITY_WEIGHTED"
BASE = "BASE"
ARMS = (A0, A1, RWED)
DIAGNOSTIC_SCORES = ("A1_COMMON_REMOVED", "RWED_COMMON_REMOVED")   # C6 only, never designers

PRIMARY_CONTEXTS = ("RPE1", "HepG2", "Jurkat")
STRESS_CONTEXT = "K562"
REGIMES = ("SEEN", "MASKED")
DECISION_REGIME = "MASKED"
MODES = ("effect_only", "fused")
DECISION_MODE = "fused"

# --- the estimator ---------------------------------------------------------------
RHO_FLOOR = 1e-3
RELIABILITY = "within-context-centred Pearson correlation of the two views' rotated coordinates"
CONSENSUS_RELIABILITY = "Spearman-Brown, 2 rho / (1 + rho)"
PENALTY_MAP = "lambda * m_k, m_common = 1, m_k = geometric_mean(R) / R_k over residual axes"
FIRST_ORDER_NOTE = "posterior penalty (1 - gamma R)/(gamma R) taken to first order in gamma"

# --- provenance --------------------------------------------------------------------
A1_EFFECT_TOLERANCE = 1e-10
MEDIAN_TOLERANCE = 1e-12
RECORDED = {   # C-018 per-context fused medians, both regimes
    "RPE1": {"SEEN": {"A1_minus_A0": 0.004490715549557392, "A0_minus_BASE": 0.0593859607119853},
             "MASKED": {"A1_minus_A0": 0.002226085555694487, "A0_minus_BASE": 0.04819530524804283}},
    "HepG2": {"SEEN": {"A1_minus_A0": 0.0034098665353377577, "A0_minus_BASE": 0.05235267443958541},
              "MASKED": {"A1_minus_A0": 0.0051251456176384935, "A0_minus_BASE": 0.05680124561174221}},
    "Jurkat": {"SEEN": {"A1_minus_A0": 0.005300288541053744, "A0_minus_BASE": 0.03710604577533211},
               "MASKED": {"A1_minus_A0": 0.004898265477697339, "A0_minus_BASE": 0.03009521202312604}},
}
RECORDED_STRESS = {"A1_minus_A0": -0.010067180161018596, "A0_minus_BASE": -0.010281249762853273}
C018_RUN = "decision_consistent_effect_v1_runs/20260918T131418Z__inverse-model__dced_v1"

# --- decision -------------------------------------------------------------------------
MIN_GAIN = 0.005
CONTEXT_MAJORITY = 2
PER_FOLD_PRIMARY_FLOOR = 200
POSITIVE, NULL, NEGATIVE = "POSITIVE", "NULL", "NEGATIVE"
OUTCOME_ADOPT = "RWED_ADOPTED"
OUTCOME_STOP = "RWED_NOT_WORTH_ADDED_COMPLEXITY"
NEXT_ACTION = {OUTCOME_ADOPT: "ADOPT_RWE_DISTILLATION_IN_VCDESIGN_V2",
               OUTCOME_STOP: "STOP_CLOSED_FORM_EFFECT_REFINEMENT_AND_WRITE_PAPER"}
STOP_CLOSES = (
    "alternative reliability formula", "nonlinear shrinkage", "learned reliability network",
    "reliability gate", "context-specific reliability", "candidate-specific confidence scoring",
    "additional ridge variants",
)
FORBIDDEN = (
    "candidate reliability, norm, confidence or support multiplied into or added to a score",
    "dot-product scoring", "gamma", "learned confidence head", "support-aware gate",
    "a second reliability formula", "fusion change", "base retraining",
)
