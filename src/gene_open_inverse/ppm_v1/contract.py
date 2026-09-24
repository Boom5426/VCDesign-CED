"""PERTURBATION_PROGRAM_MEMORY_V1, frozen before the first number.

The protocol is ``docs/PERTURBATION_PROGRAM_MEMORY_V1_PROTOCOL.md``.  Every architectural
number, loss weight, optimizer setting, selection rule and decision threshold lives here, and
nothing in the other modules restates one.
"""
from __future__ import annotations

MISSION = "PERTURBATION_PROGRAM_MEMORY_V1"
SCHEMA = "VCDESIGN_PERTURBATION_PROGRAM_MEMORY_V1"

# --- arms ------------------------------------------------------------------------------
RIDGE = "RIDGE_UNIT"                 # C-018 A1, the reference
FULL = "PPM_FULL"
PARAMETRIC = "PPM_PARAMETRIC"
MEMORY = "PPM_MEMORY"
# Deviation 2 (the single Section 11 fix): the same arms predicting a correction to A1.
FULL_ANCHORED = "PPM_FULL_ANCHORED"
PARAMETRIC_ANCHORED = "PPM_PARAMETRIC_ANCHORED"
MEMORY_ANCHORED = "PPM_MEMORY_ANCHORED"
ANCHORED = (FULL_ANCHORED, PARAMETRIC_ANCHORED, MEMORY_ANCHORED)
PPM_ARMS = (FULL, PARAMETRIC, MEMORY) + ANCHORED
FULL_ARMS = (FULL, FULL_ANCHORED)
ABLATION_OF = {FULL: (PARAMETRIC, MEMORY), FULL_ANCHORED: (PARAMETRIC_ANCHORED, MEMORY_ANCHORED)}
BASE = "BASE"
USES_ROUTER = {FULL: True, PARAMETRIC: True, MEMORY: False,
               FULL_ANCHORED: True, PARAMETRIC_ANCHORED: True, MEMORY_ANCHORED: False}
USES_MEMORY = {FULL: True, PARAMETRIC: False, MEMORY: True,
               FULL_ANCHORED: True, PARAMETRIC_ANCHORED: False, MEMORY_ANCHORED: True}
ANCHOR_OOF_NAMESPACE = "VCDESIGN_PPM_V1_ANCHOR_OOF"
ANCHOR_OOF_FOLDS = 5

PRIMARY_CONTEXTS = ("RPE1", "HepG2", "Jurkat")
STRESS_CONTEXT = "K562"
REGIMES = ("SEEN", "MASKED")
MODES = ("effect_only", "fused")
SEEN_POOL = "SEEN"

# --- architecture (one configuration, no search) ----------------------------------------
MODALITY_WIDTHS = (512, 1024)        # STRING, MAPKG; then one presence flag per modality
HIDDEN = 256
ENCODER_BLOCKS = 2
FFN_WIDTH = 512
DROPOUT = 0.1
# Deviation 1 (pre-formal, from the RPE1 smoke's training-context inner validation only).
# Dropout at rate p on standardized inputs is, for a linear map under squared loss, ridge with
# penalty (p / (1 - p)) * diag(X^T X) = (p / (1 - p)) * n_rows (Wager, Wang and Liang 2013).
# Matching A1's selected penalty 1e4 at n_rows of about 8e3 gives p = 1e4 / (1e4 + 8e3), about
# 0.55, rounded to 0.5.  Derived from the frozen ridge result, not searched.
INPUT_DROPOUT = 0.5
PROGRAMS = 256
ACTIVE_PROGRAMS = 16
EXPERTS = 4
ACTIVE_EXPERTS = 2
NEIGHBOURS = 32
HEADS = 4
GATE_WIDTH = 64

# --- objective -----------------------------------------------------------------------------
LOSS_WEIGHTS = {"dir": 1.0, "tok": 1.0, "view": 0.25, "metric": 1.0, "div": 1.0, "balance": 0.01}

# --- optimization and selection ---------------------------------------------------------------
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
GRADIENT_CLIP = 1.0
BATCH_ROWS = 256
MAX_EPOCHS = 200
PATIENCE = 25
SEED = 20260918
REPLICATE_SEED = 20260919
VALIDATION_NAMESPACE = "VCDESIGN_PPM_V1_INNER_VALIDATION"
VALIDATION_MODULUS = 10              # identities with hash mod 10 == 0 are inner validation
ZERO_NORM = 1e-12

# --- provenance --------------------------------------------------------------------------------
C018_RUN = "decision_consistent_effect_v1_runs/20260918T131418Z__inverse-model__dced_v1"
A1_EFFECT_TOLERANCE = 1e-10

# --- decision -----------------------------------------------------------------------------------
MIN_GAIN = 0.010
CONTEXT_MAJORITY = 2
REVERSAL_FLOOR = -0.005
MASKED_TO_SEEN_RATIO = 0.5
PER_FOLD_PRIMARY_FLOOR = 200
HOMOGENEITY_FLOOR = 0.99
SHARE_RATIO_TOLERANCE = 1.05
POSITIVE, NULL, NEGATIVE = "POSITIVE", "NULL", "NEGATIVE"
SUPPORTED = "PPM_V1_SUPPORTED"
PARTIAL = "PPM_V1_PARTIALLY_SUPPORTED"
NOT_SUPPORTED = "PPM_V1_NOT_SUPPORTED"

FORBIDDEN = (
    "candidate identity embedding", "query-conditioned candidate scoring", "cosine replacement",
    "fusion change or beta search", "gate, norm or confidence entering a score",
    "downstream MBRU surrogate, budget or ranking loss", "response-amplitude loss",
    "architecture, head-count, K or loss sweep", "reading G_check",
)
