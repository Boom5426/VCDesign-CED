"""Frozen contract of EXTERNAL_BASELINE_AND_EVALUATION_CLOSURE_V1.

Machine-readable twin of the external-baseline and evaluation-closure protocol.
Nothing here is tuned on an outcome.  VCDesign itself is frozen: ``RIDGE_UNIT`` (C-018 A1) with
the frozen fusion, read from C-018's saved effects and never refitted.
"""
from __future__ import annotations

MISSION = "EXTERNAL_BASELINE_AND_EVALUATION_CLOSURE_V1"
PRIMARY_CONTEXTS = ("RPE1", "HepG2", "Jurkat")
TRAINING_CONTEXTS = ("K562", "RPE1", "HepG2", "Jurkat")
REGIMES = ("SEEN", "MASKED")
SEEN_POOL = "SEEN"
MASK_FOLDS = 5
READ_ANCHOR_G_CHECK = False

# --- frozen inputs (paths are CLI defaults in run_all.sh, recorded in every output) ---------
C018_RUN = "inputs/runs/decision_consistent_effect_v1"
EVIDENCE_RUN = "inputs/runs/evidence_v1"
FOUR_CONTEXT_RUN = "inputs/runs/four_context_v1"

# --- arms -----------------------------------------------------------------------------------
RANDOM = "RANDOM"
PRIOR = "GOAL_BLIND_PRIOR"                    # repaired predicted-effect magnitude prior, ||raw CED||
BASE = "BASE"
CELLNAVI = "CELLNAVI_DESIGN_COMPATIBLE"
GEARS = "GEARS_FORWARD_ENUMERATE"
INTERNAL_FTM = "INTERNAL_FORWARD_THEN_MATCH"  # RIDGE_UNIT effect-only cosine
VCDESIGN = "VCDESIGN_BASE_PLUS_RIDGE_UNIT"
ORACLE = "MEASURED_RESPONSE_ORACLE"           # analysis only
RIDGE_K562 = "RIDGE_UNIT_K562_ATLAS"          # information-matched companion of GEARS
CELLNAVI_NATIVE = "CELLNAVI_NATIVE_SCALE"     # Deviation 5: classes = the held context's C-016 SEEN stratum
PROFILED = "PROFILED_SIGNATURE_RETRIEVAL"     # SEEN_ONLY_LIBRARY_RETRIEVAL companion
MAIN_ROWS = (RANDOM, PRIOR, BASE, CELLNAVI, CELLNAVI_NATIVE, GEARS, INTERNAL_FTM, VCDESIGN, ORACLE)
TAXONOMY = {
    RANDOM: "A. random",
    PRIOR: "B. goal-blind candidate prior",
    BASE: "C. direct inverse (and E. VCDesign base)",
    CELLNAVI: "C. direct inverse, published",
    GEARS: "D. forward-enumerate-retrieve, published",
    INTERNAL_FTM: "D. forward-then-match, internal reference",
    VCDESIGN: "E. VCDesign",
    ORACLE: "F. analysis-only oracle",
    RIDGE_K562: "companion: same K562 atlas as GEARS, RIDGE_UNIT recipe",
    CELLNAVI_NATIVE: "C. direct inverse, published, at its native class scale (Deviation 5)",
    PROFILED: "companion: CMap-style library retrieval on training-context signatures",
}
LEGALITY = {
    RANDOM: "no information",
    PRIOR: "strict legal, goal blind",
    BASE: "strict legal",
    CELLNAVI: "MEASURED-CANDIDATE inverse baseline: a class needs the candidate's measured cells in a "
              "training context; SEEN only",
    GEARS: "strict legal in both regimes: no candidate response of a masked identity enters its fit",
    INTERNAL_FTM: "strict legal",
    VCDESIGN: "strict legal",
    ORACLE: "DIAGNOSTIC_ORACLE_NOT_DEPLOYABLE: reads the candidate's measured held-context response",
    RIDGE_K562: "strict legal",
    CELLNAVI_NATIVE: "MEASURED-CANDIDATE inverse baseline, SEEN only; same legality as CELLNAVI",
    PROFILED: "SEEN_ONLY_LIBRARY_RETRIEVAL: reads the candidate's measured training-context responses",
}
RANDOM_SEED = 20260919
UNSCORED = "a candidate a method cannot represent gets a zero effect, which the frozen effect cosine scores 0"

# --- metrics (contract V1) --------------------------------------------------------------------
BUDGETS = (10, 20, 50)
PRIMARY = "BU@20"
COMPANIONS = ("BU@10", "BU@50", "MU@20", "HvHit@20")
MIN_PHR_POOL = 200   # contract V1 MIN_PRIMARY_POOL, four times the largest budget

# --- PHR --------------------------------------------------------------------------------------
PHR_RULE = ("four-way sub-pool: candidates in the C-016 stratum and queries that meet MIN_VIEW_CELLS in "
            "all four batch quarters; a view-v row's action is chosen on side v's quarter pair, the "
            "side that also builds its query, and realized J is read on side 1-v's pair")
PHR_RECONSTRUCTION_TOLERANCE = 1e-5   # float32 storage of responses of magnitude <= ~4
PHR_UNAVAILABLE = "PHR_UNAVAILABLE_DUE_TO_PACK_LIMITATION"

# --- GEARS --------------------------------------------------------------------------------------
GEARS_UPSTREAM = "https://github.com/snap-stanford/GEARS"
GEARS_PACKAGE = "cell-gears==0.1.2"
GEARS_ENV = "pertbench"
GEARS_TRAINING_CONTEXT = "K562"       # GEARS pairs a perturbed cell with a random control of its own dataset
GEARS_CELLS_PER_IDENTITY = 16        # Deviation 1 (compute, before any GEARS prediction): was 32
GEARS_MASK_RULE = ("fold union: the MASKED fit for C-016 fold f excludes, from K562 G_fit, every identity "
                   "that is C-016 eligible and in fold f in any of the three held pools; one fit serves all "
                   "three held contexts, and each held pool's masked set is a subset of it")
GEARS_CONTROL_CELLS = 4096
GEARS_CELL_NAMESPACE = "VCDESIGN_EBC_V1_GEARS_CELLS"
GEARS_VAL_NAMESPACE = "VCDESIGN_EBC_V1_GEARS_VAL"
GEARS_VAL_MODULUS = 10               # bucket 0 is GEARS's own validation split
GEARS_HYPER = {"hidden_size": 64, "epochs": 20, "lr": 1e-3, "weight_decay": 5e-4,
               "batch_size": 32, "test_batch_size": 128}
GEARS_REPRO_VERDICT = "GEARS_OFFICIAL_REPRODUCTION_FAILED"   # composition 9/43/19/36 vs 9/52/18/37; headline 0.2456 vs 0.2540
GEARS_PLACEMENT = ("labelled companion outside the main table (Deviation 4); DEGENERATE: every fit predicts one "
                   "perturbation-independent shift (GEARS_COLLAPSE_DIAGNOSTIC.json), so its ranking is the tie rule")
GEARS_PREDICTION_CONTROLS = 8
GEARS_INVARIANCE_TOLERANCE = 1e-4
GEARS_SEED = 20260919
GEARS_REPRO = {"source": "demo/model_tutorial.ipynb (snap-stanford/GEARS master), printed output",
               "dataset": "norman", "split": "simulation", "split_seed": 1, "batch_size": 32,
               "test_batch_size": 128, "hidden_size": 64, "epochs": 1, "lr": 1e-3,
               "target_test_top20_de_mse": 0.2540, "tolerance": 0.05,
               "target_composition": {"combo_seen0": 9, "combo_seen1": 52, "combo_seen2": 18, "unseen_single": 37}}

# --- CellNavi -------------------------------------------------------------------------------------
CELLNAVI_UPSTREAM = "https://github.com/DLS5-Omics/CellNavi"
CELLNAVI_COMMIT = "ea1660fc3057c63ecd849d506f24241bdc4b3f33"
CELLNAVI_ENV = "cellnavi"
CELLNAVI_PRETRAIN_MD5 = "cfc0158f70b52984e2cb8c22803c8eb7"
CELLNAVI_TUTORIAL_CKPT_MD5 = "277368e4376bb2b4eae85adcfd491644"
CELLNAVI_REPRO_TARGET = {"accuracy": 0.510, "weighted_f1": 0.498, "cells": 3158, "classes": 70}
CELLNAVI_REPRO_TOLERANCE = 0.005      # three-decimal rounding of the published tutorial output
CELLNAVI_RECIPE = {"global_batch_size": 128, "local_batch_size": 1, "mixed_precision": "true",
                   "nr_step": 1050, "warmup_step": 500, "lr": 0.001,
                   "chk_time_interval": 3600, "chk_step_interval": 100}
CELLNAVI_EVAL_STEP = 1000             # the checkpoint the official tutorial evaluates
CELLNAVI_CELLS_PER_CLASS = 32         # per (training context, identity), deterministic
CELLNAVI_QUERY_CELLS = 16             # per (query, view), deterministic
CELLNAVI_CELL_NAMESPACE = "VCDESIGN_EBC_V1_CELLNAVI_CELLS"
CELLNAVI_SCORE = "mean over the query-view cells of log_softmax(logits)[class of c]"
CELLNAVI_CLASS_RULES = {"all_training": "every identity the held context's RIDGE_UNIT SEEN fit may train on",
                        "stratum": "the held context's C-016 SEEN stratum: the candidate library the task ranks"}

# --- sources ------------------------------------------------------------------------------------
SOURCES = {
    "K562": {"path": "inputs/public/ReplogleWeissman2022_K562_gwps.h5ad",
             "perturbation": "perturbation", "control": "control", "batch": "batch", "symbol": None},
    "RPE1": {"path": "inputs/public/ReplogleWeissman2022_rpe1.h5ad",
             "perturbation": "perturbation", "control": "control", "batch": "batch", "symbol": None},
    "HepG2": {"path": "inputs/public/GSE264667_hepg2_raw_singlecell_01.h5ad",
              "perturbation": "gene", "control": "non-targeting", "batch": "gem_group", "symbol": "gene_name"},
    "Jurkat": {"path": "inputs/public/GSE264667_jurkat_raw_singlecell_01.h5ad",
               "perturbation": "gene", "control": "non-targeting", "batch": "gem_group", "symbol": "gene_name"},
}
LIBRARY_TARGET = 1.0e4

# --- PDGrapher ----------------------------------------------------------------------------------
PDGRAPHER_UPSTREAM = "https://github.com/mims-harvard/PDGrapher"
