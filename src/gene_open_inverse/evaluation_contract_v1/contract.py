"""The frozen VCDesign evaluation contract V1.  See docs/VCDesign_EVALUATION_CONTRACT_V1.md."""
from __future__ import annotations

VERSION = "VCDESIGN_EVALUATION_CONTRACT_V1"
FROZEN = "2026-09-18"

BUDGETS = (10, 20, 50)
PRIMARY_BUDGET = 20
PRIMARY = f"BU@{PRIMARY_BUDGET}"
FAMILIES = ("BU", "MU", "HvHit", "PHR", "QWR")
ROLES = {"BU": "primary", "MU": "secondary", "HvHit": "experimental efficiency",
         "PHR": "validity guardrail", "QWR": "consistency"}
RETIRED_FROM_PAPER = ("MBRU", "Best@B", "J RU@B")
HIGH_VALUE_QUANTILE = 0.95

STATISTIC = "median of paired per-row differences, mean reported beside it"
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260916
CLUSTER = "query gene"

ANCHOR = "K562 locked G_check: sealed, reported on its pre-registered MBRU only"
PRIMARY_CONTEXTS = ("RPE1", "HepG2", "Jurkat")
DECISION_REGIME = "MASKED"
STRESS = "K562 6255 unseen-candidate pool: limitation only, never decisive"

MIN_WORTHWHILE_GAIN = 0.005
CONTEXT_MAJORITY = 2
QWR_GATE = 0.5
POSITIVE, NULL, NEGATIVE = "POSITIVE", "NULL", "NEGATIVE"
ADOPT = "ADOPT_NEW_METHOD"
KEEP = "KEEP_INCUMBENT"
