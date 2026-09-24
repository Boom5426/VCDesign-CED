"""Frozen specification of the evidence experiments.  Every spec is the frozen A1 recipe
(``four_context_v1.predictors.fit`` on unit consensus targets) with exactly one input changed:
the training atlas, a row-count match, the candidate modality, or the basis rank."""
from __future__ import annotations

import itertools

MISSION = "SINGLE_GENE_ESTIMATOR_EVIDENCE_V1"
PRIMARY_CONTEXTS = ("RPE1", "HepG2", "Jurkat")
ALL_CONTEXTS = ("K562", "RPE1", "HepG2", "Jurkat")
FORMAL_RANK = 256
COMPANION_RANKS = (64, 128, 512)
MODALITIES = ("STRING", "MAPKG")
ROW_NAMESPACE = "VCDESIGN_EVIDENCE_V1_ROWS"
C018_RUN = "inputs/runs/decision_consistent_effect_v1"
FULL = "full"


def specs(held: str) -> list[dict]:
    """E1/E5 atlases (all subsets of the three training contexts, natural and row-matched),
    E3 modality ablations and E4 companion ranks.  ``full`` natural at rank 256 is A1 itself."""
    training = tuple(c for c in ALL_CONTEXTS if c != held)
    out = []
    for size in (1, 2, 3):
        for subset in itertools.combinations(training, size):
            name = FULL if size == 3 else "+".join(subset)
            out.append({"name": name, "contexts": subset, "matched": False, "modality": "both", "rank": FORMAL_RANK,
                        "experiment": "atlas"})
            out.append({"name": f"{name}@matched", "contexts": subset, "matched": True, "modality": "both",
                        "rank": FORMAL_RANK, "experiment": "atlas_matched"})
    for modality in MODALITIES:
        out.append({"name": f"{modality}_only", "contexts": training, "matched": False, "modality": modality,
                    "rank": FORMAL_RANK, "experiment": "modality"})
    for rank in COMPANION_RANKS:
        out.append({"name": f"rank{rank}", "contexts": training, "matched": False, "modality": "both", "rank": rank,
                    "experiment": "rank"})
    return out
