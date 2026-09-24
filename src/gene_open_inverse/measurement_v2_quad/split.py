"""Deterministic quarter assignment, cell incidence, and eligibility.

Every constant is inherited from the frozen M3 lock rather than re-chosen, so
the two-way behaviour of this module must reproduce that lock exactly.  The one
new constant is the refinement namespace, which supplies a second hash bit that
is independent of the bit the frozen split already used.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


# Inherited verbatim from the frozen measurement_v2_m3 lock.  Not re-chosen here.
VIEW_SEED = 20260914
VIEW_NAMESPACE = "REPLOGLE_BATCH_DISJOINT_V1"
CONTROL = "control"
MIN_VIEW_CELLS = 25
LOCKED_A_BATCHES, LOCKED_B_BATCHES = 131, 136
LOCKED_ELIGIBLE_IDENTITIES = 9205
LOCKED_FEATURES = 8248

# The only new constant.  A separate namespace keeps the refinement bit
# independent of the bit the frozen split consumed for the same batch name.
QUAD_NAMESPACE = "REPLOGLE_BATCH_QUAD_REFINEMENT_V1"
QUARTERS = (0, 1, 2, 3)


def _hash_bit(namespace: str, batch: str) -> int:
    token = f"{namespace}|{VIEW_SEED}|{batch}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big") & 1


def view_side(batch: str) -> int:
    """The frozen A/B bit, reimplemented from the recorded rule."""
    return _hash_bit(VIEW_NAMESPACE, batch)


def refinement_bit(batch: str) -> int:
    """The new within-side bit."""
    return _hash_bit(QUAD_NAMESPACE, batch)


def quarter_of(batch: str) -> int:
    """``Q0, Q1`` refine side A and ``Q2, Q3`` refine side B."""
    return view_side(batch) * 2 + refinement_bit(batch)


def batch_quarters(batches: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the per-batch quarter and the frozen per-batch side."""
    sides = np.asarray([view_side(str(name)) for name in batches], dtype=np.int8)
    quarters = np.asarray([quarter_of(str(name)) for name in batches], dtype=np.int8)
    if not np.array_equal(quarters // 2, sides):
        raise RuntimeError("quarter assignment is not a refinement of the frozen A/B split")
    return quarters, sides


@dataclass(frozen=True)
class SplitCounts:
    """Cell and batch occupancy of every identity in every quarter."""

    identities: np.ndarray
    batches: np.ndarray
    quarters: np.ndarray
    sides: np.ndarray
    cells: np.ndarray
    occupied_batches: np.ndarray
    control_cells_per_batch: np.ndarray
    incidence: np.ndarray

    @property
    def cells_by_side(self) -> np.ndarray:
        return np.stack((self.cells[:, 0] + self.cells[:, 1], self.cells[:, 2] + self.cells[:, 3]), axis=1)

    def two_way_eligible(self, minimum: int = MIN_VIEW_CELLS) -> np.ndarray:
        return (self.cells_by_side >= minimum).all(axis=1)

    def four_way_eligible(self, minimum: int = MIN_VIEW_CELLS) -> np.ndarray:
        return (self.cells >= minimum).all(axis=1)


def incidence_from_metadata(perturbation: np.ndarray, batch: np.ndarray) -> SplitCounts:
    """Build quarter occupancy from cell metadata only; ``.X`` is never touched."""
    perturbation = np.asarray(perturbation, dtype=str)
    batch = np.asarray(batch, dtype=str)
    if perturbation.shape != batch.shape:
        raise ValueError("perturbation and batch must have one entry per cell")
    batches = np.asarray(np.unique(batch), dtype=str)
    identities = np.asarray(np.unique(perturbation[perturbation != CONTROL]), dtype=str)
    quarters, sides = batch_quarters(batches)
    if int((sides == 0).sum()) != LOCKED_A_BATCHES or int((sides == 1).sum()) != LOCKED_B_BATCHES:
        raise RuntimeError(
            f"reimplemented A/B allocation is {(sides == 0).sum()}/{(sides == 1).sum()}, "
            f"the frozen lock requires {LOCKED_A_BATCHES}/{LOCKED_B_BATCHES}"
        )
    order = {name: index for index, name in enumerate(identities.tolist())}
    batch_order = {name: index for index, name in enumerate(batches.tolist())}
    rows = np.asarray([order.get(name, -1) for name in perturbation.tolist()], dtype=np.int64)
    columns = np.asarray([batch_order[name] for name in batch.tolist()], dtype=np.int64)
    incidence = np.zeros((len(identities), len(batches)), dtype=np.int64)
    keep = rows >= 0
    np.add.at(incidence, (rows[keep], columns[keep]), 1)
    control_per_batch = np.zeros(len(batches), dtype=np.int64)
    np.add.at(control_per_batch, columns[~keep], 1)
    cells = np.stack([incidence[:, quarters == q].sum(axis=1) for q in QUARTERS], axis=1)
    occupied = np.stack([(incidence[:, quarters == q] > 0).sum(axis=1) for q in QUARTERS], axis=1)
    return SplitCounts(identities, batches, quarters, sides, cells, occupied, control_per_batch, incidence)


def assert_partition(counts: SplitCounts) -> None:
    """Every invariant the downstream estimator silently relies on."""
    if set(np.unique(counts.quarters).tolist()) != set(QUARTERS):
        raise RuntimeError("some quarter is empty")
    for quarter in QUARTERS:
        if int((counts.control_cells_per_batch[counts.quarters == quarter] == 0).sum()):
            raise RuntimeError(f"quarter {quarter} contains a batch with no non-targeting controls")
    if not np.array_equal(counts.cells.sum(axis=1), counts.cells_by_side.sum(axis=1)):
        raise RuntimeError("quarter cell counts do not sum to the two-way counts")
    eligible = counts.two_way_eligible()
    if int(eligible.sum()) != LOCKED_ELIGIBLE_IDENTITIES:
        raise RuntimeError(
            f"reimplemented two-way eligibility yields {int(eligible.sum())} identities, "
            f"the frozen lock requires {LOCKED_ELIGIBLE_IDENTITIES}"
        )
