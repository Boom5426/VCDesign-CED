"""PHR@B under contract V1 for the external contexts: the quarter gate and the realized ``J`` matrices.

The quarter responses come from the same streaming pass that built the two views
(``four_context_v1.build`` saved them next to the side responses), so no rebuild is needed.  The
gate proves it: for every four-way identity, the cell-count weighted mean of Q0 and Q1 must equal
view A and that of Q2 and Q3 view B, and the pack's responses must equal the build's side responses.

The frozen build zeroes quarter responses of identities below ``MIN_VIEW_CELLS`` in any quarter, so
``J`` exists only on the four-way sub-pool.  PHR is therefore computed on that sub-pool, never
approximated elsewhere.  For a view-v row the action is chosen on side v's quarter pair (which also
builds the row's query), and ``J`` is realized on side 1-v's pair, which nothing else touched.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..action_conditioned_utility_v1 import contract as act
from . import contract as cc


def _build_axes(four_context_run: str, held: str, pack_ids: np.ndarray, pool_rows: np.ndarray) -> dict:
    build = Path(four_context_run) / held
    ids = np.load(build / "identity_axis.npy", allow_pickle=True).astype(str)
    genes = np.load(build / "gene_axis.npy", allow_pickle=True).astype(str)
    g4 = np.load(Path(four_context_run) / "genes" / "G_4.npy", allow_pickle=True).astype(str)
    position = {g: i for i, g in enumerate(genes.tolist())}
    if not np.array_equal(ids[pool_rows], np.asarray(pack_ids, dtype=str)):
        raise RuntimeError(f"{held}: pack identities are not the build's pool rows")
    return {"build": build, "columns": np.asarray([position[g] for g in g4.tolist()], dtype=np.int64)}


def reconstruction_gate(four_context_run: str, held: str, pack) -> dict:
    axes = _build_axes(four_context_run, held, pack.ids, pack.pool_rows)
    build, columns = axes["build"], axes["columns"]
    four_way = np.load(build / "four_way_eligible.npy")[pack.pool_rows]
    incidence = np.load(build / "incidence.npy")[pack.pool_rows]
    quarter_of_batch = np.load(build / "batch_quarter.npy")
    side = np.load(build / "side_responses.npy", mmap_mode="r")
    quarters = np.load(build / "quarter_responses.npy", mmap_mode="r")
    rows = pack.pool_rows[four_way]
    n = np.stack([incidence[:, quarter_of_batch == q].sum(axis=1) for q in range(4)], axis=1)[four_way].astype(np.float64)
    q = np.asarray(quarters[:, rows], dtype=np.float64)
    s = np.asarray(side[:, rows], dtype=np.float64)
    view_a = (n[:, 0, None] * q[0] + n[:, 1, None] * q[1]) / (n[:, 0] + n[:, 1])[:, None]
    view_b = (n[:, 2, None] * q[2] + n[:, 3, None] * q[3]) / (n[:, 2] + n[:, 3])[:, None]
    pack_error = float(np.abs(np.asarray(side[:, pack.pool_rows][:, :, columns]) - np.asarray(pack.responses)).max())
    record = {"four_way_identities": int(four_way.sum()), "pool": int(pack.ids.size),
              "view_A_from_Q0Q1_max_abs": float(np.abs(view_a - s[0]).max()),
              "view_B_from_Q2Q3_max_abs": float(np.abs(view_b - s[1]).max()),
              "pack_equals_build_side_max_abs": pack_error,
              "min_quarter_cells": int(n.min()) if n.size else 0,
              "quarter_batches_disjoint_refine_sides": bool(np.array_equal(quarter_of_batch // 2, np.load(build / "batch_side.npy"))),
              "tolerance": cc.PHR_RECONSTRUCTION_TOLERANCE}
    record["passes"] = bool(record["view_A_from_Q0Q1_max_abs"] <= cc.PHR_RECONSTRUCTION_TOLERANCE
                            and record["view_B_from_Q2Q3_max_abs"] <= cc.PHR_RECONSTRUCTION_TOLERANCE
                            and pack_error == 0.0 and record["quarter_batches_disjoint_refine_sides"]
                            and record["min_quarter_cells"] >= 25)
    return record


def realized_j(four_context_run: str, held: str, pack, stratum: np.ndarray) -> dict:
    """``J`` on the four-way sub-pool, per view: rows are four-way queries, columns sub-pool candidates."""
    axes = _build_axes(four_context_run, held, pack.ids, pack.pool_rows)
    build, columns = axes["build"], axes["columns"]
    four_way = np.load(build / "four_way_eligible.npy")[pack.pool_rows]
    quarters = np.load(build / "quarter_responses.npy", mmap_mode="r")
    members = np.flatnonzero(four_way)                        # pool positions with four quarters
    local = {int(p): i for i, p in enumerate(members.tolist())}
    q = [np.asarray(quarters[k][pack.pool_rows[members]][:, columns], dtype=np.float64) for k in range(4)]
    self_positions = np.asarray(pack.self_positions)
    query_rows = np.flatnonzero(four_way[self_positions])     # pack query rows usable for PHR
    candidates = np.flatnonzero(four_way & np.asarray(stratum, dtype=bool))
    q_local = np.asarray([local[int(self_positions[r])] for r in query_rows.tolist()], dtype=np.int64)
    c_local = np.asarray([local[int(c)] for c in candidates.tolist()], dtype=np.int64)
    out = {"query_rows": query_rows, "candidates": candidates, "J": []}
    for view in (0, 1):
        proposer = act.view_statistics(q[2 * view], q[2 * view + 1])
        grader = act.view_statistics(q[2 * (1 - view)], q[2 * (1 - view) + 1])
        action, _ = act.optimal_attenuation(proposer)
        gain = act.cross_fitted_value(action, grader)
        out["J"].append(np.ascontiguousarray(gain[np.ix_(q_local, c_local)]))
    return out
