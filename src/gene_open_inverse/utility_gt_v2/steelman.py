#!/usr/bin/env python3
"""Does the incumbent's high-magnitude prefix actually move the state toward the goal?

The goal-independent control cannot settle by itself whether ``-||r_c||^2`` is
nuisance or signal, because the cosine divides ``||r_c||`` out by construction
and so is structurally favoured by any control that rewards removing the
candidate main effect.  The question the control cannot answer is the direct one:
of the candidates each definition puts in the budget prefix, how many actually
reduce the squared endpoint distance, and by how much.

``V > 0`` is exactly ``||r_c - d_q||^2 < ||d_q||^2``, that is, the candidate
leaves the cell closer to the goal than doing nothing.  Grading one definition's
prefix under the other definition's independent view answers the question with no
shared measurement and no shared normalization.

Run from the repository root, either as a script or as
``python -m gene_open_inverse.utility_gt_v2.steelman``.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np

from gene_open_inverse.model.decision_alignment_v1.assets import MODALITIES, AuditConfig
from gene_open_inverse.model.global_objective_knowledge_attribution_v1.assets import GlobalAttributionAssets
from gene_open_inverse.utility_gt_v2 import contract as gt
from gene_open_inverse.utility_gt_v2 import validate as val

parser = argparse.ArgumentParser()
for name in ("asset", "config", "result", "output"):
    parser.add_argument(f"--{name}", required=True)
args = parser.parse_args()
asset, result_dir = Path(args.asset), Path(args.result)

quarter_all = np.load(asset / "quarter_responses.npy", mmap_mode="r")
identity_axis = np.asarray([str(v) for v in np.load(asset / "identity_axis.npy", allow_pickle=True)])
four_way = np.load(asset / "four_way_eligible.npy")
assets = GlobalAttributionAssets(modalities=MODALITIES, **AuditConfig.load(Path(args.config)).asset_arguments())
axis_index = {n: i for i, n in enumerate(identity_axis.tolist())}

role = assets.role_arrays("G_select")
ids = np.asarray([str(v) for v in role.base.ids])
position = np.asarray([axis_index[n] for n in ids.tolist()], dtype=np.int64)
eligible = np.load(result_dir / "G_select_eligible.npy")
candidate_rows = np.flatnonzero(four_way[position])
candidate_ids = ids[candidate_rows]
rows = np.flatnonzero(eligible[candidate_rows])

quarters = [np.asarray(quarter_all[i][position][candidate_rows], dtype=np.float64) for i in range(4)]
sides = [np.asarray(role.base.transitions[s], dtype=np.float64)[candidate_rows] for s in (0, 1)]
latent = [gt.latent_value(quarters[0], quarters[1]), gt.latent_value(quarters[2], quarters[3])]
cosine = [gt.cosine_value(sides[0]), gt.cosine_value(sides[1])]

order = {
    "incumbent_cosine": val.legal_orders(cosine[0][rows], candidate_ids, rows),
    "latent_value": val.legal_orders(latent[0][rows], candidate_ids, rows),
}
# graded under the OTHER definition measured on the independent view
graded_value = latent[1][rows]
graded_cosine = cosine[1][rows]
energy = np.einsum("ij,ij->i", quarters[2], quarters[3])[rows]

record: dict = {"schema": "VCDESIGN_UTILITY_GT_V2_STEELMAN_V1", "queries": int(rows.size),
                "candidates": int(candidate_rows.size), "prefix": {}}
row_index = np.arange(rows.size)[:, None]
for name, chosen in order.items():
    entry = {}
    for budget in (10, 20, 50):
        picked = chosen[:, :budget]
        value = graded_value[row_index, picked]
        entry[str(budget)] = {
            "fraction_closer_to_the_goal_than_no_op": float((value > 0).mean()),
            "per_query_fraction_closer_median": float(np.median((value > 0).mean(axis=1))),
            "value_median": float(np.median(value)),
            "value_relative_to_transition_energy_median": float(np.median(value / energy[:, None])),
            "cosine_under_the_independent_view_median": float(np.median(graded_cosine[row_index, picked])),
        }
    record["prefix"][name] = entry

whole = graded_value[:, :]
mask = np.ones(whole.shape, dtype=bool)
mask[row_index[:, 0], rows] = False
relative = np.where(mask, whole / energy[:, None], np.nan)
record["whole_pool_reference"] = {
    "fraction_closer_to_the_goal_than_no_op": float((whole[mask] > 0).mean()),
    "per_query_fraction_closer_median": float(np.median((whole > 0).sum(axis=1) / mask.sum(axis=1))),
    "value_median": float(np.median(whole[mask])),
    "value_relative_to_transition_energy_median": float(np.nanmedian(relative)),
    "cosine_under_the_independent_view_median": float(np.median(graded_cosine[mask])),
}
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps(record, indent=2, sort_keys=True))
