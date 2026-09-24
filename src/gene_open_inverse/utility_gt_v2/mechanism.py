#!/usr/bin/env python3
"""What the latent value actually ranks by, since its ordering is nearly goal-independent.

Run from the repository root, either as a script or as
``python -m gene_open_inverse.utility_gt_v2.mechanism``.
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
latent = gt.latent_value(quarters[0], quarters[1])
cosine = gt.cosine_value(sides[0])

# cross-view estimate of ||r_c||^2, the query-independent term the value subtracts
magnitude = np.einsum("ij,ij->i", quarters[0], quarters[1])
prior = latent[rows].mean(axis=0)
centred_prior, centred_magnitude = prior - prior.mean(), -magnitude + magnitude.mean()
alignment = float((centred_prior * centred_magnitude).sum()
                  / np.sqrt((centred_prior ** 2).sum() * (centred_magnitude ** 2).sum()))

# variance decomposition of the value matrix over the eligible queries
block = latent[rows]
column_mean = block.mean(axis=0)
shared = float(((column_mean - block.mean()) ** 2).mean())
total = float(((block - block.mean()) ** 2).mean())

order_latent = val.legal_orders(block, candidate_ids, rows)
order_cosine = val.legal_orders(cosine[rows], candidate_ids, rows)
percentile = np.empty(len(magnitude))
percentile[np.argsort(magnitude, kind="stable")] = np.arange(len(magnitude)) / (len(magnitude) - 1)

# The decisive attribution test: rank by the candidate-only term alone.  The
# goal-independent prior packages every query-independent component together, so
# it cannot say which one drives the ordering.  A policy that ranks purely by
# -<r_c^Q0, r_c^Q1>, the cross-view estimate of -||r_c||^2, uses nothing else at
# all.  Grading it under the independent view puts the attribution on the same
# scale as the prior and the definition's own ordering.
candidate_only = np.broadcast_to(-magnitude, block.shape)
candidate_only_order = val.legal_orders(np.ascontiguousarray(candidate_only), candidate_ids, rows)
graded_second = gt.latent_value(quarters[2], quarters[3])[rows]
attribution = {
    "candidate_only_minus_response_energy": val.summarize(
        val.grade(candidate_only_order, graded_second, candidate_ids, rows).mbru),
    "goal_independent_prior": val.summarize(
        val.grade(val.goal_independent_order(block, candidate_ids, rows), graded_second, candidate_ids, rows).mbru),
    "own_ordering": val.summarize(
        val.grade(val.legal_orders(block, candidate_ids, rows), graded_second, candidate_ids, rows).mbru),
}

record = {
    "schema": "VCDESIGN_UTILITY_GT_V2_MECHANISM_V2",
    "attribution_in_decision_units": attribution,
    "prior_correlation_with_negative_response_energy": alignment,
    "share_of_value_variance_that_is_query_independent": shared / total,
    "median_response_energy_percentile_of_the_selected_prefix": {
        str(budget): {
            "latent_value": float(np.median(percentile[order_latent[:, :budget]])),
            "incumbent_cosine": float(np.median(percentile[order_cosine[:, :budget]])),
            "whole_pool_reference": 0.5,
        } for budget in (10, 20, 50)
    },
    "response_energy_non_positive_fraction": float((magnitude <= 0).mean()),
    "candidates": int(len(candidate_rows)), "queries": int(rows.size),
}
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps(record, indent=2, sort_keys=True))
