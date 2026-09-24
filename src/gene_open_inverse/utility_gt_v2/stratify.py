#!/usr/bin/env python3
"""What the frozen eligibility gate buys, stated in the same decision units.

Run from the repository root, either as a script or as
``python -m gene_open_inverse.utility_gt_v2.stratify``.

The gate is Phase 1's only deliverable, so its effect has to be quantified on
the incumbent utility as well as on the new one, and on the queries it rejects
as well as the ones it keeps.  Nothing here changes a rule or a threshold.
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
parser.add_argument("--asset", required=True)
parser.add_argument("--config", required=True)
parser.add_argument("--result", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

asset, result_dir = Path(args.asset), Path(args.result)
quarter_all = np.load(asset / "quarter_responses.npy", mmap_mode="r")
identity_axis = np.asarray([str(v) for v in np.load(asset / "identity_axis.npy", allow_pickle=True)])
four_way = np.load(asset / "four_way_eligible.npy")
config = AuditConfig.load(Path(args.config))
assets = GlobalAttributionAssets(modalities=MODALITIES, **config.asset_arguments())
axis_index = {name: index for index, name in enumerate(identity_axis.tolist())}

role = assets.role_arrays("G_select")
ids = np.asarray([str(v) for v in role.base.ids])
position = np.asarray([axis_index[n] for n in ids.tolist()], dtype=np.int64)
eligible = np.load(result_dir / "G_select_eligible.npy")
candidate_rows = np.flatnonzero(four_way[position])
candidate_ids = ids[candidate_rows]

quarters = [np.asarray(quarter_all[i][position][candidate_rows], dtype=np.float64) for i in range(4)]
sides = [np.asarray(role.base.transitions[s], dtype=np.float64)[candidate_rows] for s in (0, 1)]
latent = [gt.latent_value(quarters[0], quarters[1]), gt.latent_value(quarters[2], quarters[3])]
cosine = [gt.cosine_value(sides[0]), gt.cosine_value(sides[1])]

keep = eligible[candidate_rows]
strata = {"all_four_way_survivors": np.arange(len(candidate_rows)),
          "eligible": np.flatnonzero(keep), "rejected_by_the_gate": np.flatnonzero(~keep)}

record = {"schema": "VCDESIGN_UTILITY_GT_V2_GATE_STRATIFICATION_V1", "strata": {}}
for stratum, rows in strata.items():
    entry = {"queries": int(rows.size)}
    for name, pair in (("incumbent_cosine", cosine), ("latent_value", latent)):
        both, prior_both = [], []
        for proposing, grading in ((0, 1), (1, 0)):
            order = val.legal_orders(pair[proposing][rows], candidate_ids, rows)
            both.append(val.grade(order, pair[grading][rows], candidate_ids, rows).mbru)
            prior = val.goal_independent_order(pair[proposing][rows], candidate_ids, rows)
            prior_both.append(val.grade(prior, pair[grading][rows], candidate_ids, rows).mbru)
        own, prior_value = np.mean(both, axis=0), np.mean(prior_both, axis=0)
        entry[name] = {
            "MBRU": val.summarize(own),
            "goal_independent_prior_MBRU": val.summarize(prior_value),
            "goal_conditioned_advantage": val.summarize(own - prior_value),
            "advantage_bootstrap": val.cluster_bootstrap(own - prior_value),
        }
    record["strata"][stratum] = entry

output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
for stratum, entry in record["strata"].items():
    print(f"{stratum} n={entry['queries']}")
    for name in ("incumbent_cosine", "latent_value"):
        item = entry[name]
        print(f"  {name:18s} MBRU med={item['MBRU']['median']:.4f} mean={item['MBRU']['mean']:.4f}  "
              f"prior med={item['goal_independent_prior_MBRU']['median']:.4f}  "
              f"advantage med={item['goal_conditioned_advantage']['median']:.4f} "
              f"CI95={[round(v,4) for v in item['advantage_bootstrap']['mean']['ci95']]}")
