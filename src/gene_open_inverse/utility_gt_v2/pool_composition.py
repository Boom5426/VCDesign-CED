#!/usr/bin/env python3
"""Is the latent value's goal-independent reproducibility a property of the pool?

76% of the 1081 candidates are identities whose own transition the frozen rule
judges unmeasurable.  If ``-||r_c||^2`` separates those identities stably in both
views, then the query-independent component would be an artifact of pool
composition rather than of the value definition, and the Phase-2 reading would
have to change.  The test is direct: rerun the same comparison with the candidate
vocabulary restricted to the eligible identities only.

This is a diagnostic, not a production rule.  The production candidate vocabulary
stays complete; nothing here filters candidates by response reliability.

Run from the repository root, either as a script or as
``python -m gene_open_inverse.utility_gt_v2.pool_composition``.
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
survivors = np.flatnonzero(four_way[position])
keep = eligible[survivors]

record: dict = {"schema": "VCDESIGN_UTILITY_GT_V2_POOL_COMPOSITION_V1", "pools": {}}
for label, candidate_rows in (("complete_candidate_vocabulary", survivors),
                              ("eligible_candidates_only", survivors[keep])):
    candidate_ids = ids[candidate_rows]
    rows = np.flatnonzero(eligible[candidate_rows])
    quarters = [np.asarray(quarter_all[i][position][candidate_rows], dtype=np.float64) for i in range(4)]
    sides = [np.asarray(role.base.transitions[s], dtype=np.float64)[candidate_rows] for s in (0, 1)]
    pairs = {"latent_value": [gt.latent_value(quarters[0], quarters[1]), gt.latent_value(quarters[2], quarters[3])],
             "incumbent_cosine": [gt.cosine_value(sides[0]), gt.cosine_value(sides[1])]}
    magnitude = np.einsum("ij,ij->i", quarters[0], quarters[1])
    entry = {"queries": int(rows.size), "candidates": int(candidate_rows.size)}
    for name, pair in pairs.items():
        own, prior = [], []
        for proposing, grading in ((0, 1), (1, 0)):
            own.append(val.grade(val.legal_orders(pair[proposing][rows], candidate_ids, rows),
                                 pair[grading][rows], candidate_ids, rows).mbru)
            prior.append(val.grade(val.goal_independent_order(pair[proposing][rows], candidate_ids, rows),
                                   pair[grading][rows], candidate_ids, rows).mbru)
        own, prior = np.mean(own, axis=0), np.mean(prior, axis=0)
        entry[name] = {"MBRU": val.summarize(own), "goal_independent_prior_MBRU": val.summarize(prior),
                       "goal_conditioned_advantage": val.summarize(own - prior)}
    block = pairs["latent_value"][0][rows]
    column = block.mean(axis=0)
    entry["latent_value_query_independent_variance_share"] = float(
        ((column - block.mean()) ** 2).mean() / ((block - block.mean()) ** 2).mean())
    centred_prior, centred_magnitude = column - column.mean(), -magnitude + magnitude.mean()
    entry["latent_prior_correlation_with_negative_response_energy"] = float(
        (centred_prior * centred_magnitude).sum()
        / np.sqrt((centred_prior ** 2).sum() * (centred_magnitude ** 2).sum()))
    record["pools"][label] = entry

Path(args.output).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
for label, entry in record["pools"].items():
    print(f"{label}: queries={entry['queries']} candidates={entry['candidates']} "
          f"query_independent_share={entry['latent_value_query_independent_variance_share']:.4f} "
          f"prior_corr_with_neg_energy={entry['latent_prior_correlation_with_negative_response_energy']:.4f}")
    for name in ("incumbent_cosine", "latent_value"):
        item = entry[name]
        print(f"  {name:18s} MBRU med={item['MBRU']['median']:.4f}  prior med={item['goal_independent_prior_MBRU']['median']:.4f}  "
              f"advantage med={item['goal_conditioned_advantage']['median']:.4f}")
