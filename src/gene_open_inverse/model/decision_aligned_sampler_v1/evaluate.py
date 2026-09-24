#!/usr/bin/env python3
"""Step 3: training-side metrics under the direction utility, plus the J guardrail.

Two layers, deliberately separate.

Training-side, on the frozen eligible ``G_select`` queries under the direction
utility that the models were actually trained on: MBRU, Mean@B and
HighValueCount@B.  This is the surrogate the sampler optimises.

Scientific guardrail, on the eligible four-way subset under the
attenuation-conditioned endpoint improvement ``J(a) = 2a<r,d> - a^2||r||^2``,
cross-fitted so the measurement that chooses the attenuation never grades it:
fraction ``J > 0``, median realized ``J``, and RU@B in physical units.  The
guardrail does not train anything and does not enter any training objective.  It
exists to catch the case where the surrogate improves while the deployment value
does not.

The starting checkpoint is evaluated alongside both arms, so an arm's number is
never reported only as a difference from the other arm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ...utility_gt_v2.validate import cluster_bootstrap, legal_orders, summarize
from ...action_conditioned_utility_v1 import contract as act
from ...action_conditioned_utility_v1 import evaluate as actev
from ..decision_alignment_v1.assets import MODALITIES, AuditConfig
from ..decision_alignment_v1.contract import BUDGETS, mbru_from_nru, normalized_ranked_utility, query_references
from ..global_objective_knowledge_attribution_v1.assets import GlobalAttributionAssets
from ..global_objective_knowledge_attribution_v1.model import GlobalObjectiveKnowledgeAttributionModel
from ..open_vocab_dual_encoder_v1.data import FrozenTransitionAssets
from ..open_vocab_dual_encoder_v1.metrics import stable_order


HIGH_VALUE_QUANTILE = 0.95
QUERY_BATCH = 64
SIDE_QUARTERS = {0: (0, 1), 1: (2, 3)}


def _score(model, role, view: int, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        descriptors = {name: torch.from_numpy(value).to(device) for name, value in role.capabilities.items()}
        masks = {name: torch.from_numpy(value).to(device) for name, value in role.capability_present.items()}
        candidates = model.encode_candidates(descriptors, masks)
        rows = []
        for start in range(0, len(role.base.ids), QUERY_BATCH):
            stop = min(start + QUERY_BATCH, len(role.base.ids))
            source = torch.from_numpy(role.source[view, start:stop]).to(device)
            goal = torch.from_numpy(role.goal[view, start:stop]).to(device)
            rows.append(model.score_embeddings(model.encode_query(source, goal), candidates).float().cpu())
        return torch.cat(rows).numpy()


def _direction_metrics(scores: np.ndarray, utility: np.ndarray, ids: np.ndarray, rows: np.ndarray) -> dict:
    mbru = np.empty(len(rows))
    mean_at = {budget: np.empty(len(rows)) for budget in BUDGETS}
    high_at = {budget: np.empty(len(rows)) for budget in BUDGETS}
    for position, query in enumerate(rows.tolist()):
        order = stable_order(scores[query], ids, excluded_index=query)
        order = order[order != query]
        references = query_references(utility[query], ids, query)
        mbru[position] = mbru_from_nru(normalized_ranked_utility(utility[query, order], references))
        legal = np.delete(utility[query], query)
        threshold = float(np.quantile(legal, HIGH_VALUE_QUANTILE))
        for budget in BUDGETS:
            picked = utility[query, order[:budget]]
            mean_at[budget][position] = float(picked.mean())
            high_at[budget][position] = float((picked >= threshold).sum())
    return {"MBRU": mbru, **{f"Mean@{budget}": mean_at[budget] for budget in BUDGETS},
            **{f"HighValueCount@{budget}": high_at[budget] for budget in BUDGETS}}


def _guardrail(scores: np.ndarray, quarters, candidate_rows: np.ndarray, rows: np.ndarray,
               ids: np.ndarray, proposer, grader) -> dict:
    proposing = act.view_statistics(quarters[proposer[0]], quarters[proposer[1]]).restrict(rows)
    grading = act.view_statistics(quarters[grader[0]], quarters[grader[1]]).restrict(rows)
    action, _ = act.optimal_attenuation(proposing)
    gain = act.cross_fitted_value(action, grading)
    model_scores = np.ascontiguousarray(scores[np.ix_(candidate_rows[rows], candidate_rows)])
    order = legal_orders(model_scores, ids, rows)
    raw = actev.raw_ranked_utility(order, gain)
    index = np.arange(len(rows))[:, None]
    record = {f"RU@{budget}": summarize(raw[budget]) for budget in BUDGETS}
    record["_ru"] = raw
    for budget in BUDGETS:
        realized = gain[index, order[:, :budget]]
        record[f"prefix@{budget}"] = {
            "fraction_J_positive": float((realized > 0).mean()),
            "realized_J_median": float(np.median(realized)),
            "realized_J_mean": float(realized.mean()),
        }
    return record


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    config = AuditConfig.load(Path(args.config))
    assets = GlobalAttributionAssets(modalities=MODALITIES, **config.asset_arguments())
    role = assets.role_arrays("G_select")
    ids_all = np.asarray([str(value) for value in role.base.ids])
    eligible = np.load(Path(args.eligibility) / "G_select_eligible.npy")

    asset = Path(args.asset)
    quarter_all = np.load(asset / "quarter_responses.npy", mmap_mode="r")
    identity_axis = np.asarray([str(v) for v in np.load(asset / "identity_axis.npy", allow_pickle=True)])
    four_way = np.load(asset / "four_way_eligible.npy")
    axis_index = {name: index for index, name in enumerate(identity_axis.tolist())}
    position = np.asarray([axis_index[name] for name in ids_all.tolist()], dtype=np.int64)
    candidate_rows = np.flatnonzero(four_way[position])
    guard_ids = ids_all[candidate_rows]
    guard_rows = np.flatnonzero(eligible[candidate_rows])
    quarters = [np.asarray(quarter_all[index][position][candidate_rows], dtype=np.float64) for index in range(4)]

    eligible_rows = np.flatnonzero(eligible)
    checkpoints = {"STARTING": Path(config.checkpoints["M_SET"]["path"])}
    for arm in ("CURRENT", "DECISION_ALIGNED"):
        checkpoints[arm] = Path(args.continuation) / arm / "final.pt"

    result: dict = {"schema": "VCDESIGN_DECISION_ALIGNED_SAMPLER_V1_EVALUATION",
                    "queries_direction": int(eligible_rows.size), "candidates_direction": int(len(ids_all)),
                    "queries_guardrail": int(guard_rows.size), "candidates_guardrail": int(candidate_rows.size),
                    "high_value_quantile": HIGH_VALUE_QUANTILE, "models": {}}
    stored: dict = {}
    for name, path in checkpoints.items():
        payload = torch.load(path, map_location=device, weights_only=False)
        model = GlobalObjectiveKnowledgeAttributionModel(assets.capability_dims).to(device)
        model.load_state_dict(payload["model_state"], strict=True)
        entry: dict = {"checkpoint": str(path), "direction": {}, "guardrail": {}}
        direction_arrays: dict = {}
        guard_arrays: dict = {}
        for view in (0, 1):
            scores = _score(model, role, view, device)
            utility = np.asarray(FrozenTransitionAssets.cross_view_utility(role.base, view), dtype=np.float64)
            metrics = _direction_metrics(scores, utility, ids_all, eligible_rows)
            for key, values in metrics.items():
                direction_arrays.setdefault(key, []).append(values)
            proposer, grader = (SIDE_QUARTERS[view], SIDE_QUARTERS[1 - view])
            guard = _guardrail(scores, quarters, candidate_rows, guard_rows, guard_ids, proposer, grader)
            entry["guardrail"][f"view{view}"] = {key: value for key, value in guard.items() if not key.startswith("_")}
            for budget in BUDGETS:
                guard_arrays.setdefault(f"RU@{budget}", []).append(guard["_ru"][budget])
        direction_arrays = {key: np.mean(values, axis=0) for key, values in direction_arrays.items()}
        guard_arrays = {key: np.mean(values, axis=0) for key, values in guard_arrays.items()}
        entry["direction"] = {key: summarize(values) for key, values in direction_arrays.items()}
        entry["guardrail"]["both_views"] = {key: summarize(values) for key, values in guard_arrays.items()}
        result["models"][name] = entry
        stored[name] = {"direction": direction_arrays, "guardrail": guard_arrays}
        del model, payload

    result["paired"] = {}
    for better, worse in (("DECISION_ALIGNED", "CURRENT"), ("DECISION_ALIGNED", "STARTING"), ("CURRENT", "STARTING")):
        entry = {}
        for layer in ("direction", "guardrail"):
            for key in stored[better][layer]:
                difference = stored[better][layer][key] - stored[worse][layer][key]
                entry[f"{layer}:{key}"] = {"summary": summarize(difference),
                                           "fraction_query_improved": float((difference > 0).mean()),
                                           "bootstrap": cluster_bootstrap(difference)}
        result["paired"][f"{better}_minus_{worse}"] = entry

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "evaluation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Matched sampler evaluation")
    for name in ("config", "continuation", "eligibility", "asset", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    result = run(parser.parse_args())
    print(f"queries: direction={result['queries_direction']} guardrail={result['queries_guardrail']}")
    for name, entry in result["models"].items():
        d, g = entry["direction"], entry["guardrail"]["both_views"]
        print(f"\n[{name}]")
        print(f"  direction  MBRU={d['MBRU']['median']:+.4f}  Mean@10={d['Mean@10']['median']:+.4f}  "
              f"Mean@20={d['Mean@20']['median']:+.4f}  HVC@10={d['HighValueCount@10']['median']:.2f}  "
              f"HVC@20={d['HighValueCount@20']['median']:.2f}")
        print(f"  guardrail  RU@10={g['RU@10']['median']:+.5f}  RU@20={g['RU@20']['median']:+.5f}  "
              f"RU@50={g['RU@50']['median']:+.5f}")
        for view in ("view0", "view1"):
            p = entry["guardrail"][view]["prefix@10"]
            print(f"             {view} Top-10 J>0={p['fraction_J_positive']:.3f} medJ={p['realized_J_median']:+.4f}")
    for key, entry in result["paired"].items():
        print(f"\n{key}")
        for metric in ("direction:MBRU", "direction:Mean@10", "direction:HighValueCount@10",
                       "guardrail:RU@10", "guardrail:RU@20"):
            item = entry[metric]
            low, high = item["bootstrap"]["mean"]["ci95"]
            print(f"  {metric:28s} med={item['summary']['median']:+.5f} mean={item['summary']['mean']:+.5f} "
                  f"CI95=[{low:+.5f},{high:+.5f}] improved={item['fraction_query_improved']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
