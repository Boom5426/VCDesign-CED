#!/usr/bin/env python3
"""QUERY_CONDITIONAL_CAPACITY_V1: is the scorer's interaction form the bottleneck?

Frozen and untouched: query eligibility, the direction utility, the candidate and
state representations, the FIT/SELECT split, the sampler, the GT and the action
model.  ``G_check`` is not read.  No architecture grid: exactly two scorers.

Both probes see one pair set, built once and saved before either is trained, and
they are trained with the same loss, optimizer, step budget and batch order.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ...utility_gt_v2.validate import cluster_bootstrap, legal_orders, summarize
from ...action_conditioned_utility_v1 import contract as act
from ...action_conditioned_utility_v1 import evaluate as actev
from ..decision_alignment_v1.assets import MODALITIES, AuditConfig
from ..decision_alignment_v1.contract import BUDGETS, mbru_from_nru, normalized_ranked_utility, query_references
from ..global_objective_knowledge_attribution_v1.assets import GlobalAttributionAssets
from ..global_objective_knowledge_attribution_v1.model import GlobalObjectiveKnowledgeAttributionModel
from ..open_vocab_dual_encoder_v1.data import FrozenTransitionAssets
from ..open_vocab_dual_encoder_v1.metrics import stable_order
from . import pairs as pairmod
from .probes import PROBES


SEED, LR, WEIGHT_DECAY, CLIP_NORM, BATCH_QUERIES = 20260916, 1e-4, 1e-4, 1.0, 16
SIDE_QUARTERS = {0: (0, 1), 1: (2, 3)}


def _tensors(role, device):
    return {
        "source": torch.from_numpy(role.source).to(device),
        "goal": torch.from_numpy(role.goal).to(device),
        "descriptors": {name: torch.from_numpy(value).to(device) for name, value in role.capabilities.items()},
        "masks": {name: torch.from_numpy(value).to(device) for name, value in role.capability_present.items()},
    }


def _starting_scores(config, assets, role, device) -> list[np.ndarray]:
    payload = torch.load(Path(config.checkpoints["M_SET"]["path"]), map_location=device, weights_only=False)
    model = GlobalObjectiveKnowledgeAttributionModel(assets.capability_dims).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    tensors = _tensors(role, device)
    out = []
    with torch.no_grad():
        candidates = model.encode_candidates(tensors["descriptors"], tensors["masks"])
        for view in (0, 1):
            rows = []
            for start in range(0, len(role.base.ids), 64):
                stop = min(start + 64, len(role.base.ids))
                query = model.encode_query(tensors["source"][view, start:stop], tensors["goal"][view, start:stop])
                rows.append(model.score_embeddings(query, candidates).float().cpu())
            out.append(torch.cat(rows).numpy())
    del model, payload
    return out


def _score_all(probe, tensors, view, count, device, batch=64) -> np.ndarray:
    probe.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, count, batch):
            stop = min(start + batch, count)
            rows.append(probe(tensors["source"][view, start:stop], tensors["goal"][view, start:stop],
                              tensors["descriptors"], tensors["masks"]).float().cpu())
    probe.train()
    return torch.cat(rows).numpy()


def _train(probe, tensors, pair_tables, device, steps_per_epoch, epochs, log) -> list[dict]:
    optimizer = torch.optim.AdamW(probe.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    history = []
    units = [(view, query) for view in (0, 1) for query in sorted(pair_tables[view])]
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        order = torch.randperm(len(units), generator=generator).tolist()
        totals = {"loss": 0.0, "pairs": 0.0, "grad": 0.0}
        steps = 0
        for start in range(0, len(order), BATCH_QUERIES):
            chunk = [units[index] for index in order[start:start + BATCH_QUERIES]]
            optimizer.zero_grad(set_to_none=True)
            loss_numerator = torch.zeros((), device=device)
            loss_denominator = torch.zeros((), device=device)
            count = 0
            for view in (0, 1):
                rows = [query for side, query in chunk if side == view]
                if not rows:
                    continue
                index = torch.as_tensor(rows, device=device)
                scores = probe(tensors["source"][view, index], tensors["goal"][view, index],
                               tensors["descriptors"], tensors["masks"])
                for position, query in enumerate(rows):
                    better, worse, weight = pair_tables[view][query]
                    margin = scores[position, worse] - scores[position, better]
                    loss_numerator = loss_numerator + (weight * F.softplus(margin)).sum()
                    loss_denominator = loss_denominator + weight.sum()
                    count += int(weight.numel())
            loss = loss_numerator / loss_denominator.clamp_min(1e-12)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite probe loss")
            loss.backward()
            squared = sum((p.grad.detach().float().norm().square() for p in probe.parameters() if p.grad is not None),
                          torch.zeros((), device=device))
            gradient = float(squared.sqrt().item())
            torch.nn.utils.clip_grad_norm_(probe.parameters(), CLIP_NORM)
            optimizer.step()
            totals["loss"] += float(loss.detach().item())
            totals["pairs"] += count
            totals["grad"] += gradient
            steps += 1
            if steps >= steps_per_epoch:
                break
        entry = {"epoch": epoch, "steps": steps, **{k: v / max(steps, 1) for k, v in totals.items()},
                 "seconds": time.perf_counter() - started}
        history.append(entry)
        print(f"  [{log}] epoch {epoch}: loss={entry['loss']:.6f} pairs/step={entry['pairs']:.0f} "
              f"grad={entry['grad']:.4f} {entry['seconds']:.1f}s", flush=True)
    return history


def _pairwise_accuracy(scores: list[np.ndarray], pair_sets: dict) -> dict:
    record = {}
    for kind in pairmod.KINDS:
        values = []
        for view in (0, 1):
            subset = pair_sets[view].of_kind(kind)
            if len(subset) == 0:
                continue
            row = scores[view]
            values.append((row[subset.query, subset.better] > row[subset.query, subset.worse]).astype(np.float64))
        joined = np.concatenate(values)
        record[kind] = {"accuracy": float(joined.mean()), "pairs": int(joined.size),
                        "chance": 0.5}
    return record


def _direction_and_prior(scores: list[np.ndarray], role, rows: np.ndarray, ids: np.ndarray,
                         explicit_prior: list[np.ndarray] | None) -> dict:
    full, empirical, explicit = [], [], []
    for view in (0, 1):
        utility = np.asarray(FrozenTransitionAssets.cross_view_utility(role.base, view), dtype=np.float64)
        prior_vector = scores[view][rows].mean(axis=0)
        for query in rows.tolist():
            references = query_references(utility[query], ids, query)
            for source, store in ((scores[view][query], full), (prior_vector, empirical)):
                order = stable_order(source, ids, excluded_index=query)
                order = order[order != query]
                store.append(mbru_from_nru(normalized_ranked_utility(utility[query, order], references)))
            if explicit_prior is not None:
                order = stable_order(explicit_prior[view], ids, excluded_index=query)
                order = order[order != query]
                explicit.append(mbru_from_nru(normalized_ranked_utility(utility[query, order], references)))
    full, empirical = np.asarray(full), np.asarray(empirical)
    record = {"MBRU_full": summarize(full), "MBRU_empirical_candidate_prior": summarize(empirical),
              "goal_conditioned_advantage": summarize(full - empirical),
              "goal_conditioned_advantage_bootstrap": cluster_bootstrap(full - empirical)}
    if explicit_prior is not None:
        explicit = np.asarray(explicit)
        record["MBRU_explicit_prior_branch"] = summarize(explicit)
        record["advantage_over_explicit_prior"] = summarize(full - explicit)
    record["_full"] = full
    record["_advantage"] = full - empirical
    return record


def _guardrail(scores: list[np.ndarray], quarters, candidate_rows, guard_rows, guard_ids) -> dict:
    record, store = {}, {}
    for view in (0, 1):
        proposing = act.view_statistics(quarters[SIDE_QUARTERS[view][0]], quarters[SIDE_QUARTERS[view][1]]).restrict(guard_rows)
        grading = act.view_statistics(quarters[SIDE_QUARTERS[1 - view][0]], quarters[SIDE_QUARTERS[1 - view][1]]).restrict(guard_rows)
        action, _ = act.optimal_attenuation(proposing)
        gain = act.cross_fitted_value(action, grading)
        matrix = np.ascontiguousarray(scores[view][np.ix_(candidate_rows[guard_rows], candidate_rows)])
        order = legal_orders(matrix, guard_ids, guard_rows)
        raw = actev.raw_ranked_utility(order, gain)
        index = np.arange(len(guard_rows))[:, None]
        record[f"view{view}"] = {
            **{f"RU@{budget}": summarize(raw[budget]) for budget in BUDGETS},
            **{f"fraction_J_positive@{budget}": float((gain[index, order[:, :budget]] > 0).mean()) for budget in BUDGETS},
            "realized_J_median@10": float(np.median(gain[index, order[:, :10]])),
        }
        for budget in BUDGETS:
            store.setdefault(f"RU@{budget}", []).append(raw[budget])
    record["both_views"] = {key: summarize(np.mean(values, axis=0)) for key, values in store.items()}
    record["_ru10"] = np.mean(store["RU@10"], axis=0)
    return record


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    config = AuditConfig.load(Path(args.config))
    assets = GlobalAttributionAssets(modalities=MODALITIES, **config.asset_arguments())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    fit_role, select_role = assets.role_arrays("G_fit"), assets.role_arrays("G_select")
    fit_ids = np.asarray([str(v) for v in fit_role.base.ids])
    select_ids = np.asarray([str(v) for v in select_role.base.ids])
    eligibility = Path(args.eligibility)
    fit_rows = np.flatnonzero(np.load(eligibility / "G_fit_eligible.npy"))
    select_rows = np.flatnonzero(np.load(eligibility / "G_select_eligible.npy"))

    starting = {"G_fit": _starting_scores(config, assets, fit_role, device),
                "G_select": _starting_scores(config, assets, select_role, device)}

    built = {}
    for name, role, ids, rows in (("G_fit", fit_role, fit_ids, fit_rows), ("G_select", select_role, select_ids, select_rows)):
        for view in (0, 1):
            utility = np.asarray(FrozenTransitionAssets.cross_view_utility(role.base, view), dtype=np.float64)
            pair_set = pairmod.build(utility, starting[name][view], ids, rows)
            pairmod.save(output / f"pairs_{name}_view{view}.npz", pair_set)
            built[(name, view)] = pair_set
            print(f"pairs {name} view{view}: {len(pair_set)} "
                  f"({ {kind: len(pair_set.of_kind(kind)) for kind in pairmod.KINDS} })", flush=True)

    fit_tensors, select_tensors = _tensors(fit_role, device), _tensors(select_role, device)
    pair_tables = {}
    for view in (0, 1):
        table = {}
        pair_set = built[("G_fit", view)]
        for query in np.unique(pair_set.query).tolist():
            keep = pair_set.query == query
            table[int(query)] = (torch.as_tensor(pair_set.better[keep].astype(np.int64), device=device),
                                 torch.as_tensor(pair_set.worse[keep].astype(np.int64), device=device),
                                 torch.as_tensor(pair_set.weight[keep], dtype=torch.float32, device=device))
        pair_tables[view] = table
    steps_per_epoch = (2 * len(fit_rows) + BATCH_QUERIES - 1) // BATCH_QUERIES

    asset = Path(args.asset)
    quarter_all = np.load(asset / "quarter_responses.npy", mmap_mode="r")
    identity_axis = np.asarray([str(v) for v in np.load(asset / "identity_axis.npy", allow_pickle=True)])
    four_way = np.load(asset / "four_way_eligible.npy")
    axis_index = {name: index for index, name in enumerate(identity_axis.tolist())}
    position = np.asarray([axis_index[name] for name in select_ids.tolist()], dtype=np.int64)
    candidate_rows = np.flatnonzero(four_way[position])
    guard_ids = select_ids[candidate_rows]
    guard_rows = np.flatnonzero(np.load(eligibility / "G_select_eligible.npy")[candidate_rows])
    quarters = [np.asarray(quarter_all[index][position][candidate_rows], dtype=np.float64) for index in range(4)]

    eval_pairs = {view: built[("G_select", view)] for view in (0, 1)}
    result: dict = {
        "schema": "VCDESIGN_QUERY_CONDITIONAL_CAPACITY_V1",
        "frozen": ["query_eligibility", "direction_utility", "candidate_representation",
                   "state_representation", "FIT_SELECT_split", "sampler", "GT", "action_model"],
        "read_G_check": False, "architecture_grid": False,
        "training": {"seed": SEED, "lr": LR, "weight_decay": WEIGHT_DECAY, "clip": CLIP_NORM,
                     "batch_queries": BATCH_QUERIES, "epochs": args.epochs, "steps_per_epoch": steps_per_epoch},
        "pairs": {"per_kind": pairmod.PAIRS_PER_KIND, "seed": pairmod.PAIR_SEED,
                  "identical_across_probes": True, "updated_during_training": False,
                  "decision_critical_defined_by": "STARTING M_SET epoch_010"},
        "queries": {"train": int(len(fit_rows)), "evaluate": int(len(select_rows)),
                    "guardrail": int(guard_rows.size)},
        "probes": {},
    }

    stored = {}
    reference = {"STARTING": starting["G_select"]}
    for name, builder in PROBES.items():
        torch.manual_seed(SEED)
        probe = builder(assets.capability_dims).to(device)
        parameters = sum(p.numel() for p in probe.parameters())
        print(f"\n[{name}] parameters={parameters:,}", flush=True)
        history = _train(probe, fit_tensors, pair_tables, device, steps_per_epoch, args.epochs, name)
        scores = [_score_all(probe, select_tensors, view, len(select_ids), device) for view in (0, 1)]
        explicit = None
        with torch.no_grad():
            branch = probe.candidate_prior(select_tensors["descriptors"], select_tensors["masks"])
        if branch is not None:
            explicit = [branch.float().cpu().numpy()] * 2
        entry = {"parameters": int(parameters), "history": history,
                 "pairwise_accuracy": _pairwise_accuracy(scores, eval_pairs)}
        direction = _direction_and_prior(scores, select_role, select_rows, select_ids, explicit)
        guard = _guardrail(scores, quarters, candidate_rows, guard_rows, guard_ids)
        entry["direction"] = {k: v for k, v in direction.items() if not k.startswith("_")}
        entry["guardrail"] = {k: v for k, v in guard.items() if not k.startswith("_")}
        result["probes"][name] = entry
        stored[name] = {"advantage": direction["_advantage"], "full": direction["_full"], "ru10": guard["_ru10"]}
        torch.save(probe.state_dict(), output / f"{name}.pt")
        del probe

    entry = {"pairwise_accuracy": _pairwise_accuracy(reference["STARTING"], eval_pairs)}
    direction = _direction_and_prior(reference["STARTING"], select_role, select_rows, select_ids, None)
    guard = _guardrail(reference["STARTING"], quarters, candidate_rows, guard_rows, guard_ids)
    entry["direction"] = {k: v for k, v in direction.items() if not k.startswith("_")}
    entry["guardrail"] = {k: v for k, v in guard.items() if not k.startswith("_")}
    result["probes"]["STARTING_REFERENCE"] = entry
    stored["STARTING_REFERENCE"] = {"advantage": direction["_advantage"], "full": direction["_full"],
                                    "ru10": guard["_ru10"]}

    result["paired"] = {}
    for better, worse in (("TRILINEAR_CONDITIONAL_PROBE", "CURRENT_SCORER_PROBE"),
                          ("TRILINEAR_CONDITIONAL_PROBE", "STARTING_REFERENCE"),
                          ("CURRENT_SCORER_PROBE", "STARTING_REFERENCE")):
        result["paired"][f"{better}_minus_{worse}"] = {
            key: {"summary": summarize(stored[better][key] - stored[worse][key]),
                  "fraction_improved": float((stored[better][key] - stored[worse][key] > 0).mean()),
                  "bootstrap": cluster_bootstrap(stored[better][key] - stored[worse][key])}
            for key in ("advantage", "full", "ru10")
        }

    result["runtime"] = {"timestamp": datetime.now(timezone.utc).isoformat(),
                         "python": platform.python_version(), "torch": torch.__version__, "device": str(device)}
    result["provenance"] = config.provenance()
    (output / "capacity.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Query-conditional capacity probe")
    for name in ("config", "eligibility", "asset", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    result = run(parser.parse_args())
    print("\n" + "=" * 78)
    for name, entry in result["probes"].items():
        accuracy = entry["pairwise_accuracy"]
        direction = entry["direction"]
        guard = entry["guardrail"]["both_views"]
        print(f"[{name}]")
        print("   pairwise  " + "  ".join(f"{kind}={accuracy[kind]['accuracy']:.4f}" for kind in accuracy))
        print(f"   MBRU full={direction['MBRU_full']['median']:+.4f} "
              f"prior={direction['MBRU_empirical_candidate_prior']['median']:+.4f} "
              f"advantage={direction['goal_conditioned_advantage']['median']:+.4f} "
              f"CI95={[round(v, 4) for v in direction['goal_conditioned_advantage_bootstrap']['mean']['ci95']]}")
        print(f"   guardrail RU@10={guard['RU@10']['median']:+.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
