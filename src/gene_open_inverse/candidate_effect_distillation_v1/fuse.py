#!/usr/bin/env python3
"""Phase 8 and 9, in the smallest form that respects the firewall.

The integrated score is

    s(q,c) = z[ s_incumbent(q,c) ] + beta * z[ cos(d_q, r_hat_c) ]

where ``z`` is a per-query standardisation across candidates.  The only fitted
quantity is the scalar ``beta``, and it is fitted on ``G_fit`` alone.

Two things make that honest.  First, a ``G_fit`` candidate's effect prediction must
be out of fold: the ridge saw that candidate's measured response during training, so
an in-sample prediction would make the effect branch look far better on ``G_fit``
than it can be on a deployment candidate, and ``beta`` would be chosen for a model
that does not exist.  Second, ``G_select`` is never consulted while choosing
``beta``; it is read once, afterwards, with the choice already fixed.

``beta = 1`` is reported alongside, because a unit weight needs no fitting at all
and a result that only survives at a fitted weight is a weaker result.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ..model.decision_aligned_sampler_v1.evaluate import _direction_metrics
from ..model.decision_alignment_v1.assets import AuditConfig
from ..model.global_objective_knowledge_attribution_v1.assets import GlobalAttributionAssets
from ..model.open_vocab_dual_encoder_v1.data import FrozenTransitionAssets
from ..model.query_conditional_capacity_v1.run import _guardrail, _starting_scores
from ..utility_gt_v2.validate import cluster_bootstrap, summarize
from . import basis as bs
from . import contract as ced
from . import predictor as pr


BETA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def _unit(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-30)


def _row_z(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    return (values - values.mean(axis=1, keepdims=True)) / np.maximum(
        values.std(axis=1, keepdims=True), 1e-12)


def out_of_fold_effect(features: np.ndarray, targets: np.ndarray, keep: np.ndarray,
                       penalty: float, basis, candidates: int) -> np.ndarray:
    """Predicted effect for every G_fit candidate, never from a fold that saw it."""
    coefficients = np.zeros((candidates, targets.shape[1]), dtype=np.float64)
    for held in pr.identity_folds(keep.size):
        train = np.setdiff1d(np.arange(keep.size), held)
        path = pr.RidgePath(features[keep][train], targets[keep][train])
        coefficients[keep[held]] = path.predict(features[keep][held], penalty)
    effect = basis.reconstruct(coefficients)
    missing = np.ones(candidates, dtype=bool)
    missing[keep] = False
    effect[missing] = 0.0
    return effect


def fusion_scores(incumbent: list[np.ndarray], transitions: np.ndarray, effect: np.ndarray,
                  beta: float) -> list[np.ndarray]:
    unit = _unit(effect)
    missing = np.linalg.norm(effect, axis=1) <= 0.0
    output = []
    for view in (0, 1):
        cosine = _unit(transitions[view]) @ unit.T
        cosine[:, missing] = ced.ALL_MISSING_SCORE
        output.append(_row_z(incumbent[view]) + beta * _row_z(cosine))
    return output


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    config = AuditConfig.load(Path(args.config))
    assets = GlobalAttributionAssets(modalities=ced.LEGACY_MODALITIES, **config.asset_arguments())
    fit_role, select_role = assets.role_arrays("G_fit"), assets.role_arrays("G_select")
    fit_ids = np.asarray([str(v) for v in fit_role.base.ids])
    select_ids = np.asarray([str(v) for v in select_role.base.ids])
    fit_rows = np.flatnonzero(np.load(Path(args.eligibility) / "G_fit_eligible.npy"))
    select_rows = np.flatnonzero(np.load(Path(args.eligibility) / "G_select_eligible.npy"))

    previous = json.loads((Path(args.distillation) / "candidate_effect_distillation.json").read_text())
    penalty = float(previous["predictor"]["cross_validation"]["selected_penalty"])

    fit_transitions = np.asarray(fit_role.base.transitions, dtype=np.float64)
    consensus = bs.consensus_response(fit_transitions)
    basis = bs.randomized_basis(consensus)
    targets = basis.project(consensus)
    features = pr.static_features(fit_role, ced.PRIMARY_MODALITIES)
    keep = features.fit_rows()

    effect_fit = out_of_fold_effect(features.values, targets, keep, penalty, basis, fit_ids.size)
    in_sample = basis.reconstruct(
        pr.RidgePath(features.values[keep], targets[keep]).predict(features.values, penalty))
    in_sample[~features.present_any] = 0.0
    honesty = {
        "out_of_fold_cosine_to_measured_median": float(np.median(np.einsum(
            "ij,ij->i", _unit(effect_fit[keep]), _unit(consensus[keep])))),
        "in_sample_cosine_to_measured_median": float(np.median(np.einsum(
            "ij,ij->i", _unit(in_sample[keep]), _unit(consensus[keep])))),
    }

    fit_incumbent = _starting_scores(config, assets, fit_role, device)
    fit_utility = [np.asarray(FrozenTransitionAssets.cross_view_utility(fit_role.base, view),
                              dtype=np.float64) for view in (0, 1)]
    selection = {}
    for beta in BETA_GRID:
        fused = fusion_scores(fit_incumbent, fit_transitions, effect_fit, beta)
        values = np.concatenate([_direction_metrics(fused[view], fit_utility[view], fit_ids,
                                                    fit_rows)["MBRU"] for view in (0, 1)])
        selection[f"{beta:g}"] = float(np.median(values))
    chosen = float(max(BETA_GRID, key=lambda b: selection[f"{b:g}"]))
    del fit_incumbent, fit_utility

    select_incumbent = _starting_scores(config, assets, select_role, device)
    select_transitions = np.asarray(select_role.base.transitions, dtype=np.float64)
    select_utility = [np.asarray(FrozenTransitionAssets.cross_view_utility(select_role.base, view),
                                 dtype=np.float64) for view in (0, 1)]
    select_features = pr.static_features(select_role, ced.PRIMARY_MODALITIES)
    select_effect = basis.reconstruct(
        pr.RidgePath(features.values[keep], targets[keep]).predict(select_features.values, penalty))
    select_effect[~select_features.present_any] = 0.0

    asset = Path(args.asset)
    quarter_all = np.load(asset / "quarter_responses.npy", mmap_mode="r")
    identity_axis = np.asarray([str(v) for v in np.load(asset / "identity_axis.npy", allow_pickle=True)])
    four_way = np.load(asset / "four_way_eligible.npy")
    axis_index = {name: index for index, name in enumerate(identity_axis.tolist())}
    position = np.asarray([axis_index[name] for name in select_ids.tolist()], dtype=np.int64)
    candidate_rows = np.flatnonzero(four_way[position])
    guard_rows = np.flatnonzero(np.load(Path(args.eligibility) / "G_select_eligible.npy")[candidate_rows])
    quarters = [np.asarray(quarter_all[index][position][candidate_rows], dtype=np.float64)
                for index in range(4)]

    record: dict = {
        "schema": "VCDESIGN_CANDIDATE_EFFECT_DISTILLATION_V1_FUSION",
        "read_G_check": False,
        "beta_grid": list(BETA_GRID), "beta_selected_on": "G_fit eligible queries only",
        "beta_selection_curve_G_fit_MBRU": selection, "beta_selected": chosen,
        "out_of_fold_note": "G_fit effect predictions are out of fold; in-sample values are shown "
                            "only to show how much an in-sample beta would have been misled",
        "prediction_honesty": honesty,
        "ridge_penalty": penalty, "rows": {},
    }

    incumbent_mbru = None
    for name, beta in (("INCUMBENT", None), ("FUSED_BETA_1", 1.0), (f"FUSED_BETA_SELECTED", chosen)):
        matrices = select_incumbent if beta is None else fusion_scores(
            select_incumbent, select_transitions, select_effect, beta)
        per_view = [_direction_metrics(matrices[view], select_utility[view], select_ids, select_rows)
                    for view in (0, 1)]
        direction = {key: summarize(np.concatenate([entry[key] for entry in per_view]))
                     for key in per_view[0]}
        mbru = np.concatenate([entry["MBRU"] for entry in per_view])
        if incumbent_mbru is None:
            incumbent_mbru = mbru
        entry = {"beta": beta, "direction": direction,
                 "guardrail": {k: v for k, v in _guardrail(matrices, quarters, candidate_rows,
                                                           guard_rows, select_ids[candidate_rows]).items()
                               if not k.startswith("_")}}
        difference = mbru - incumbent_mbru
        entry["paired_against_incumbent"] = {
            **summarize(difference), "bootstrap": cluster_bootstrap(difference),
            "fraction_query_improved": float((difference > 0).mean())}
        record["rows"][name] = entry

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    record["runtime"] = {"timestamp": datetime.now(timezone.utc).isoformat(),
                         "python": platform.python_version(), "numpy": np.__version__}
    (output / "fusion.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Minimal integrated model, beta fitted on G_fit")
    for name in ("config", "eligibility", "asset", "distillation", "output"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--device", default="cuda")
    record = run(parser.parse_args())
    print(f"out-of-fold G_fit effect cosine {record['prediction_honesty']['out_of_fold_cosine_to_measured_median']:.4f} "
          f"against in-sample {record['prediction_honesty']['in_sample_cosine_to_measured_median']:.4f}")
    print("beta selection on G_fit (MBRU median): " + "  ".join(
        f"{k}:{v:+.4f}" for k, v in record["beta_selection_curve_G_fit_MBRU"].items()))
    print(f"beta selected on G_fit: {record['beta_selected']:g}")
    print(f"\n{'row':22s} {'beta':>5s} {'MBRU':>8s} {'Mean@10':>9s} {'Mean@20':>9s} {'HVC@10':>7s} "
          f"{'HVC@20':>7s} {'J RU@10':>9s} {'J>0@10':>7s}   paired vs incumbent")
    for name, entry in record["rows"].items():
        d, g = entry["direction"], entry["guardrail"]
        p = entry["paired_against_incumbent"]
        low, high = p["bootstrap"]["median"]["ci95"]
        beta = "-" if entry["beta"] is None else f"{entry['beta']:g}"
        print(f"{name:22s} {beta:>5s} {d['MBRU']['median']:+8.4f} {d['Mean@10']['median']:+9.4f} "
              f"{d['Mean@20']['median']:+9.4f} {d['HighValueCount@10']['median']:7.2f} "
              f"{d['HighValueCount@20']['median']:7.2f} {g['both_views']['RU@10']['median']:+9.5f} "
              f"{g['view0']['fraction_J_positive@10']:7.4f}   "
              f"{p['median']:+.4f} [{low:+.4f},{high:+.4f}] improved={p['fraction_query_improved']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
