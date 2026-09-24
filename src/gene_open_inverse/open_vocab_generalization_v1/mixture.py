#!/usr/bin/env python3
"""Phase C: the open-vocabulary deployment mixture, and whether support predicts harm.

A real candidate vocabulary is neither all measured nor all unmeasured.  It is a
mixture, and the question a lab actually faces is whether the effect branch pushes
candidates it cannot predict into the first ten, twenty or fifty slots.  So the pool,
the queries, the base and the graded utility are identical across every share here,
and the only thing that moves is which candidates take their predicted effect from a
predictor that had never measured them.

The share assignment is nested: whatever is unseen at 25 percent is unseen at 50 and
at 75.  That makes the three shares one monotone sequence over a single fixed ordering
of the candidates rather than three unrelated draws, so a difference between shares is
about how much unseen material the pool holds and not about which candidates happened
to be drawn.

Nothing here builds a gate.  Section 21 asks only whether support predicts the sign of
what the effect branch buys, and a promotion's worth is measured against what it
actually displaced rather than against an average.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..four_context_v1 import contract as fc
from ..four_context_v1 import evaluate as ev
from ..four_context_v1 import harness as hn
from ..model.decision_alignment_v1.contract import BUDGETS
from ..model.open_vocab_dual_encoder_v1.metrics import stable_order
from . import contract as ov
from .natural import paired_all, quantile_strata


def share_value(ids: np.ndarray, namespace: str = ov.MIXTURE_NAMESPACE) -> np.ndarray:
    """A fixed uniform value per identity, so the shares nest instead of being redrawn."""
    digest = [hashlib.sha256(f"{namespace}:{value}".encode()).digest()
              for value in np.asarray(ids, dtype=str).tolist()]
    return np.asarray([int.from_bytes(value[:8], "big") / float(1 << 64) for value in digest],
                      dtype=np.float64)


def _order_rows(scores: np.ndarray, candidate_ids: np.ndarray, self_positions: np.ndarray,
                budget: int) -> list[np.ndarray]:
    """The frozen ranking convention, applied row by row on one candidate stratum."""
    orders = []
    for row in range(scores.shape[0]):
        position = int(self_positions[row])
        excluded = position if position >= 0 else None
        order = stable_order(scores[row], candidate_ids, excluded)
        if excluded is not None:
            order = order[order != position]
        orders.append(order[:budget])
    return orders


def composition(scores: np.ndarray, utility: np.ndarray, candidate_ids: np.ndarray,
                self_positions: np.ndarray, groups: dict) -> dict:
    """What the first B slots are made of, and what each group contributed to their value."""
    record: dict = {}
    for budget in BUDGETS:
        picks = _order_rows(scores, candidate_ids, self_positions, budget)
        counts = {name: [] for name in groups}
        value = {name: [] for name in groups}
        for row, order in enumerate(picks):
            for name, mask in groups.items():
                inside = mask[order]
                counts[name].append(float(inside.sum()) / max(order.size, 1))
                value[name].append(float(utility[row, order[inside]].sum()) / max(order.size, 1))
        record[f"@{budget}"] = {
            "fraction": {name: float(np.mean(values)) for name, values in counts.items()},
            "utility_contribution": {name: float(np.mean(values)) for name, values in value.items()},
            "mean_utility": float(np.mean([utility[row, order].mean()
                                           for row, order in enumerate(picks)]))}
    return record


def entrant_value(base: np.ndarray, mixed: np.ndarray, utility: np.ndarray,
                  candidate_ids: np.ndarray, self_positions: np.ndarray,
                  budget: int) -> dict:
    """What the effect branch bought by promoting each candidate into the first B slots.

    A candidate that enters the budget under the fused score but not under the base
    alone displaced someone.  Crediting it with its own utility would say nothing,
    because a pool's candidates are not interchangeable; crediting it with its utility
    minus the mean utility of what it displaced says whether the swap was worth making.
    """
    totals = np.zeros(candidate_ids.size, dtype=np.float64)
    counts = np.zeros(candidate_ids.size, dtype=np.int64)
    base_picks = _order_rows(base, candidate_ids, self_positions, budget)
    mixed_picks = _order_rows(mixed, candidate_ids, self_positions, budget)
    for row, (before, after) in enumerate(zip(base_picks, mixed_picks)):
        entered = np.setdiff1d(after, before, assume_unique=False)
        left = np.setdiff1d(before, after, assume_unique=False)
        if entered.size == 0 or left.size == 0:
            continue
        reference = float(utility[row, left].mean())
        totals[entered] += utility[row, entered] - reference
        counts[entered] += 1
    return {"total": totals, "count": counts}


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    def rank(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        ranks = np.empty(values.size, dtype=np.float64)
        ranks[order] = np.arange(values.size, dtype=np.float64)
        unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        totals = np.zeros(unique.size, dtype=np.float64)
        np.add.at(totals, inverse, ranks)
        return (totals / counts)[inverse]
    a, b = rank(np.asarray(left, dtype=np.float64)), rank(np.asarray(right, dtype=np.float64))
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 0 else 0.0


def run_context(pack, cached: dict, table: dict) -> dict:
    """One held context, three unseen shares, on one fixed candidate pool.

    The fusion is taken over the full context pool and the evaluation restricts
    afterwards, which is what every other phase of this programme does.  Fusing over the
    restricted pool instead would standardize the two terms over a different column
    population and so run the effect branch at a different effective weight here than in
    Phases A and B, and the same arm would then report two different numbers in two
    documents.  A candidate outside the eligible set keeps its seen-arm effect in every
    share, because no training context measured it and its exposure is not what varies.
    """
    eligible = np.asarray(cached["eligible"], dtype=bool)
    columns = np.flatnonzero(eligible)
    ids = pack.ids[columns]
    values = np.zeros(eligible.size, dtype=np.float64)
    values[columns] = share_value(ids)
    leverage_full = np.asarray(table["leverage"], dtype=np.float64)
    leverage = leverage_full[columns]
    strata = quantile_strata(leverage, ov.MASK_SUPPORT_STRATA, ov.MASK_STRATUM_NAMES)

    seen = [np.asarray(cached[f"seen_{view}"]) for view in (0, 1)]
    masked = [np.asarray(cached[f"masked_{view}"]) for view in (0, 1)]

    record: dict = {"eligible_candidates": int(columns.size),
                    "pool": int(pack.ids.size),
                    "queries": int(pack.query_rows.size),
                    "fusion": "row standardized over the full context pool, then restricted",
                    "shares": {}, "support_predicts_harm": {}}

    arms_reference = {
        "BASE": list(pack.base),
        "ALL_SEEN": [ev.fuse(pack.base[view], seen[view]) for view in (0, 1)],
        "ALL_UNSEEN": [ev.fuse(pack.base[view], masked[view]) for view in (0, 1)],
    }
    for share in ov.MIXTURE_SHARES:
        unseen = eligible & (values < share)
        mixed = [np.where(unseen[None, :], masked[view], seen[view]) for view in (0, 1)]
        arms = dict(arms_reference)
        arms["MIXED"] = [ev.fuse(pack.base[view], mixed[view]) for view in (0, 1)]

        block = ev.evaluate_arms(arms, pack.utility, pack.ids, pack.self_positions, eligible)
        vectors = block.pop("_vectors")
        block["paired_vs_base"] = {name: paired_all(vectors, name, "BASE")
                                   for name in vectors if name != "BASE"}
        block["realised_unseen_share"] = float(unseen[columns].mean())

        # The composition diagnostics rank inside the evaluated pool, exactly as the
        # evaluator does, so they are read off the restricted columns of the same scores.
        restricted = {name: [np.asarray(arms[name][view])[:, columns] for view in (0, 1)]
                      for name in ("BASE", "MIXED")}
        utility = [np.asarray(pack.utility[view])[:, columns] for view in (0, 1)]
        remap = {int(value): index for index, value in enumerate(columns.tolist())}
        inside = np.asarray([remap.get(int(p), -1) for p in pack.self_positions.tolist()],
                            dtype=np.int64)
        groups = {"seen": ~unseen[columns], "unseen": unseen[columns]}
        for name, mask in strata.items():
            groups[f"unseen_{name}"] = unseen[columns] & mask
        block["composition"] = {
            arm: composition(restricted[arm][0], utility[0], ids, inside, groups)
            for arm in ("BASE", "MIXED")}
        record["shares"][f"{share:.2f}"] = block

        if abs(share - 0.50) < 1e-9:
            harm: dict = {}
            for budget in BUDGETS:
                pooled_total = np.zeros(columns.size)
                pooled_count = np.zeros(columns.size, dtype=np.int64)
                for view in (0, 1):
                    part = entrant_value(restricted["BASE"][view], restricted["MIXED"][view],
                                         utility[view], ids, inside, budget)
                    pooled_total += part["total"]
                    pooled_count += part["count"]
                promoted = pooled_count > 0
                mean_value = np.zeros(columns.size)
                mean_value[promoted] = pooled_total[promoted] / pooled_count[promoted]
                is_unseen = unseen[columns]
                harm[f"@{budget}"] = {
                    "promoted_candidates": int(promoted.sum()),
                    "promoted_unseen": int((promoted & is_unseen).sum()),
                    "spearman_leverage_vs_entrant_value": spearman(
                        leverage[promoted], mean_value[promoted]),
                    "spearman_leverage_vs_entrant_value_unseen_only": spearman(
                        leverage[promoted & is_unseen], mean_value[promoted & is_unseen])
                    if int((promoted & is_unseen).sum()) > 2 else None,
                    "mean_entrant_value": float(mean_value[promoted].mean()),
                    "positive_fraction": float((mean_value[promoted] > 0).mean()),
                    "mean_entrant_value_seen": float(mean_value[promoted & ~is_unseen].mean())
                    if int((promoted & ~is_unseen).sum()) else None,
                    "mean_entrant_value_unseen": float(mean_value[promoted & is_unseen].mean())
                    if int((promoted & is_unseen).sum()) else None,
                    "by_support": {
                        name: {"promoted": int((promoted & is_unseen & mask).sum()),
                               "mean_entrant_value": float(
                                   mean_value[promoted & is_unseen & mask].mean())
                               if int((promoted & is_unseen & mask).sum()) else None,
                               "positive_fraction": float(
                                   (mean_value[promoted & is_unseen & mask] > 0).mean())
                               if int((promoted & is_unseen & mask).sum()) else None}
                        for name, mask in strata.items()},
                }
            record["support_predicts_harm"] = harm
    return record


def run(args: argparse.Namespace) -> dict:
    packs = {name: hn.load_pack(name, args.packs) for name in fc.CONTEXTS}
    source = Path(args.masking_dir)
    record: dict = {
        "schema": "VCDESIGN_OPEN_VOCAB_V1_MIXED_POOL",
        "read_anchor_G_check": False,
        "mixture_namespace": ov.MIXTURE_NAMESPACE, "shares": list(ov.MIXTURE_SHARES),
        "mixture_rule": ov.MIXTURE_RULE,
        "entrant_value_definition": ov.ENTRANT_VALUE_DEFINITION,
        "no_gate_in_this_mission": ov.NO_GATE_IN_THIS_MISSION,
        "contexts": {},
    }
    for held in fc.CONTEXTS:
        with np.load(source / f"effect_cosine_{held}.npz", allow_pickle=False) as cached, \
             np.load(source / f"support_{held}.npz", allow_pickle=False) as table:
            record["contexts"][held] = run_context(
                packs[held], {key: cached[key] for key in cached.files},
                {key: table[key] for key in table.files})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record["runtime"] = {"timestamp": datetime.now(timezone.utc).isoformat(),
                         "python": platform.python_version(), "numpy": np.__version__}
    output.write_text(json.dumps(record, indent=2, sort_keys=True, default=float) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase C: open-vocabulary mixed candidate pools")
    parser.add_argument("--packs", required=True)
    parser.add_argument("--masking-dir", dest="masking_dir", required=True)
    parser.add_argument("--output", required=True)
    record = run(parser.parse_args())
    for held, block in record["contexts"].items():
        print(f"\n[{held}] pool {block['eligible_candidates']} queries {block['queries']}")
        for share, entry in block["shares"].items():
            arms = entry["paired_vs_base"]
            line = "  ".join(
                f"{name} {arms[name]['MBRU']['median']:+.4f}"
                for name in ("ALL_SEEN", "MIXED", "ALL_UNSEEN") if name in arms)
            print(f"  share {share} (realised {entry['realised_unseen_share']:.3f})  {line}")
            top = entry["composition"]["MIXED"]["@10"]["fraction"]
            print(f"      top-10 unseen {top['unseen']:.3f}  "
                  f"low support unseen {top['unseen_low_support']:.3f}")
        for budget, entry in block.get("support_predicts_harm", {}).items():
            print(f"  entrant value {budget}: promoted {entry['promoted_candidates']} "
                  f"mean {entry['mean_entrant_value']:+.4f} positive "
                  f"{entry['positive_fraction']:.3f} spearman(leverage) "
                  f"{entry['spearman_leverage_vs_entrant_value']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
