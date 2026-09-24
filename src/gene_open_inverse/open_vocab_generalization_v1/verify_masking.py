#!/usr/bin/env python3
"""An independent witness for the controlled-masking run.

Lesson L3 in this programme's decision log: a script must not be its own only witness.
So this file re-derives, from the cached packs and the frozen assets alone, the things
the masking run asserts, using its own code rather than calling the run's.  It recomputes
the masking folds from the symbols, rebuilds one context's masked training pools without
touching ``masked_training_pool``, refits a predictor and checks that the penalty and
held-out cosine the record reports are reproduced, and re-ranks both arms with an
independent ordering to check the reported top-50 utilities.

It refuses to overwrite its own output, and it downgrades to FAIL rather than raising, so
a failure is a recorded verdict and not a missing file.
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
from ..four_context_v1 import predictors as pd
from ..model.decision_alignment_v1.contract import BUDGETS
from . import contract as ov


def independent_folds(ids: np.ndarray) -> np.ndarray:
    """The fold rule written out again rather than imported."""
    output = np.empty(len(ids), dtype=np.int64)
    for position, value in enumerate(np.asarray(ids, dtype=str).tolist()):
        payload = ov.MASK_NAMESPACE + ":" + value
        output[position] = int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16) \
            % ov.MASK_FOLDS
    return output


def independent_pool(packs: dict, names: tuple[str, ...], excluded: set) -> dict:
    """The pooled training rows, assembled by a different loop than the run's."""
    features, responses, present, identities = [], [], [], []
    for name in names:
        pack = packs[name]
        for row in range(pack.ids.size):
            if not bool(pack.fit_mask[row]) or str(pack.ids[row]) in excluded:
                continue
            features.append(pack.features[row])
            responses.append(0.5 * (pack.responses[0, row].astype(np.float64)
                                    + pack.responses[1, row].astype(np.float64)))
            present.append(bool(pack.present[row]))
            identities.append(f"{name}|{pack.ids[row]}")
    return {"features": np.asarray(features), "responses": np.asarray(responses),
            "present": np.asarray(present), "keys": np.asarray(identities, dtype=str)}


def independent_mean_at(scores: np.ndarray, utility: np.ndarray, columns: np.ndarray,
                        self_positions: np.ndarray, budget: int) -> np.ndarray:
    """Mean utility of the top ``budget``, ordered without the frozen tie rule.

    Ties are broken differently here on purpose.  If the two implementations agree, the
    ordering is not resting on a tie convention; if they disagree, the size of the
    disagreement is the tie rate and is reported rather than hidden.
    """
    scores = np.asarray(scores, dtype=np.float64)[:, columns]
    utility = np.asarray(utility, dtype=np.float64)[:, columns]
    remap = {int(value): index for index, value in enumerate(columns.tolist())}
    inside = np.asarray([remap.get(int(p), -1) for p in np.asarray(self_positions).tolist()],
                        dtype=np.int64)
    output = np.empty(scores.shape[0], dtype=np.float64)
    for row in range(scores.shape[0]):
        values = scores[row].copy()
        if inside[row] >= 0:
            values[inside[row]] = -np.inf
        order = np.argsort(-values, kind="stable")[:budget]
        output[row] = float(utility[row, order].mean())
    return output


def run(args: argparse.Namespace) -> dict:
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite an existing verification record at {output}")
    record = json.loads(Path(args.record).read_text())
    packs = {name: hn.load_pack(name, args.packs) for name in fc.CONTEXTS}
    checks: list[dict] = []

    def note(name: str, passed: bool, detail) -> None:
        checks.append({"check": name, "pass": bool(passed), "detail": detail})

    # 1.  the fold rule, re-derived from the symbols
    for held in fc.CONTEXTS:
        block = record["cross_context"][held]
        pack = packs[held]
        folds = independent_folds(pack.ids)
        full = hn.training_pool(packs, tuple(block["training"]))
        supervised = set(np.unique(
            [key.split("|", 1)[1] for key in full["keys"].tolist()]).tolist())
        eligible = np.asarray([str(value) in supervised for value in pack.ids.tolist()]) \
            & np.asarray(pack.present, dtype=bool)
        note(f"{held}: eligible candidate count",
             int(eligible.sum()) == block["eligible_candidates"],
             {"independent": int(eligible.sum()), "recorded": block["eligible_candidates"]})
        sizes = {str(f): int((eligible & (folds == f)).sum()) for f in range(ov.MASK_FOLDS)}
        recorded = {f: entry["masked_identities"]
                    for f, entry in block["fold_predictors"].items()}
        note(f"{held}: masked identities per fold", sizes == recorded,
             {"independent": sizes, "recorded": recorded})
        note(f"{held}: folds partition the eligible set",
             sum(sizes.values()) == int(eligible.sum()), sizes)

    # 2.  one context rebuilt from scratch, pool and predictor
    held = args.context
    block = record["cross_context"][held]
    pack = packs[held]
    folds = independent_folds(pack.ids)
    training = tuple(block["training"])
    full = hn.training_pool(packs, training)
    supervised = set(np.unique(
        [key.split("|", 1)[1] for key in full["keys"].tolist()]).tolist())
    eligible = np.asarray([str(v) in supervised for v in pack.ids.tolist()]) \
        & np.asarray(pack.present, dtype=bool)

    fold = int(args.fold)
    inside = eligible & (folds == fold)
    excluded = set(pack.ids[inside].tolist())
    pool = independent_pool(packs, training, excluded)
    leaked = [key for key in pool["keys"].tolist() if key.split("|", 1)[1] in excluded]
    note(f"{held} fold {fold}: no masked identity survives in any training context",
         not leaked, {"leaked_rows": len(leaked)})

    fitted = pd.fit(pool["features"], pool["responses"], pool["present"], pool["keys"], training)
    entry = block["fold_predictors"][str(fold)]
    note(f"{held} fold {fold}: predictor rows", fitted.rows == entry["rows"],
         {"independent": fitted.rows, "recorded": entry["rows"]})
    note(f"{held} fold {fold}: selected penalty", fitted.penalty == entry["penalty"],
         {"independent": fitted.penalty, "recorded": entry["penalty"]})
    note(f"{held} fold {fold}: held-out gene-space cosine",
         abs(fitted.held_out_cosine - entry["held_out_gene_space_cosine"]) < 1e-12,
         {"independent": fitted.held_out_cosine,
          "recorded": entry["held_out_gene_space_cosine"]})

    # 3.  the graded numbers, re-ranked by an independent ordering
    seen_predictor = pd.fit(full["features"], full["responses"], full["present"],
                            full["keys"], training)
    note(f"{held}: seen predictor penalty",
         seen_predictor.penalty == block["seen_predictor"]["penalty"],
         {"independent": seen_predictor.penalty,
          "recorded": block["seen_predictor"]["penalty"]})
    seen_effect = seen_predictor.effect(pack.features, pack.present)
    masked_effect = fitted.effect(pack.features, pack.present)
    arms = {"EFFECT_SEEN": hn.effect_arms(pack, seen_effect),
            "EFFECT_MASKED": hn.effect_arms(pack, masked_effect)}
    columns = np.flatnonzero(inside)
    deviations = {}
    for name, by_view in arms.items():
        for budget in BUDGETS:
            values = np.concatenate([
                independent_mean_at(by_view[view], pack.utility[view], columns,
                                    pack.self_positions, budget) for view in (0, 1)])
            got = float(np.median(values))
            want = record["cross_context"][held]["folds"][str(fold)]["arms"][name][
                f"Mean@{budget}"]["median"]
            deviations[f"{name}|Mean@{budget}"] = {"independent": got, "recorded": want,
                                                   "absolute_difference": abs(got - want)}
    worst = max(item["absolute_difference"] for item in deviations.values())
    note(f"{held} fold {fold}: independent top-B utilities reproduce the record",
         worst <= float(args.tolerance), {"worst_absolute_difference": worst,
                                          "tolerance": float(args.tolerance),
                                          "per_metric": deviations})

    # 4.  the arms are graded on one pool
    note(f"{held} fold {fold}: both arms graded on the same candidate pool",
         int(inside.sum()) == record["cross_context"][held]["folds"][str(fold)][
             "masked_candidates"],
         {"independent": int(inside.sum())})

    verdict = "PASS" if all(item["pass"] for item in checks) else "FAIL"
    payload = {"schema": "VCDESIGN_OPEN_VOCAB_V1_MASKING_VERIFICATION",
               "verdict": verdict, "record": str(args.record),
               "verified_context": held, "verified_fold": fold,
               "tolerance": float(args.tolerance), "checks": checks,
               "runtime": {"timestamp": datetime.now(timezone.utc).isoformat(),
                           "python": platform.python_version(), "numpy": np.__version__}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="independent verification of Phase B")
    parser.add_argument("--packs", required=True)
    parser.add_argument("--record", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--context", default="RPE1")
    parser.add_argument("--fold", default="0")
    # The graded metrics are deterministic, so the only legitimate source of disagreement
    # is the tie convention between the two orderings.  A nonzero tolerance is therefore a
    # measurement of the tie rate, not a licence, and the observed value is recorded.
    parser.add_argument("--tolerance", default="1e-12")
    payload = run(parser.parse_args())
    for item in payload["checks"]:
        print(f"{'PASS' if item['pass'] else 'FAIL'}  {item['check']}")
        if not item["pass"]:
            print(f"        {json.dumps(item['detail'], default=float)[:400]}")
    print(f"\nVERDICT {payload['verdict']}")
    return 0 if payload["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
