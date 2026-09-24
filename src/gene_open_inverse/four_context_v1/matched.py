#!/usr/bin/env python3
"""Phase 8: separate training quantity from context diversity.

Phase 7 shows what happens when the atlas grows naturally, but a three-context pool
has both more rows and more contexts, so a win there cannot be attributed.  This
phase holds one of the two fixed at a time.

8B, row-count matched.  Every arm trains on exactly ``N`` rows.  ``N`` is the
largest budget every comparable arm can meet, namely the smallest single-context
training pool in the fold, and it is computed from row counts alone.  A one-context
arm takes all ``N`` from one atlas, a two-context arm takes ``N/2`` from each, a
three-context arm ``N/3`` from each.  Anything that differs between them is context
mix, not size.

8C, unique-candidate matched.  Rows are restricted to the identities every training
context measured, so the candidate vocabulary is identical across arms too.  Three
readings: one context at ``M`` rows, the same identities spread across three contexts
at ``M`` rows, and all three contexts at ``3M`` rows.  The middle arm is the sharp
one: same identities, same row count, only the context mix moves.

Five pre-registered seeds, fixed in the contract.  The seed count is never raised
after seeing a result.
"""
from __future__ import annotations

import argparse
import itertools
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..candidate_effect_distillation_v1 import contract as ced
from . import contract as fc
from . import evaluate as ev
from . import harness as hn
from . import overlap as ov
from . import predictors as pd


def _context_of(keys: np.ndarray) -> np.ndarray:
    return np.asarray([key.split("|", 1)[0] for key in np.asarray(keys, dtype=str).tolist()])


def _identity_of(keys: np.ndarray) -> np.ndarray:
    return np.asarray([key.split("|", 1)[1] for key in np.asarray(keys, dtype=str).tolist()])


def row_budget(packs: dict, training: tuple[str, ...]) -> int:
    """The largest budget every single-context arm in this fold can meet."""
    return int(min(int((packs[name].fit_mask & packs[name].present).sum()) for name in training))


def sample_rows(pool: dict, design: tuple[str, ...], budget: int, seed: int) -> np.ndarray:
    """``budget`` usable rows, split as evenly as the design allows, deterministically."""
    generator = np.random.default_rng(seed)
    context = _context_of(pool["keys"])
    usable = np.asarray(pool["present"], dtype=bool)
    share, remainder = divmod(budget, len(design))
    picked = []
    for position, name in enumerate(sorted(design)):
        available = np.flatnonzero(usable & (context == name))
        want = share + (1 if position < remainder else 0)
        if available.size < want:
            raise ValueError(f"{name} holds {available.size} usable rows, fewer than the {want} asked")
        picked.append(generator.choice(available, size=want, replace=False))
    return np.sort(np.concatenate(picked))


def common_identities(pool: dict, training: tuple[str, ...]) -> np.ndarray:
    """Identities with a usable training row in every training context."""
    context, identity = _context_of(pool["keys"]), _identity_of(pool["keys"])
    usable = np.asarray(pool["present"], dtype=bool)
    sets = [set(identity[usable & (context == name)].tolist()) for name in training]
    common = set.intersection(*sets) if sets else set()
    return np.asarray(sorted(common), dtype=str)


def _fit_and_score(pool: dict, rows: np.ndarray, design: tuple[str, ...],
                   pack: hn.ContextPack) -> tuple[dict, np.ndarray]:
    fitted = pd.fit(pool["features"][rows], pool["responses"][rows], pool["present"][rows],
                    pool["keys"][rows], design)
    effect = fitted.effect(pack.features, pack.present)
    return ({"contexts": list(design), "n_contexts": len(design), "rows": fitted.rows,
             "identities": fitted.identities, "penalty": fitted.penalty,
             "held_out_gene_space_cosine": fitted.held_out_cosine}, effect)


def _delta(pack: hn.ContextPack, effect: np.ndarray, mask: np.ndarray) -> dict | None:
    scores = {"BASE": pack.base, "ARM": hn.effect_arms(pack, effect)}
    block = ev.evaluate_arms(scores, pack.utility, pack.ids, pack.self_positions, mask)
    if "skipped" in block:
        return None
    vectors = block.pop("_vectors")
    return {"arm": block["arms"]["ARM"], "base": block["arms"]["BASE"],
            "delta_MBRU": ev.paired(vectors["ARM"]["MBRU"], vectors["BASE"]["MBRU"])}


def run(args: argparse.Namespace) -> dict:
    packs = {name: hn.load_pack(name, args.packs) for name in fc.CONTEXTS}
    common4 = np.asarray(json.loads(Path(args.common4).read_text())["identities"], dtype=str) \
        if args.common4 else np.zeros(0, dtype=str)

    record: dict = {"schema": "VCDESIGN_FOUR_CONTEXT_V1_MATCHED",
                    "seeds": list(fc.MATCH_SEEDS), "row_budget_rule": fc.ROW_BUDGET_RULE,
                    "read_anchor_G_check": False, "folds": {}}

    for held in fc.CONTEXTS:
        training = tuple(sorted(name for name in fc.CONTEXTS if name != held))
        pack = packs[held]
        pool = hn.training_pool(packs, training)
        seen = np.unique(_identity_of(pool["keys"])[np.asarray(pool["present"], dtype=bool)])
        masks = ov.strata_masks(pack.ids, seen, common4)
        budget = row_budget(packs, training)
        too_small = budget <= ced.BASIS_RANK
        fold: dict = {"held": held, "training": list(training), "row_budget": budget,
                      "row_budget_supports_the_frozen_rank": not too_small,
                      "row_budget_components": {name: int((packs[name].fit_mask
                                                           & packs[name].present).sum())
                                                for name in training},
                      "row_matched": {}, "identity_matched": {}}

        # --- 8B row-count matched ------------------------------------------------
        arms: list[tuple[str, tuple[str, ...]]] = []
        for size in (1, 2, 3):
            for design in itertools.combinations(training, size):
                arms.append((f"N{size}[{'+'.join(design)}]", tuple(sorted(design))))
        if too_small:
            fold["row_matched"] = {"skipped": (
                f"the matched budget {budget} is not larger than the frozen basis rank "
                f"{ced.BASIS_RANK}; lowering the rank for this arm would make it a "
                "different method and the comparison meaningless")}
            arms = []
        for label, design in arms:
            per_seed = []
            for seed in fc.MATCH_SEEDS:
                try:
                    rows = sample_rows(pool, design, budget, seed)
                except ValueError as error:
                    per_seed.append({"seed": seed, "skipped": str(error)})
                    continue
                meta, effect = _fit_and_score(pool, rows, design, pack)
                entry = {"seed": seed, **meta}
                for stratum, mask in masks.items():
                    result = _delta(pack, effect, mask)
                    if result is not None:
                        entry.setdefault("strata", {})[stratum] = result
                per_seed.append(entry)
            fold["row_matched"][label] = {
                "n_contexts": len(design), "per_seed": per_seed,
                "summary": _across_seeds(per_seed, masks)}

        # --- 8C unique-candidate matched ----------------------------------------
        identities = common_identities(pool, training)
        identity = _identity_of(pool["keys"])
        context = _context_of(pool["keys"])
        usable = np.asarray(pool["present"], dtype=bool)
        fold["identity_matched"] = {"identities_in_every_training_context": int(identities.size)}
        if identities.size <= 256:
            fold["identity_matched"]["skipped"] = (
                f"only {int(identities.size)} identities are measured in all of {training}, "
                "which cannot support the frozen rank-256 basis")
        else:
            inside = usable & np.isin(identity, identities)
            arms_8c: dict = {}
            for name in training:
                arms_8c[f"SINGLE[{name}]"] = {
                    "design": (name,), "rows": lambda n=name: np.flatnonzero(inside & (context == n))}
            arms_8c[f"ALL3[{'+'.join(training)}]"] = {
                "design": training, "rows": lambda: np.flatnonzero(inside)}
            entries: dict = {}
            for label, spec in arms_8c.items():
                meta, effect = _fit_and_score(pool, spec["rows"](), spec["design"], pack)
                entries[label] = {**meta, "strata": {}}
                for stratum, mask in masks.items():
                    result = _delta(pack, effect, mask)
                    if result is not None:
                        entries[label]["strata"][stratum] = result
            # The sharp control: same identities, same row count, mixed contexts.
            mixed = []
            for seed in fc.MATCH_SEEDS:
                generator = np.random.default_rng(seed)
                rows = []
                for name in identities.tolist():
                    options = np.flatnonzero(inside & (identity == name))
                    rows.append(int(generator.choice(options)))
                meta, effect = _fit_and_score(pool, np.sort(np.asarray(rows)), training, pack)
                entry = {"seed": seed, **meta, "strata": {}}
                for stratum, mask in masks.items():
                    result = _delta(pack, effect, mask)
                    if result is not None:
                        entry["strata"][stratum] = result
                mixed.append(entry)
            entries["MIXED_MATCHED"] = {"per_seed": mixed, "summary": _across_seeds(mixed, masks)}
            fold["identity_matched"]["arms"] = entries
        record["folds"][held] = fold

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record["runtime"] = {"timestamp": datetime.now(timezone.utc).isoformat(),
                         "python": platform.python_version(), "numpy": np.__version__}
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def _across_seeds(per_seed: list, masks: dict) -> dict:
    """Mean and spread of the per-seed delta, so subsampling variance is visible."""
    summary: dict = {}
    for stratum in masks:
        values = [entry["strata"][stratum]["delta_MBRU"]["median"]
                  for entry in per_seed if "strata" in entry and stratum in entry.get("strata", {})]
        if not values:
            continue
        array = np.asarray(values, dtype=np.float64)
        summary[stratum] = {"seeds": int(array.size), "mean": float(array.mean()),
                            "min": float(array.min()), "max": float(array.max()),
                            "std": float(array.std(ddof=1)) if array.size > 1 else 0.0}
    rows = [entry.get("rows") for entry in per_seed if entry.get("rows") is not None]
    if rows:
        summary["rows"] = int(rows[0])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 8: quantity versus diversity")
    parser.add_argument("--packs", required=True)
    parser.add_argument("--common4", default=None)
    parser.add_argument("--output", required=True)
    record = run(parser.parse_args())
    for held, fold in record["folds"].items():
        print(f"\n[held {held}] row budget {fold['row_budget']} "
              f"from {fold['row_budget_components']}")
        for label, entry in sorted(fold["row_matched"].items()):
            block = entry["summary"].get("ALL")
            if block:
                print(f"  {label:34s} nctx {entry['n_contexts']}  deltaMBRU "
                      f"{block['mean']:+.4f} [{block['min']:+.4f},{block['max']:+.4f}] "
                      f"over {block['seeds']} seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
