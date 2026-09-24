#!/usr/bin/env python3
"""Phases 6 and 7: leave-one-context-out V1, and the atlas-size scaling inside it.

Four folds.  In each, the held context's responses are invisible to every predictor,
because the training pool is built only from the other three.  Seven atlas designs
per fold: three single contexts, three pairs and the pooled triple, all under one
unchanged recipe and with no context feature anywhere.

The evaluation population is fixed per fold.  ``SEEN_CANDIDATE`` is defined against
the *full* three-context training set rather than against each design, so a
one-context design and a three-context design are graded on the same candidates.
Each design additionally reports what fraction of each stratum it actually
supervised, which is the number that would otherwise be hidden inside a moving
population.

Nothing here reads the anchor's ``G_check``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ..candidate_effect_distillation_v1 import contract as ced
from ..model.decision_alignment_v1.assets import AuditConfig
from ..model.global_objective_knowledge_attribution_v1.assets import GlobalAttributionAssets
from . import contexts as cx
from . import contract as fc
from . import evaluate as ev
from . import harness as hn
from . import overlap as ov
from . import predictors as pd


RANDOM_SEED = 20260918


def designs(training: tuple[str, ...]) -> list[tuple[str, ...]]:
    """Every non-empty subset of the training contexts, smallest first."""
    output: list[tuple[str, ...]] = []
    for size in (1, 2, 3):
        output.extend(tuple(sorted(item)) for item in itertools.combinations(sorted(training), size))
    return output


def load_contexts(args) -> tuple[dict, np.ndarray, GlobalAttributionAssets]:
    config = AuditConfig.load(Path(args.config))
    assets = GlobalAttributionAssets(modalities=("STRING", "MAPKG", "TEXT"), **config.asset_arguments())
    genes = np.asarray([str(v) for v in np.load(args.genes, allow_pickle=True)])
    contexts = {fc.ANCHOR_CONTEXT: cx.load_anchor(assets, args.anchor_sources, args.anchor_genes,
                                                  args.eligibility, genes)}
    for item in args.context:
        name, path = item.split("=", 1)
        contexts[name] = cx.load_built(name, path, genes)
    if set(contexts) != set(fc.CONTEXTS):
        raise RuntimeError(f"expected {fc.CONTEXTS}, received {sorted(contexts)}")
    return contexts, genes, assets


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    contexts, genes, assets = load_contexts(args)
    anchor_genes = np.asarray([str(v) for v in np.load(args.anchor_genes, allow_pickle=True)])
    bridge = hn.anchor_bridge(genes, anchor_genes)
    model, lock = hn.load_base_model(args.lock, assets, device)
    packs = {name: hn.pack_context(context, assets, model, bridge, device)
             for name, context in contexts.items()}
    del model
    torch.cuda.empty_cache()
    if args.pack_cache:
        for pack in packs.values():
            hn.save_pack(pack, args.pack_cache)

    record: dict = {
        "schema": "VCDESIGN_FOUR_CONTEXT_V1_LOCO",
        "read_anchor_G_check": False, "context_feature_given_to_any_model": False,
        "G_4": int(genes.size), "base_ranker": "frozen CLEAN_BASE, never retrained",
        "base_input_convention": "G_4 values embedded on the anchor axis, zero elsewhere",
        "lock_commit": lock.get("git_commit"),
        "lock_sha256": lock.get("lock_sha256"),
        "contexts": {name: contexts[name].summary() for name in fc.CONTEXTS},
        "pool_sizes": {name: int(packs[name].ids.size) for name in fc.CONTEXTS},
        "candidate_feature_verification": {name: packs[name].verification for name in fc.CONTEXTS},
        "overlap": ov.overlap_matrix(contexts),
        "folds": {},
    }

    common4 = ov.measured_everywhere(contexts)
    for held in fc.CONTEXTS:
        training = tuple(name for name in fc.CONTEXTS if name != held)
        pack = packs[held]
        seen = ov.supervised_identities(contexts, training)
        masks = ov.strata_masks(pack.ids, seen, common4)

        # RANDOM is not the comparison of record and never becomes one.  It exists so
        # that BASE is interpretable in a context the frozen base ranker has never seen:
        # if BASE - RANDOM collapses somewhere, then a large V1 - BASE there says more
        # about a failed base than about the effect model, and that has to be visible.
        generator = np.random.default_rng(RANDOM_SEED + fc.CONTEXTS.index(held))
        random_arm = [generator.standard_normal(pack.base[view].shape) for view in (0, 1)]
        score_by_arm: dict = {"BASE": pack.base, "RANDOM": random_arm, "ORACLE": pack.utility}
        predictor_record: dict = {}
        skipped_designs: dict = {}
        for design in designs(training):
            pool = hn.training_pool(packs, design)
            usable_rows = int(np.asarray(pool["present"], dtype=bool).sum())
            if usable_rows <= ced.BASIS_RANK:
                # Recorded rather than raised: a design that cannot support the frozen
                # rank is a fact about the atlas, and the remaining designs still answer
                # the scaling question.  Lowering the rank for it would make it a
                # different method and the comparison meaningless.
                skipped_designs["+".join(design)] = {
                    "usable_rows": usable_rows, "basis_rank": ced.BASIS_RANK,
                    "reason": "training pool is not larger than the frozen basis rank"}
                continue
            fitted = pd.fit(pool["features"], pool["responses"], pool["present"],
                            pool["keys"], design)
            effect = fitted.effect(pack.features, pack.present)
            score_by_arm[f"V1[{'+'.join(design)}]"] = hn.effect_arms(pack, effect)
            supervised = set(np.unique([k.split("|", 1)[1] for k in pool["keys"].tolist()]).tolist())
            predictor_record[f"V1[{'+'.join(design)}]"] = {
                "contexts": list(design), "n_contexts": len(design), "rows": fitted.rows,
                "identities": fitted.identities, "penalty": fitted.penalty,
                "held_out_gene_space_cosine": fitted.held_out_cosine,
                "explained_energy": fitted.basis.explained,
                "stratum_supervised_fraction": {
                    name: float(np.isin(pack.ids[mask], list(supervised)).mean()) if mask.any() else 0.0
                    for name, mask in masks.items()},
            }

        fold: dict = {"held": held, "training": list(training),
                      "queries": int(pack.query_rows.size), "pool": int(pack.ids.size),
                      "strata_counts": {name: int(mask.sum()) for name, mask in masks.items()},
                      "predictors": predictor_record, "skipped_designs": skipped_designs,
                      "strata": {}}
        for stratum, mask in masks.items():
            block = ev.evaluate_arms(score_by_arm, pack.utility, pack.ids,
                                     pack.self_positions, mask)
            if "skipped" in block:
                fold["strata"][stratum] = block
                continue
            vectors = block.pop("_vectors")
            block["paired_vs_base"] = {
                name: ev.paired(vectors[name]["MBRU"], vectors["BASE"]["MBRU"])
                for name in vectors if name not in ("BASE",)}
            block["base_minus_random"] = ev.paired(vectors["BASE"]["MBRU"],
                                                   vectors["RANDOM"]["MBRU"])
            block["paired_pooled_vs_single"] = {}
            pooled = f"V1[{'+'.join(sorted(training))}]"
            for name in list(vectors):
                if name.startswith("V1[") and name != pooled:
                    block["paired_pooled_vs_single"][f"{pooled}_minus_{name}"] = ev.paired(
                        vectors[pooled]["MBRU"], vectors[name]["MBRU"])
            fold["strata"][stratum] = block
        record["folds"][held] = fold

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record["runtime"] = {"timestamp": datetime.now(timezone.utc).isoformat(),
                         "python": platform.python_version(), "numpy": np.__version__}
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    (output.parent / "common4.json").write_text(json.dumps(
        {"identities": sorted(common4.tolist()), "count": int(common4.size),
         "rule": "identities with a usable response in all four contexts"},
        indent=2, sort_keys=True) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Phases 6 and 7: LOCO V1 and atlas scaling")
    parser.add_argument("--context", action="append", required=True, help="NAME=build_directory")
    parser.add_argument("--config", required=True)
    parser.add_argument("--genes", required=True, help="G_4.npy")
    parser.add_argument("--anchor-genes", dest="anchor_genes", required=True)
    parser.add_argument("--anchor-sources", dest="anchor_sources", required=True)
    parser.add_argument("--eligibility", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pack-cache", dest="pack_cache", default=None,
                        help="directory to cache per-context packs so later phases skip the GPU pass")
    record = run(parser.parse_args())
    print(f"G_4 {record['G_4']} genes; pools {record['pool_sizes']}")
    for held, fold in record["folds"].items():
        print(f"\n[held {held}] {fold['queries']} queries, pool {fold['pool']}, "
              f"strata {fold['strata_counts']}")
        block = fold["strata"].get("ALL", {})
        if "arms" not in block:
            continue
        anchor = block.get("base_minus_random", {})
        if anchor:
            low, high = anchor["bootstrap"]["median"]["ci95"]
            print(f"  BASE - RANDOM {anchor['median']:+.4f} [{low:+.4f},{high:+.4f}]  "
                  f"(does the frozen base transfer to this context at all)")
        for name, entry in sorted(block["arms"].items()):
            print(f"  {name:28s} MBRU {entry['MBRU']['median']:+.4f}")
        for name, item in sorted(block["paired_vs_base"].items()):
            low, high = item["bootstrap"]["median"]["ci95"]
            print(f"    {name:28s} - BASE  {item['median']:+.4f} [{low:+.4f},{high:+.4f}] "
                  f"improved {item['fraction_query_improved']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
