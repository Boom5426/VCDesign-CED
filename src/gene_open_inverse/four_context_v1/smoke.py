#!/usr/bin/env python3
"""Integration check on the anchor plus whichever contexts exist, and the axis record.

Two jobs, neither of which is a result.

It exercises the parts most likely to be wired wrongly and that no unit test can
reach: the anchor loader, the gene bridge into the frozen base ranker, the
candidate-feature reproduction, and one full fit-and-grade cycle in each external
context.  It prints the anchor-trained effect gain, which for RPE1 the previous
mission measured independently, so a gross wiring fault shows up as a number that
does not resemble it rather than as a silent success.

It also quantifies what the gene axis alone changes.  Run it once on the intersection
of whatever contexts are available and again with ``--genes G_4.npy``: every other
input is identical, so the difference is attributable to the axis and to nothing else.
That record has to exist before any new-context model result is read, so a later shift
in the RPE1 number cannot be mistaken for a change in method.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from ..model.decision_alignment_v1.assets import AuditConfig
from ..model.global_objective_knowledge_attribution_v1.assets import GlobalAttributionAssets
from . import contexts as cx
from . import evaluate as ev
from . import harness as hn
from . import predictors as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Integration check across available contexts")
    parser.add_argument("--context", action="append", required=True, help="NAME=build_directory")
    parser.add_argument("--config", required=True)
    parser.add_argument("--anchor-genes", dest="anchor_genes", required=True)
    parser.add_argument("--anchor-sources", dest="anchor_sources", required=True)
    parser.add_argument("--eligibility", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--genes", default=None, help="explicit gene axis, for example G_4")
    parser.add_argument("--output", default=None,
                        help="write the record as JSON, so the axis attribution is an "
                             "artifact rather than a line in a log")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = AuditConfig.load(Path(args.config))
    assets = GlobalAttributionAssets(modalities=("STRING", "MAPKG", "TEXT"), **config.asset_arguments())
    anchor_genes = np.asarray([str(v) for v in np.load(args.anchor_genes, allow_pickle=True)])

    directories = {}
    for item in args.context:
        name, path = item.split("=", 1)
        directories[name] = Path(path)

    if args.genes:
        shared = np.asarray([str(v) for v in np.load(args.genes, allow_pickle=True)])
        label = f"explicit axis {Path(args.genes).name}"
    else:
        shared = anchor_genes
        for path in directories.values():
            other = np.asarray([str(v) for v in np.load(path / "gene_axis.npy", allow_pickle=True)])
            shared = np.intersect1d(shared, other)
        shared = np.sort(shared)
        label = f"intersection of the anchor and {', '.join(directories)}"
    print(f"gene axis {shared.size} ({label}); anchor axis {anchor_genes.size}")
    record: dict = {"schema": "VCDESIGN_FOUR_CONTEXT_V1_INTEGRATION_CHECK",
                    "gene_axis_size": int(shared.size), "gene_axis_source": label,
                    "anchor_axis_size": int(anchor_genes.size),
                    "read_anchor_G_check": False, "contexts": {}}

    contexts = {"K562": cx.load_anchor(assets, args.anchor_sources, args.anchor_genes,
                                       args.eligibility, shared)}
    for name, path in directories.items():
        contexts[name] = cx.load_built(name, path, shared)
    for name, context in contexts.items():
        print(f"{name}: {json.dumps(context.summary())}")
        record["contexts"][name] = {"summary": context.summary()}

    device = torch.device(args.device)
    bridge = hn.anchor_bridge(shared, anchor_genes)
    model, _ = hn.load_base_model(args.lock, assets, device)
    packs = {name: hn.pack_context(context, assets, model, bridge, device)
             for name, context in contexts.items()}
    del model
    torch.cuda.empty_cache()
    for name, pack in packs.items():
        print(f"{name} candidate feature check {json.dumps(pack.verification)}")
        record["contexts"][name]["candidate_feature_check"] = pack.verification

    pool = hn.training_pool(packs, ("K562",))
    fitted = pd.fit(pool["features"], pool["responses"], pool["present"], pool["keys"], ("K562",))
    print(f"anchor predictor: rows {fitted.rows}, penalty {fitted.penalty:g}, "
          f"held-out cosine {fitted.held_out_cosine:.4f}, basis explains {fitted.basis.explained:.4f}")
    record["anchor_predictor"] = {"rows": fitted.rows, "penalty": fitted.penalty,
                                  "held_out_gene_space_cosine": fitted.held_out_cosine,
                                  "explained_energy": fitted.basis.explained}

    for name in directories:
        pack = packs[name]
        effect = fitted.effect(pack.features, pack.present)
        generator = np.random.default_rng(20260918)
        arms = {"BASE": pack.base, "EFFECT": hn.effect_arms(pack, effect),
                "RANDOM": [generator.standard_normal(pack.base[view].shape) for view in (0, 1)],
                "ORACLE": pack.utility}
        block = ev.evaluate_arms(arms, pack.utility, pack.ids, pack.self_positions,
                                 np.ones(pack.ids.size, dtype=bool))
        vectors = block.pop("_vectors")
        print(f"\n[{name}] {block['queries']} query views, pool {block['candidates']}")
        for arm, entry in block["arms"].items():
            print(f"  {arm:8s} MBRU {entry['MBRU']['median']:+.4f}  "
                  f"Mean@20 {entry['Mean@20']['median']:+.4f}")
        record["contexts"][name]["arms"] = block["arms"]
        record["contexts"][name]["queries"] = block["queries"]
        record["contexts"][name]["pool"] = block["candidates"]
        record["contexts"][name]["paired"] = {}
        for better, worse in (("EFFECT", "BASE"), ("BASE", "RANDOM")):
            delta = ev.paired(vectors[better]["MBRU"], vectors[worse]["MBRU"])
            record["contexts"][name]["paired"][f"{better}_minus_{worse}"] = delta
            low, high = delta["bootstrap"]["median"]["ci95"]
            print(f"  {better} - {worse}: {delta['median']:+.4f} [{low:+.4f},{high:+.4f}] "
                  f"improved {delta['fraction_query_improved']:.3f}")

    if args.output:
        record["runtime"] = {"timestamp": datetime.now(timezone.utc).isoformat(),
                             "python": platform.python_version(), "numpy": np.__version__}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
