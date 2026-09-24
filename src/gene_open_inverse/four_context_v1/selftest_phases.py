#!/usr/bin/env python3
"""Run Phases 8, 9, 10 and 12 to 17 end to end on synthetic contexts.

This asserts nothing about biology.  It exists because those phases are long, are
only reachable once every real context is built, and would otherwise be executed for
the first time on the real data, where a shape or key error costs hours and looks
like a result that failed to appear.  The synthetic contexts are built so that the
answers are known in direction: one candidate axis carries a planted static-knowledge
to effect map, so V1 must beat a random ranker, and the context residual is planted
with a known variance share, so the residual measurement must recover roughly that.

Run it whenever the phase code changes.  It is not part of the result record.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import contract as fc
from . import evaluate as ev
from . import harness as hn


def synthetic_packs(directory: Path, candidates: int = 400, genes: int = 320,
                    queries: int = 60, width: int = 24, residual_share: float = 0.2,
                    seed: int = 20260918) -> dict:
    """Four contexts sharing a candidate vocabulary and one planted effect map."""
    generator = np.random.default_rng(seed)
    ids = np.asarray([f"G{i:04d}" for i in range(candidates)], dtype=str)
    features = generator.standard_normal((candidates, width))
    shared_map = generator.standard_normal((width, genes))
    shared_effect = features @ shared_map

    packs = {}
    for index, name in enumerate(fc.CONTEXTS):
        # Each context sees a subset, so the strata are not degenerate.
        keep = np.sort(generator.choice(candidates, size=int(candidates * 0.85), replace=False))
        context_map = generator.standard_normal((width, genes))
        effect = ((1.0 - residual_share) * shared_effect[keep]
                  + residual_share * (features[keep] @ context_map))
        noise = 0.25 * generator.standard_normal((2, keep.size, genes))
        responses = (effect[None] + noise).astype(np.float32)
        sources = generator.standard_normal((2, keep.size, genes)).astype(np.float32) + index
        query_rows = np.sort(generator.choice(keep.size, size=queries, replace=False))
        base = [generator.standard_normal((queries, keep.size)) for _ in (0, 1)]
        utility = [np.ascontiguousarray(ev.cross_view_utility(responses, view)[query_rows])
                   for view in (0, 1)]
        fit_mask = np.ones(keep.size, dtype=bool)
        if name == fc.ANCHOR_CONTEXT:          # the anchor trains and is queried disjointly
            fit_mask = np.zeros(keep.size, dtype=bool)
            fit_mask[: int(keep.size * 0.7)] = True
            fit_mask[query_rows] = False
        pack = hn.ContextPack(
            name=name, ids=ids[keep], pool_rows=keep, query_rows=query_rows,
            self_positions=query_rows, responses=responses, sources=sources,
            query_responses=np.ascontiguousarray(responses[:, query_rows]),
            query_sources=np.ascontiguousarray(sources[:, query_rows]),
            features=features[keep], present=np.ones(keep.size, dtype=bool),
            base=base, utility=utility, fit_mask=fit_mask,
            verification={"synthetic": True})
        hn.save_pack(pack, directory)
        packs[name] = pack
    common = ids
    for pack in packs.values():
        common = np.intersect1d(common, pack.ids)
    (directory / "common4.json").write_text(json.dumps(
        {"identities": sorted(common.tolist()), "count": int(common.size)}, indent=2) + "\n")
    return packs


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end self test of the later phases")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--residual-share", dest="residual_share", type=float, default=0.2)
    args = parser.parse_args()
    workspace = Path(args.workspace)
    packs_dir = workspace / "packs"
    packs_dir.mkdir(parents=True, exist_ok=True)
    packs = synthetic_packs(packs_dir, residual_share=args.residual_share)
    print(f"synthetic packs: " + ", ".join(
        f"{name} {pack.ids.size} candidates, {int(pack.fit_mask.sum())} trainable, "
        f"{pack.query_rows.size} queries" for name, pack in packs.items()))

    from . import matched, residual, scaling, v2

    print("\n--- Phase 10, context residual ---")
    residual_record = residual.run(SimpleNamespace(
        packs=str(packs_dir), common4=str(packs_dir / "common4.json"),
        output=str(workspace / "residual.json")))
    primary = residual_record["unit_normalized_primary"]
    print(f"  residual variance share {primary['residual_variance_share']:.4f} "
          f"(planted {args.residual_share})")
    print(f"  measurement resolved: {primary['reliability']['resolved_above_control']}")
    print(f"  BUILD_V2 = {residual_record['verdict']['build_v2']}")

    print("\n--- Phases 6 to 8, matched controls ---")
    matched_record = matched.run(SimpleNamespace(
        packs=str(packs_dir), common4=str(packs_dir / "common4.json"),
        output=str(workspace / "matched.json")))
    for held, fold in matched_record["folds"].items():
        summary = fold["row_matched"].get("N1[" + fold["training"][0] + "]", {}).get("summary", {})
        print(f"  held {held}: budget {fold['row_budget']}, "
              f"single-context delta {summary.get('ALL', {}).get('mean')}")

    print("\n--- Phases 12 to 17, context-conditioned V2 ---")
    v2_record = v2.run(SimpleNamespace(
        packs=str(packs_dir), common4=str(packs_dir / "common4.json"),
        output=str(workspace / "v2.json")))
    for held, fold in v2_record["folds"].items():
        block = fold["strata"].get("ALL", {}).get("paired", {})
        item = block.get("V2_minus_V1", {})
        print(f"  held {held}: penalty {fold['penalty_selection']['selected_penalty']:g}, "
              f"V2 - V1 {item.get('median')}")
    print(f"  PASS criteria: {json.dumps(v2_record['pass_criteria'])}")

    print("\n--- Phase 9, scaling summary ---")
    scaling_record = scaling.run(SimpleNamespace(
        loco=str(workspace / "fake_loco.json"), matched=str(workspace / "matched.json"),
        output=str(workspace / "scaling.json"))) if (workspace / "fake_loco.json").exists() else None
    if scaling_record is None:
        (workspace / "fake_loco.json").write_text(json.dumps({"folds": {}}) + "\n")
        scaling_record = scaling.run(SimpleNamespace(
            loco=str(workspace / "fake_loco.json"), matched=str(workspace / "matched.json"),
            output=str(workspace / "scaling.json")))
    print(f"  headline {scaling_record.get('headline')}")
    print("\nall phases completed without error")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
