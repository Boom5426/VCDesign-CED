#!/usr/bin/env python3
"""Phase 3: freeze the four-context gene axis.

``G_4`` is the intersection of the four measured axes, computed once, before any
model in this programme runs.  Genes are never selected on a downstream result and
the axis is never revisited after a number exists.

The K562 historical V1 result keeps its own 8248-gene axis and is not recomputed.
This is a separate experiment.

The report answers the question that decides whether ``G_4`` is usable at all: how
much of each context's measured response energy survives the restriction.  A common
axis that keeps most genes but little energy would make every downstream cosine a
comparison of leftovers.
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import contexts as cx
from . import contract as fc


def _axis(spec: str) -> tuple[str, Path]:
    name, path = spec.split("=", 1)
    return name, Path(path)


def energy_retained(responses: np.ndarray, usable: np.ndarray, columns: np.ndarray) -> dict:
    """Fraction of squared response energy that lies on the retained columns."""
    rows = np.flatnonzero(usable)
    total, kept, per_row = 0.0, 0.0, []
    for start in range(0, rows.size, 512):
        block = np.asarray(responses[:, rows[start:start + 512]], dtype=np.float64)
        whole = (block * block).sum(axis=2)
        part = (block[:, :, columns] ** 2).sum(axis=2)
        total += float(whole.sum())
        kept += float(part.sum())
        per_row.append((part / np.maximum(whole, 1e-30)).ravel())
    fractions = np.concatenate(per_row) if per_row else np.zeros(0)
    return {"pooled": kept / max(total, 1e-30),
            "per_identity_median": float(np.median(fractions)) if fractions.size else 0.0,
            "per_identity_p05": float(np.quantile(fractions, 0.05)) if fractions.size else 0.0}


def run(args: argparse.Namespace) -> dict:
    axes, directories = {}, {}
    for spec in args.context:
        name, path = _axis(spec)
        directories[name] = path
        axes[name] = np.asarray([str(v) for v in np.load(path / "gene_axis.npy", allow_pickle=True)])
    if args.anchor_genes:
        axes[fc.ANCHOR_CONTEXT] = np.asarray(
            [str(v) for v in np.load(args.anchor_genes, allow_pickle=True)])
    if set(axes) != set(fc.CONTEXTS):
        raise RuntimeError(f"expected exactly {fc.CONTEXTS}, received {sorted(axes)}")

    common = axes[fc.CONTEXTS[0]]
    for name in fc.CONTEXTS[1:]:
        common = np.intersect1d(common, axes[name])
    common = np.sort(common)

    record: dict = {
        "schema": "VCDESIGN_FOUR_CONTEXT_V1_GENE_AXIS",
        "contexts": list(fc.CONTEXTS), "selected_on_a_result": False,
        "G_4_size": int(common.size),
        "per_context": {},
        "pairwise_shared": {},
        "anchor_axis_note": "the historical K562 V1 result keeps its 8248-gene axis and is not recomputed",
        "anchor_G_check_read": False,
    }
    for name in fc.CONTEXTS:
        record["per_context"][name] = {
            "measured_genes": int(axes[name].size),
            "retained_fraction_of_axis": float(common.size / axes[name].size)}
    for left in fc.CONTEXTS:
        for right in fc.CONTEXTS:
            if left < right:
                record["pairwise_shared"][f"{left}|{right}"] = int(
                    np.intersect1d(axes[left], axes[right]).size)

    for name, path in directories.items():
        responses = np.load(path / "side_responses.npy", mmap_mode="r")
        usable = np.load(path / "usable.npy")
        ids = np.asarray([str(v) for v in np.load(path / "identity_axis.npy", allow_pickle=True)])
        position = {gene: index for index, gene in enumerate(axes[name].tolist())}
        columns = np.asarray([position[gene] for gene in common.tolist()], dtype=np.int64)
        record["per_context"][name]["response_energy_retained"] = energy_retained(
            responses, usable, columns)
        measured = np.isin(ids, common)
        record["per_context"][name]["candidate_target_gene_coverage"] = {
            "identities": int(ids.size),
            "target_gene_in_G_4": int(measured.sum()),
            "fraction": float(measured.mean()),
            "usable_and_target_in_G_4": int((measured & usable).sum())}
        del responses

    if args.anchor_config:
        # The anchor is loaded through the same audited path the rest of the programme
        # uses, which drops every G_check row before any array exists.  Reading the raw
        # response cache here instead would compute this statistic over the sealed
        # split, which the mission forbids, and the fact that it is only an aggregate
        # does not make it not a read.
        from ..model.decision_alignment_v1.assets import AuditConfig
        from ..model.global_objective_knowledge_attribution_v1.assets import GlobalAttributionAssets

        config = AuditConfig.load(Path(args.anchor_config))
        assets = GlobalAttributionAssets(modalities=("STRING", "MAPKG", "TEXT"),
                                         **config.asset_arguments())
        anchor = cx.load_anchor(assets, args.anchor_sources, args.anchor_genes, args.eligibility)
        position = {gene: index for index, gene in enumerate(anchor.genes.tolist())}
        columns = np.asarray([position[gene] for gene in common.tolist()], dtype=np.int64)
        entry = record["per_context"][fc.ANCHOR_CONTEXT]
        entry["response_energy_retained"] = energy_retained(anchor.responses, anchor.usable, columns)
        measured = np.isin(anchor.ids, common)
        entry["candidate_target_gene_coverage"] = {
            "identities": int(anchor.ids.size), "target_gene_in_G_4": int(measured.sum()),
            "fraction": float(measured.mean()),
            "usable_and_target_in_G_4": int((measured & anchor.usable).sum())}
        entry["G_check_rows_read"] = False
        entry["reachable_identities"] = int(anchor.ids.size)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "G_4.npy", common)
    record["runtime"] = {"timestamp": datetime.now(timezone.utc).isoformat(),
                         "python": platform.python_version(), "numpy": np.__version__}
    (output / "FOUR_CONTEXT_GENE_AXIS.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3: freeze G_4")
    parser.add_argument("--context", action="append", required=True, help="NAME=build_directory")
    parser.add_argument("--anchor-genes", dest="anchor_genes", default=None)
    parser.add_argument("--anchor-config", dest="anchor_config", default=None,
                        help="asset config; the anchor is read through the G_check-free loader")
    parser.add_argument("--anchor-sources", dest="anchor_sources", default=None)
    parser.add_argument("--eligibility", default=None)
    parser.add_argument("--output", required=True)
    record = run(parser.parse_args())
    print(f"G_4 = {record['G_4_size']} genes")
    for name, entry in record["per_context"].items():
        energy = entry.get("response_energy_retained", {})
        coverage = entry.get("candidate_target_gene_coverage", {})
        print(f"  {name:8s} axis {entry['measured_genes']:6d}  retained {entry['retained_fraction_of_axis']:.4f}"
              f"  energy {energy.get('pooled', float('nan')):.4f}"
              f"  target-in-G4 {coverage.get('fraction', float('nan')):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
