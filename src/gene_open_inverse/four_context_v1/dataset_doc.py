#!/usr/bin/env python3
"""Assemble the four-context dataset audit document from the records on disk.

Every number is read from a JSON artifact rather than transcribed, so the document
regenerates and cannot drift from the run that produced it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: str | None) -> dict:
    return json.loads(Path(path).read_text()) if path and Path(path).exists() else {}


def render(audits: dict, builds: dict, gene_axis: dict, manifest: dict,
           verifications: dict) -> str:
    lines: list[str] = ["# Four-context dataset audit", ""]
    lines.append("K562, RPE1, HepG2 and Jurkat. This document records what the data are, "
                 "not what any model does with them. No filtering decision here depends on "
                 "a downstream result.")
    lines.append("")

    if manifest:
        lines.append("## Provenance")
        lines.append("")
        lines.append(f"Accession `{manifest.get('accession')}`, "
                     f"{manifest.get('series_title', '')}.")
        lines.append("")
        lines.append("| file | bytes | matches server | sha256 | downloaded |")
        lines.append("| --- | --- | --- | --- | --- |")
        for name, entry in manifest.get("files", {}).items():
            if not entry.get("present"):
                lines.append(f"| {name} | absent | | | |")
                continue
            lines.append(f"| `{name}` | {entry['bytes']} | {entry['size_matches_server']} | "
                         f"`{entry['sha256'][:16]}...` | {entry['downloaded_at'][:19]} |")
        lines.append("")
        lines.append(f"FASTQ reprocessing performed: {manifest.get('fastq_reprocessing_performed')}. "
                     f"{manifest.get('fastq_reprocessing_note', '')}")
        lines.append("")

    lines.append("## Verdicts")
    lines.append("")
    lines.append("| context | verdict | failed conditions |")
    lines.append("| --- | --- | --- |")
    for name, record in audits.items():
        failed = ", ".join(f"`{f}`" for f in record.get("failed_conditions", [])) or "none"
        lines.append(f"| {name} | **`{record.get('verdict', 'UNKNOWN')}`** | {failed} |")
    lines.append("")

    lines.append("## Measurement contract, as built")
    lines.append("")
    lines.append("| context | cells retained | genes | batches | A/B | control cells | "
                 "nominal | usable | eligible | four-way |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for name, record in builds.items():
        source = record.get("source", {})
        batches = record.get("batches", {})
        identities = record.get("identities", {})
        lines.append(
            f"| {name} | {source.get('n_obs_retained', source.get('n_obs'))} | "
            f"{source.get('n_vars')} | {batches.get('total')} | "
            f"{batches.get('side_A')}/{batches.get('side_B')} | {batches.get('control_cells')} | "
            f"{identities.get('nominal')} | {identities.get('usable')} | "
            f"{record.get('query_gate', {}).get('eligible')} | "
            f"{record.get('four_way', {}).get('usable_and_four_way')} |")
    lines.append("")

    if verifications:
        lines.append("## Independent verification")
        lines.append("")
        lines.append("All contexts are verified under the frozen "
                     "`RESPONSE_ASSET_VERIFIER_CONTRACT_V1_FLOAT32_STORAGE_AWARE`, with no "
                     "per-context tolerance. Each run also carries a deliberately corrupted "
                     "asset witness; a run whose witness does not FAIL is itself a failure.")
        lines.append("")
        lines.append("| context | verdict | worst check, as a fraction of its bound | "
                     "witness fires |")
        lines.append("| --- | --- | --- | --- |")
        for name, record in verifications.items():
            ratios = [v.get("max_ratio_to_bound") for v in record.get("checks", {}).values()
                      if v.get("max_ratio_to_bound") is not None]
            worst = f"{max(ratios):.2f}" if ratios else "exact only"
            witness = record.get("corruption_witness", {}).get("witness_passed")
            lines.append(f"| {name} | **{record.get('verdict')}** | {worst} | {witness} |")
        lines.append("")

    if gene_axis:
        lines.append("## The frozen four-context gene axis")
        lines.append("")
        lines.append(f"`G_4` holds **{gene_axis.get('G_4_size')}** genes, taken once as the "
                     "intersection of the four measured axes, before any model in this "
                     "programme ran. Genes are never selected on a result. The historical "
                     "K562 V1 result keeps its 8248-gene axis and is not recomputed; this is "
                     "a separate experiment.")
        lines.append("")
        lines.append("| context | measured genes | retained | response energy retained | "
                     "target gene in G_4 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for name, entry in gene_axis.get("per_context", {}).items():
            energy = entry.get("response_energy_retained", {})
            coverage = entry.get("candidate_target_gene_coverage", {})
            lines.append(f"| {name} | {entry.get('measured_genes')} | "
                         f"{entry.get('retained_fraction_of_axis', 0):.4f} | "
                         f"{energy.get('pooled', float('nan')):.4f} | "
                         f"{coverage.get('fraction', float('nan')):.4f} |")
        lines.append("")
        lines.append("| pair | shared genes |")
        lines.append("| --- | --- |")
        for pair, count in gene_axis.get("pairwise_shared", {}).items():
            lines.append(f"| {pair} | {count} |")
        lines.append("")

    for name, record in audits.items():
        dual = record.get("dual_guide", {})
        if not dual.get("column"):
            continue
        verdict = dual["case_verdict"]
        lines.append(f"## {name}: dual-sgRNA identity")
        lines.append("")
        lines.append(f"Case **{verdict['case']}**, dominant label shape "
                     f"`{verdict['dominant_label_shape']}` at "
                     f"{verdict['dominant_label_fraction']:.4f} of labels.")
        lines.append("")
        lines.append("| pair shape | labels |")
        lines.append("| --- | --- |")
        for key, value in dual["pair_shape"]["label_counts"].items():
            lines.append(f"| {key} | {value} |")
        lines.append("")
        lines.append(f"Single-gene identities after excluding cross-gene and unparsed pairs: "
                     f"**{verdict['single_gene_identities_after_exclusion']}** against a "
                     f"required minimum of {verdict['minimum_required']}. "
                     f"Agreement with the file's own perturbation column: "
                     f"{dual.get('agreement_with_perturbation_column', 0):.4f}.")
        unparsed = verdict.get("unparsed_pair_shape") or {}
        if unparsed:
            lines.append("")
            lines.append(f"Of the {unparsed.get('unparsed_labels')} labels that did not "
                         f"resolve, {unparsed.get('identical_leading_token')} carry the same "
                         f"leading token in both halves and are therefore same-gene pairs "
                         f"whose symbol the universe does not carry, and "
                         f"{unparsed.get('differing_leading_token')} do not. So the labels "
                         f"that could be cross-gene anywhere in this context number "
                         f"**{verdict.get('labels_that_could_be_cross_gene')}**, and "
                         f"`no_cross_gene_pair_anywhere` is "
                         f"**{verdict.get('no_cross_gene_pair_anywhere')}**. Without this "
                         f"check, \"no cross-gene pairs\" would have been a claim about only "
                         f"the labels that parsed, and an unresolved symbol is exactly where "
                         f"a combination could hide.")
        lines.append("")

    lines.append("## What these numbers may not be used for")
    lines.append("")
    lines.append("Candidate pool sizes differ several-fold between contexts. Budget metrics "
                 "depend on pool size, so absolute MBRU is not comparable across contexts and "
                 "cannot be used to rank how hard a context is. Only paired differences "
                 "between arms inside one context, under one base ranker, are comparable.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble the four-context dataset audit")
    parser.add_argument("--audit", action="append", default=[], help="NAME=path.json")
    parser.add_argument("--build", action="append", default=[], help="NAME=path.json")
    parser.add_argument("--verification", action="append", default=[], help="NAME=path.json")
    parser.add_argument("--gene-axis", dest="gene_axis", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    def collect(items):
        out = {}
        for item in items:
            name, path = item.split("=", 1)
            out[name] = _load(path)
        return out

    text = render(collect(args.audit), collect(args.build), _load(args.gene_axis),
                  _load(args.manifest), collect(args.verification))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(text)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
