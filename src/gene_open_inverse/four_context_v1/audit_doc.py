#!/usr/bin/env python3
"""Render a Phase 1 audit record into its dataset audit document.

The numbers are transcribed by code rather than by hand.  Hand-copying a table from
a JSON record into a Markdown document is a real and recurring source of error in
this programme, and a document whose numbers cannot be regenerated from the record is
not evidence of anything.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _row(label: str, value) -> str:
    return f"| {label} | {value} |"


def render(record: dict) -> str:
    context = record["context"]
    lines: list[str] = []
    lines.append(f"# {context.upper()} data audit")
    lines.append("")
    lines.append(f"**Verdict: `{record.get('verdict', 'UNKNOWN')}`**")
    failed = record.get("failed_conditions") or []
    if failed:
        lines.append("")
        lines.append("Failed hard conditions: " + ", ".join(f"`{name}`" for name in failed) + ".")
    lines.append("")
    lines.append(f"Accession `{record.get('accession', 'GSE264667')}`, source "
                 f"`{Path(record['source']).name}`. This audit reads data properties only. "
                 "No model was run and no filtering here depends on any downstream result.")
    lines.append("")

    lines.append("## Resolved fields")
    lines.append("")
    lines.append("| field | column |")
    lines.append("| --- | --- |")
    for name, value in record.get("resolved_columns", {}).items():
        lines.append(_row(name, f"`{value}`" if value else "not found"))
    lines.append(_row("dual-guide column", f"`{record.get('dual_guide', {}).get('column')}`"))
    lines.append("")

    lines.append("## Matrix and axes")
    lines.append("")
    lines.append("| quantity | value |")
    lines.append("| --- | --- |")
    lines.append(_row("cells", record["cells"]))
    lines.append(_row("genes", record["genes"]))
    lines.append(_row("layers", record.get("layers") or "none"))
    lines.append("")

    axis = record.get("gene_axis", {})
    candidates = axis.get("symbol_axis_candidates", {})
    best = axis.get("best_symbol_axis")
    if candidates:
        lines.append("### Which axis carries gene symbols")
        lines.append("")
        lines.append("A release keyed by ENSEMBL identifier intersects a symbol axis in "
                     "nothing, and that failure would read as \"these contexts share no "
                     "genes\" rather than as a key mismatch. Every candidate axis is scored "
                     "against the reference axes here and the chosen one is recorded.")
        lines.append("")
        lines.append("| candidate axis | distinct | best overlap with a reference | example |")
        lines.append("| --- | --- | --- | --- |")
        for name, entry in sorted(candidates.items(),
                                  key=lambda kv: -(kv[1].get("best_reference_overlap") or 0)):
            mark = " **(chosen)**" if name == best else ""
            example = entry["examples"][0] if entry.get("examples") else ""
            lines.append(f"| `{name}`{mark} | {entry['distinct']} | "
                         f"{entry.get('best_reference_overlap', 0)} | `{example}` |")
        lines.append("")
        chosen = candidates.get(best, {})
        duplicates = int(record.get("genes", 0)) - int(chosen.get("distinct", record.get("genes", 0)))
        lines.append(f"`var_names` are symbols: **{axis.get('var_names_are_symbols')}**. "
                     f"The chosen axis `{best}` carries **{duplicates}** duplicated symbol"
                     f"{'' if duplicates == 1 else 's'}. Duplicated symbols are dropped at "
                     "build time, both copies: the axis is addressed by name downstream, so a "
                     "repeated symbol means one column silently wins and nothing records which "
                     "gene was intended.")
        lines.append("")

    semantics = record.get("count_semantics", {})
    if semantics:
        lines.append("### Count matrix semantics")
        lines.append("")
        lines.append("The inherited recipe normalizes to a library size of 1e4 and takes "
                     "`log1p`, which is only meaningful on raw counts. Applying it to an "
                     "already normalized matrix would produce a response that is not the "
                     "quantity the other contexts measure, and nothing downstream would "
                     "notice.")
        lines.append("")
        lines.append("| quantity | value |")
        lines.append("| --- | --- |")
        lines.append(_row("cells sampled", semantics.get("cells_sampled")))
        lines.append(_row("stored sparse", semantics.get("sparse")))
        lines.append(_row("nonzero fraction", f"{semantics.get('nonzero_fraction', 0):.4f}"))
        lines.append(_row("min, max", f"{semantics.get('min')}, {semantics.get('max')}"))
        lines.append(_row("any negative value", semantics.get("any_negative")))
        lines.append(_row("nonzero values that are integers",
                          f"{semantics.get('fraction_of_nonzero_values_that_are_integers', 0):.6f}"))
        lines.append(_row("library size median",
                          semantics.get("library_size", {}).get("median")))
        lines.append(_row("library size coefficient of variation",
                          f"{semantics.get('library_size', {}).get('coefficient_of_variation', 0):.4f}"))
        lines.append(_row("**is raw counts**", f"**{semantics.get('is_raw_counts')}**"))
        lines.append("")

    control = record.get("control", {})
    batches = record.get("batches", {})
    lines.append("## Controls and batch structure")
    lines.append("")
    lines.append("| quantity | value |")
    lines.append("| --- | --- |")
    lines.append(_row("non-targeting labels", ", ".join(f"`{v}`" for v in control.get("labels", [])[:6])))
    lines.append(_row("control cells", control.get("cells")))
    lines.append(_row("control cell fraction", f"{control.get('fraction_of_cells', 0):.4f}"))
    lines.append(_row("batch column", f"`{batches.get('column')}`"))
    lines.append(_row("batches", batches.get("count")))
    lines.append(_row("batches without controls", batches.get("batches_without_controls")))
    lines.append(_row("cells per batch, median", batches.get("cells_per_batch", {}).get("median")))
    lines.append(_row("control cells per batch, median",
                      batches.get("control_cells_per_batch", {}).get("median")))
    lines.append("")

    dual = record.get("dual_guide", {})
    if dual.get("column"):
        shape = dual["pair_shape"]["label_counts"]
        fractions = dual["pair_shape"]["label_fractions"]
        summary = dual["summary"]
        verdict = dual["case_verdict"]
        lines.append("## Dual-sgRNA identity")
        lines.append("")
        lines.append(f"Case **{verdict['case']}**. The intervention unit is a property of the "
                     "guide pair, so it is resolved explicitly here and the builder is given "
                     "the resolution rather than allowed to infer it.")
        lines.append("")
        lines.append("| pair shape | distinct labels | fraction of labels |")
        lines.append("| --- | --- | --- |")
        for key in ("same_gene_pair", "gene_plus_ntc", "ntc_plus_ntc", "geneA_plus_geneB",
                    "unparsed", "single_part_label"):
            lines.append(f"| {key} | {shape.get(key, 0)} | {fractions.get(key, 0.0):.4f} |")
        lines.append("")
        lines.append("| cell-level kind | cells | fraction |")
        lines.append("| --- | --- | --- |")
        for key, value in summary["cell_count"].items():
            lines.append(f"| {key} | {value} | {summary['cell_fraction'][key]:.4f} |")
        lines.append("")
        lines.append("| quantity | value |")
        lines.append("| --- | --- |")
        lines.append(_row("single-gene identities after exclusion",
                          verdict["single_gene_identities_after_exclusion"]))
        lines.append(_row("minimum required", verdict["minimum_required"]))
        lines.append(_row("cross-gene labels excluded", verdict["cross_gene_labels_excluded"]))
        lines.append(_row("unparsed labels excluded", verdict["unparsed_labels_excluded"]))
        lines.append(_row("agreement with the file's own perturbation column",
                          f"{dual.get('agreement_with_perturbation_column', 0):.4f}"))
        lines.append("")
        unparsed = verdict.get("unparsed_pair_shape") or {}
        if unparsed:
            lines.append("Unparsed labels are checked separately for hidden combinations, "
                         "because \"no cross-gene pairs\" would otherwise be a claim about only "
                         "the labels that parsed, and an unresolved symbol is exactly where a "
                         "combination could hide. A guide name leads with its target, so "
                         "comparing the leading token of the two halves settles it without "
                         "needing to recognise the symbol. Aliases the universe does not carry "
                         "land here.")
            lines.append("")
            lines.append("| quantity | value |")
            lines.append("| --- | --- |")
            lines.append(_row("unparsed labels", unparsed.get("unparsed_labels")))
            lines.append(_row("identical leading token, so same-gene",
                              unparsed.get("identical_leading_token")))
            lines.append(_row("differing leading token, so possibly cross-gene",
                              unparsed.get("differing_leading_token")))
            lines.append(_row("**labels that could be cross-gene anywhere**",
                              f"**{verdict.get('labels_that_could_be_cross_gene')}**"))
            lines.append(_row("**no cross-gene pair anywhere**",
                              f"**{verdict.get('no_cross_gene_pair_anywhere')}**"))
            lines.append("")
        lines.append("Cross-gene pairs are excluded from the primary single-gene programme and "
                     "counted. They are never relabelled as one of their two genes: doing so "
                     "would put a second perturbation's response into the first gene's measured "
                     "effect, and every distillation target downstream would inherit it.")
        lines.append("")
        for kind, examples in summary.get("example_labels", {}).items():
            if examples:
                lines.append(f"- `{kind}` examples: " + ", ".join(f"`{v}`" for v in examples))
        lines.append("")

    view = record.get("view_structure", {})
    lines.append("## Measurement views under the inherited contract")
    lines.append("")
    lines.append(f"Rule: `{view.get('rule')}`.")
    lines.append("")
    lines.append("| quantity | value |")
    lines.append("| --- | --- |")
    lines.append(_row("batches on side A", view.get("batches_side_A")))
    lines.append(_row("batches on side B", view.get("batches_side_B")))
    lines.append(_row("batches per quarter", view.get("batches_per_quarter")))
    lines.append(_row("minimum cells per view", view.get("min_view_cells")))
    lines.append(_row("cells per identity on side A, median",
                      view.get("cells_per_identity_side_A", {}).get("median")))
    lines.append(_row("cells per identity on side B, median",
                      view.get("cells_per_identity_side_B", {}).get("median")))
    lines.append(_row("usable identities", view.get("usable_identities")))
    lines.append(_row("four-way eligible identities", view.get("four_way_eligible")))
    lines.append("")

    shared = record.get("shared_with_reference_axes")
    if shared:
        lines.append("## Shared gene axis")
        lines.append("")
        lines.append("| against | shared genes | fraction of this context |")
        lines.append("| --- | --- | --- |")
        for name, entry in shared.items():
            lines.append(f"| {name} | {entry['shared_genes']} | "
                         f"{entry['fraction_of_this_context']:.4f} |")
        lines.append("")

    lines.append("## Hard qualification")
    lines.append("")
    lines.append("| condition | result |")
    lines.append("| --- | --- |")
    for name, value in record.get("qualification", {}).items():
        mark = "not evaluated" if value is None else ("pass" if value else "**FAIL**")
        lines.append(f"| `{name}` | {mark} |")
    lines.append("")
    lines.append(record.get("qualification_note", ""))
    lines.append("")
    lines.append(f"Generated from `{record.get('_record_path', 'the audit record')}` at "
                 f"{record.get('runtime', {}).get('timestamp', 'unknown time')}.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a dataset audit document")
    parser.add_argument("--record", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    record = json.loads(Path(args.record).read_text())
    record["_record_path"] = args.record
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(render(record))
    print(f"wrote {args.output} with verdict {record.get('verdict')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
