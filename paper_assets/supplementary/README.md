# Supplementary information and source data

The compiled appendix in `appendix/` is the paper's Supplementary Information (SI). This directory
contains the machine-readable source-data package for the finalized external-baseline and evaluation
closure added to the main paper and SI.

## Canonical source

`data/external_baseline_closure_v1.json` is the single paper-facing source for the closure values. It
records the evaluation contract, metric definitions, levels, paired contrasts, PHR@20, compatibility
and reproduction outcomes, open-vocabulary retention, prediction-design analysis, verification gates,
and scope limits. Repository-relative provenance points to the authoritative result and decision log.

The CSV files are deterministic views generated from this JSON. Do not edit them independently.

```bash
python3 supplementary/scripts/export_external_closure.py \
  --source supplementary/data/external_baseline_closure_v1.json \
  --out-dir supplementary/data --check
```

Use `--write` instead of `--check` only after deliberately updating the canonical JSON.

## Paper and SI mapping

| Paper item | Machine-readable source |
|---|---|
| Main Table 1 and Figure 2a, solver levels | `external_baseline_levels.csv`; `../tables/tab_main.tex` |
| Main Figures 2b--d, 3a--b and 4a--c | `../figures/data/display_v2_data.json`; `../figures/data/display_v2_source_data.csv` (built by `../figures/scripts/build_display_v2_data.py` from `fig2_hero_data.json`, `fig2_source_data.csv`, `fig3_generalization_data.json`, `fig3_source_data.csv`, `fig4_diagnosis_data.json`, `figS_atlas_context_data.json` and the external levels and contrasts) |
| Row-level Figure 3 masking and mixed-deployment data | `../figures/data/fig3_source_data.csv`; `../figures/data/figure_3_source.json` (display-v1 panel letters: b, c are current 3a, 3b) |
| Row-level Figure 4 diagnostics | `../figures/data/fig4_source_data.csv`; `../figures/data/figure_4_source.json` (display-v1 panel letters: c, d are current 4b, 4c; current 4a is in `figS_atlas_context_data.json`) |
| SI unified external-baseline table | `external_baseline_levels.csv` |
| SI paired headline contrasts | `external_baseline_contrasts.csv` |
| SI PHR@20 table and powered contrast | `phr20_coverage.csv`; `phr20_levels.csv`; `phr20_contrasts.csv` |
| SI compatibility and reproduction audit | `compatibility_reproduction.csv` |
| SI CellNavi identification companion | `cellnavi_identification_companion.csv` |
| Open-vocabulary retention | `open_vocabulary_retention.csv` |
| Prediction-design analysis | `prediction_design_summary.csv` |

`data/SHA256SUMS` covers the canonical JSON, every generated CSV, this README, and the exporter.

## Scope and release handling

- This package exposes derived paper values, not the 83 GB model-training workspace.
- Score matrices, witness outputs, checkpoints, and the run-level immutable manifest remain governed
  by the authoritative closure record and must be routed deliberately into the anonymous artifact.
- Figure 4 submission-facing artifacts contain source roles and SHA-256 values, not machine-specific paths.
- Do not quote a manifest file count until the 343-versus-345 bookkeeping discrepancy is reconciled.
- The current availability status remains `draft_with_placeholders` until an anonymous repository or
  supplementary-upload locator is assigned and the full package passes identity-leakage review.
