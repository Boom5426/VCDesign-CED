# Manuscript result and display index

The active manuscript uses the compact section files. The mappings below follow
the active Figure 2 to Figure 4 captions and the included appendix tables. The
display-data builder checks registered numerical values before writing output.

| Manuscript item | Packaged numerical source | Generator or layout source |
|---|---|---|
| Figure 1, model overview | Conceptual diagram | `figures/Fig1_latest_editable.pptx` and preview PNG |
| Figure 2, selection performance | `figures/data/display_v2_data.json`, `display_v2_source_data.csv` | `figures/scripts/plot_fig2_v2.py` |
| Figure 3, response availability and masking | Same display-v2 sources; upstream `fig3_source_data.csv` | `figures/scripts/plot_fig3_v2.py` |
| Figure 4, limits and prediction-design analysis | Same display-v2 sources; upstream `fig4_diagnosis_data.json`, `figS_atlas_context_data.json` | `figures/scripts/plot_fig4_v2.py` |
| Main solver taxonomy table | `tables/tab_solvers.tex` | Authored table body |
| Appendix result tables | `tables/tab_main.tex`, `tab_masking.tex`, `tab_ablation.tex`, `tab_context.tex`, `tab_external.tex` | Table bodies; external levels and contrasts in `supplementary/data/` |
| External-baseline and PHR supplementary results | `supplementary/data/external_baseline_closure_v1.json` | `supplementary/scripts/export_external_closure.py` |

`figures/scripts/build_display_v2_data.py` joins the registered row-level inputs
to build both display-v2 files. It uses 10,000 query-gene bootstrap replicates
with seed 20260916 for the held-context contrasts, as specified by the paper.
The current package includes its frozen inputs, so the numerical display data
can be rebuilt without the large training workspace.

Run from `paper_assets/` and use new output paths:

```bash
python3 figures/scripts/build_display_v2_data.py \
  --output /tmp/vcdesign-display-v2.json \
  --source-csv /tmp/vcdesign-display-v2.csv
python3 figures/scripts/plot_fig2_v2.py \
  --data /tmp/vcdesign-display-v2.json --out /tmp/vcdesign-fig2
python3 figures/scripts/plot_fig3_v2.py \
  --data /tmp/vcdesign-display-v2.json --out /tmp/vcdesign-fig3
python3 figures/scripts/plot_fig4_v2.py \
  --data /tmp/vcdesign-display-v2.json --out /tmp/vcdesign-fig4
```

The paper-level numerical records are traceable here; full training and
external-baseline runs require the still-pending input assets and run manifests
listed in `../RELEASE_STATUS.md`.
