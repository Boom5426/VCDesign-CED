# Project structure

```text
src/gene_open_inverse/
├── candidate_effect_distillation_v1/  Candidate Effect Distillation
├── final_clean_model_v1/              final base scorer, selection, locked evaluation
├── four_context_v1/                   K562/RPE1/HepG2/Jurkat data contracts
├── external_baselines_v1/             comparator adapters and reporting
├── open_vocab_generalization_v1/      masking and open-vocabulary analyses
└── model/                             frozen knowledge features and model components

configs/                               portable asset paths and paper-run configuration
examples/                              synthetic CED demonstration
tools/                                 processed-data SHA-256 verification
paper/                                 final manuscript PDF
```

## Main entry points

| Task | Entry point |
|---|---|
| Synthetic end-to-end check | `python3 examples/ced_demo.py` |
| Protocol tests | `python3 -m pytest -q` |
| Train the final base scorer | `python3 -m gene_open_inverse.final_clean_model_v1.train` |
| Candidate Effect Distillation | `python3 -m gene_open_inverse.candidate_effect_distillation_v1.run` |
| Four-context processing | `python3 -m gene_open_inverse.four_context_v1.build` |
| External-baseline reports | `python3 -m gene_open_inverse.external_baselines_v1.tables` |

Use explicit input and new output paths for all long-running stages. The complete
command sequence is in [REPRODUCE.md](REPRODUCE.md).
