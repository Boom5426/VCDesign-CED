# Project structure

The release has two layers: the frozen Python implementation of VCDesign-CED,
and compact source data and generators for the manuscript displays. Large public
input matrices and compute artifacts are outside the package; their checksums and
required roles are recorded under `configs/`.

```text
src/gene_open_inverse/   model, data builders, evaluation, baseline adaptations
examples/                synthetic software demonstration; no paper result
configs/                 portable asset template and frozen input checksums
paper_assets/            figure and table source data, generators, editable Figure 1
provenance/              source snapshot hashes and packaging transformations
tools/                   read-only package and processed-data audits
requirements.txt         reference environment, excluding hardware-specific PyTorch
pyproject.toml           installable package metadata and workflow extras
DATA_SOURCES.md          public input locations and required source variants
RELEASE_STATUS.md        remaining gates for an anonymous submission archive
```

The module hierarchy preserves the frozen research code so that experimental
definitions, seeds, data roles, and imports remain traceable. Each copied file's
source and release hash appears in `provenance/SOURCE_MAP.tsv`. Five Python input
defaults were changed to relative paths. The two knowledge-feature configuration
snapshots still contain unresolved research environment variables and are not
executable release configurations; see `provenance/TRANSFORMS.md` and
`RELEASE_STATUS.md`.

## Entry points

| Task | Module or file | Inputs |
|---|---|---|
| K562 preprocessing | `model/replogle_batch_disjoint_measurability_v1` | Public K562 screen |
| RPE1 preprocessing | `external_context_v1/build.py` | Public RPE1 screen |
| HepG2/Jurkat preprocessing | `four_context_v1/build.py` | GSE264667 screens |
| Candidate features | `model/candidate_knowledge_v1`, `model/mapkg_candidate_adjudication_v1` | STRING, MAP-KG, ESM-2 assets |
| Base scorer train/select | `final_clean_model_v1/train.py`, `select.py` | Frozen asset configuration |
| Candidate Effect Distillation | `candidate_effect_distillation_v1` | Training responses and knowledge |
| Held-context evaluation | `four_context_v1`, `open_vocab_generalization_v1` | Processed packs and frozen scorer |
| External baselines | `external_baselines_v1` | Public method software and frozen packs |
| Table and figure data | `paper_assets/figures/scripts`, `paper_assets/supplementary/scripts` | Bundled derived records |

The original research modules have several phase-specific command lines. Use
`python -m <module> --help` to inspect a stage, then supply explicit input and
new output paths. The full run-to-paper command index is still pending in
`RELEASE_STATUS.md`.
