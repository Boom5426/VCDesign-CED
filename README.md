# VCDesign-CED

### Candidate-conditioned inverse modeling for cellular intervention design

VCDesign ranks candidate cellular interventions for a requested transition under
a finite experimental budget. VCDesign-CED augments a direct candidate scorer
with Candidate Effect Distillation: it predicts the response of an unmeasured
candidate from static biological knowledge and scores its alignment with the
requested transition.

[Paper](paper/VCDesign.pdf) · [Processed data and frozen artifacts](https://huggingface.co/datasets/Boom5426/VCDesign) · [Project structure](PROJECT_STRUCTURE.md) · [Reproduction guide](REPRODUCE.md)

## Highlights

- Candidate-conditioned intervention ranking over variable candidate sets.
- Static STRING and MAP-KG knowledge for candidates without measured responses.
- K562, RPE1, HepG2, and Jurkat Perturb-seq evaluation settings.
- Frozen processed inputs, checkpoints, and run records released separately from
  the source code.

## Quick start

Python 3.11 and a PyTorch 2.4 build appropriate for the local CUDA system are
required. From the repository root:

```bash
python3 -m pip install -e ".[model,data,dev]"
python3 examples/ced_demo.py
python3 -m pytest -q
```

The demo uses synthetic data and checks response-basis construction, ridge effect
prediction, and score fusion. It is a software smoke test, not a biological
result.

## Run with released data

The processed data, frozen feature matrices, reference checkpoint, and run
records are hosted in the companion Hugging Face dataset. Download them into the
repository root, then verify every released byte before running the model:

```bash
python3 -m pip install "huggingface_hub>=0.35"
hf download Boom5426/VCDesign --repo-type dataset --local-dir . \
  --include 'inputs/**' 'checkpoints/**' 'records/**' 'DATA_MANIFEST.json'
python3 tools/verify_processed_data.py --root .
```

The core training configuration is `configs/paper_run_v1.json`. It resolves
inputs relative to the repository root and records the selected epoch-8 reference
checkpoint. See [REPRODUCE.md](REPRODUCE.md) for the training command and
expected artifacts.

## Data sources

The processed release is derived from public K562 and RPE1 Perturb-seq screens,
GEO GSE264667 HepG2/Jurkat screens, STRING v12.0, MAP-KG, and ESM-2. Exact
source versions, required variants, and citations are listed in
[DATA_SOURCES.md](DATA_SOURCES.md).

## Repository layout

```text
src/gene_open_inverse/   VCDesign-CED models, preprocessing, and evaluation code
configs/                 portable and frozen run configurations
examples/                synthetic end-to-end demo
tools/                   processed-data integrity checker
paper/                   final manuscript PDF
```

## Notes

The processed data release includes the bytes needed to validate the frozen
training inputs and run the primary training pipeline. Large raw H5AD source
matrices remain available from their original public repositories. The final
manuscript documents the experimental protocol and results.
