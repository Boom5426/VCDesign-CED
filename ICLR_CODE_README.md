# VCDesign-CED: anonymous ICLR 2027 code supplement

This archive contains the source code, frozen configuration, synthetic smoke example, and protocol tests for **VCDesign: Candidate-Conditioned Inverse Modeling for Cellular Intervention Design**.

## Scope of this archive

Included:

- the `gene_open_inverse` Python source tree;
- the frozen paper-run configuration;
- a synthetic, data-free VCDesign-CED example;
- protocol tests and processed-data integrity verification; and
- exact Python package versions for a CPU verification environment.

Not included:

- Git metadata or commit history;
- the manuscript PDF, author information, or public author-account links;
- raw public H5AD matrices;
- large processed inputs, checkpoints, or cached run records; or
- outputs from local executions.

The manuscript and review submission provide the experimental definitions and reported results. Large inputs should be supplied as a separate anonymous artifact when required for review.

## Fast verification

Python 3.11 is required. From the archive root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python -m pip install -e . --no-deps
python examples/ced_demo.py
python -m pytest -q
```

The expected test result for this release is `162 passed`. The demo prints five ranked synthetic candidate indices and confirms that candidates missing all supported knowledge modalities receive a zero raw effect score. These are software checks, not biological results.

## Reproducing the primary training run

Place the separately supplied processed-data artifact at the archive root with this layout:

```text
DATA_MANIFEST.json
inputs/
checkpoints/
records/
```

Verify it before use:

```bash
python tools/verify_processed_data.py --root .
```

Then train with an explicit, new output directory:

```bash
python -m gene_open_inverse.final_clean_model_v1.train \
  --config configs/paper_run_v1.json \
  --output outputs/final_clean_train \
  --device cuda
```

The training command refuses to overwrite an existing output directory. The configuration resolves all assets relative to the archive root. The training implementation imports the frozen optimizer, scheduler, loss, sampler, batch size, seed, horizon, and objective schedule from the recorded training module.

## Main entry points

| Task | Entry point |
| :--- | :--- |
| Synthetic end-to-end check | `python examples/ced_demo.py` |
| Protocol tests | `python -m pytest -q` |
| Verify processed data | `python tools/verify_processed_data.py --root .` |
| Train the final base scorer | `python -m gene_open_inverse.final_clean_model_v1.train` |
| Run Candidate Effect Distillation | `python -m gene_open_inverse.candidate_effect_distillation_v1.run` |
| Build four-context inputs | `python -m gene_open_inverse.four_context_v1.build` |

## Data provenance

The processed artifact is derived from public K562 and RPE1 Perturb-seq screens, GEO GSE264667 HepG2 and Jurkat screens, STRING v12.0, MAP-KG, ESM-2, Reactome release 97, and NCBI human Gene Info. The manuscript provides complete citations and inclusion criteria.

## Archive integrity

`CODE_PACKAGE_MANIFEST.sha256` lists the SHA-256 digest of every payload file. To verify it on Linux or macOS:

```bash
sha256sum --check CODE_PACKAGE_MANIFEST.sha256
```

The archive intentionally contains no symlinks. All paths are relative.
