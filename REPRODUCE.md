# Reproduction guide

Run every command from the repository root. Outputs are intentionally explicit:
the long-running stages refuse to overwrite an existing output directory.

## 1. Install and verify the data snapshot

```bash
python3 -m pip install -e ".[model,data,dev]"
python3 -m pip install "huggingface_hub>=0.35"
hf download Boom5426/VCDesign --repo-type dataset --local-dir . \
  --include 'inputs/**' 'checkpoints/**' 'records/**' 'DATA_MANIFEST.json'
python3 tools/verify_processed_data.py --root .
```

The verifier checks `DATA_MANIFEST.json`, including release hashes for records
whose compute-host paths were replaced by relative release paths.

## 2. Run software checks

```bash
python3 examples/ced_demo.py
python3 -m pytest -q
```

## 3. Train the primary base scorer

Use a CUDA device and an empty output path:

```bash
python3 -m gene_open_inverse.final_clean_model_v1.train \
  --config configs/paper_run_v1.json \
  --output outputs/final_clean_train \
  --device cuda
```

The run writes checkpoints and `train_manifest.json`. The frozen reference
checkpoint is `checkpoints/final_clean_epoch_008.pt`; its SHA-256 is recorded in
both `configs/paper_run_v1.json` and the processed-data manifest.

## 4. Inspect the paper result

The final manuscript is available at [paper/VCDesign.pdf](paper/VCDesign.pdf).
It defines the candidate pools, evaluation protocol, comparator settings, and
reported results. Raw H5AD matrices can be obtained from the public sources in
[DATA_SOURCES.md](DATA_SOURCES.md).
