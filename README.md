# VCDesign-CED

### Candidate-conditioned inverse modeling for cellular intervention design

VCDesign ranks candidate interventions for a requested cellular transition under
a finite experimental budget. When a candidate has no measured response,
VCDesign-CED predicts its effect from STRING and MAP-KG knowledge and combines
goal alignment with a direct candidate scorer. The released code also contains
the batch-disjoint measurement construction, held-context and response-masking
evaluations, and the adapters used for the paper's external comparators.

**Submission staging status:** the code and figure-level source data are assembled,
but the frozen compute artifacts and full run-to-paper trace are still being
prepared. See [RELEASE_STATUS.md](RELEASE_STATUS.md). This directory is not yet
the anonymous supplementary upload.

## Quick start

Python 3.11 is required. Install a PyTorch 2.4 build appropriate for your
machine, then install the package and run the synthetic example:

```bash
python3 -m pip install -e ".[model,data,figures,dev]"
python3 examples/ced_demo.py
python3 -m pytest -q
```

The demo uses simulated features and responses to exercise response-basis
construction, ridge effect prediction, and unit-weight score fusion. Its ranking
is a software check, not a biological result.

## Use and reproduce

| Goal | Entry point | Inputs |
|---|---|---|
| Inspect the method | `src/gene_open_inverse/final_clean_model_v1/` and `candidate_effect_distillation_v1/` | Code only |
| Check method contracts | `python3 -m pytest -q` | Synthetic fixtures |
| Rebuild the main figure data | `paper_assets/figures/scripts/build_display_v2_data.py` | Bundled frozen row-level data |
| Check supplementary tables | `paper_assets/supplementary/scripts/export_external_closure.py --check` | Bundled closure JSON and CSVs |
| Retrain or re-evaluate the paper model | Stage modules under `src/gene_open_inverse/` | Public screens, knowledge assets, frozen configurations |

The full paper pipeline uses public K562 and RPE1 CRISPRi screens, HepG2 and
Jurkat screens from GEO GSE264667, STRING v12.0, and released MAP-KG and ESM-2
assets. Large input matrices and model checkpoints are outside this staging
directory. [DATA_SOURCES.md](DATA_SOURCES.md) identifies the public releases and
exact screen variants. `configs/asset_config.example.json` gives the loader's required keys;
`configs/frozen_input_checksums.json` records checksums and sizes for the frozen
inputs and the selected epoch-8 checkpoint. Match each staged input to its
checksum before an evaluation run. Once a processed-data snapshot has been
downloaded, run `python3 tools/verify_processed_data.py --root <data-root>`.
The paths in the example configuration are placeholders, not the paths used on
the compute host.

### Rebuild the numerical display data

Run from `paper_assets/`. Choose new output filenames; the builder refuses to
overwrite them by default.

```bash
python3 figures/scripts/build_display_v2_data.py \
  --output /tmp/vcdesign-display-v2.json \
  --source-csv /tmp/vcdesign-display-v2.csv
sha256sum figures/data/display_v2_data.json figures/data/display_v2_source_data.csv
python3 supplementary/scripts/export_external_closure.py \
  --source supplementary/data/external_baseline_closure_v1.json \
  --out-dir supplementary/data --check
sha256sum -c supplementary/data/SHA256SUMS
```

The packaged figure builder reproduced the two bundled display-data files
byte-for-byte in a separate output directory. Main figure plotting scripts and
the editable Figure 1 source are in `paper_assets/figures/`. Table bodies and
row-level source data are also included there. See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
for the module map.

## Package integrity and anonymity

`provenance/SOURCE_MAP.tsv` records the release checksum and source location of
each copied file. `provenance/TRANSFORMS.md` lists the input-path changes made
for portability. `provenance/PACKAGE_SHA256SUMS` covers every staged package file.
Run `python3 tools/audit_release.py` before making an archive. The audit checks
snapshot and package hashes, text for machine paths or identity strings, and
embedded Office XML. Inspect the final archive and its metadata as well.

The double-blind submission should use an anonymous supplementary archive or
anonymous repository. A named backup repository can be used after the package
is ready, but its owner link must not appear in review materials.
