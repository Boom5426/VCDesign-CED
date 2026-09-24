# Release status

## Confirmed in this staging package

- Main-model and four-context Python modules are copied without changing their
  metric definitions, split seeds, evaluation rules, or model settings.
- The RPE1 source-file default in `external_context_v1/contract.py` uses a
  package-relative input path in this release. The original source default was
  an absolute compute-host path; this packaging edit changes only input lookup.
- The external-baseline and evidence modules use package-relative defaults under
  `inputs/public/` and `inputs/runs/` in place of compute-host paths. The constants
  for methods, metrics, seeds, arms, and hyperparameters are unchanged.
- The corresponding local and compute-host Python files were hash-identical when
  checked during assembly.
- Paper figure scripts, figure-level JSON/CSV, supplementary closure, and table
  bodies are copied from the current manuscript working tree. They may contain
  uncommitted manuscript changes, so the per-file hashes in `provenance/SOURCE_MAP.tsv`
  are the release snapshot identity.
- The frozen input index records 19 asset/comparator checksums and the selected
  epoch-8 checkpoint hash without exposing compute-host paths.
- Public source locations and exact K562/HepG2/Jurkat variants are documented in
  `DATA_SOURCES.md`; the processed-data revision is not yet verified.
- The staged Python package built into a wheel on Python 3.11. The synthetic
  example ran, 162 collected protocol tests passed on the compute host, and the
  standalone K562 batch-disjoint contract check passed.
- The main display-data builder reproduced its two bundled files byte-for-byte
  in separate output paths. All three current main-figure plotters completed;
  the supplementary exporter and its original SHA-256 manifest passed.

## Required before the anonymous ICLR code upload

- Add the frozen asset configuration in a portable, anonymized form. Validate every
  source asset checksum, and document how to regenerate K562 views, STRING/MAP-KG
  features, and the four-context processed atlases from public releases.
- Verify the processed-data repository contains the expected files, fix its exact
  revision and hashes, and provide an anonymous access route for reviewers.
- Add the selected epoch-8 base checkpoint or a verified training reproduction,
  and the locked-evaluation manifest and witnessed score matrices. Check their
  relation to the paper's frozen results without refitting on evaluation identities.
- Add portable input-configuration overrides for the external-baseline scripts,
  and include their run-level manifests and independent verification outputs.
- Trace each manuscript table and figure back to a source record and executable
  command. Reconcile the recorded run-manifest file-count discrepancy before quoting
  a count.
- Test in a clean Python 3.11 environment and audit the final archive for names,
  machine paths, URLs, embedded metadata, and Git history. Verify the archive's
  extracted bytes against the manifest.
- Decide the public license and post-review citation metadata. Keep author
  fields and named-repository links out of the review archive.
- Create the named backup from this directory as a fresh Git repository with
  independent history; the enclosing research worktree must not be pushed.

Do not label this directory submission-ready until these checks pass.
