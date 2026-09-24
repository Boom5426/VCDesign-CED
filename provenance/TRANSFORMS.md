# Packaging transformations

Source snapshots are copied from the manuscript working tree. The release
changes only machine-specific input lookup, as listed here. The SHA-256 in
`SOURCE_MAP.tsv` is the released file's checksum after this transformation.

| Release file | Packaging edit |
|---|---|
| `src/gene_open_inverse/external_context_v1/contract.py` | RPE1 input default points to `inputs/`. |
| `src/gene_open_inverse/evidence_v1/contract.py` | Frozen run default points to `inputs/runs/`. |
| `src/gene_open_inverse/external_baselines_v1/contract.py` | Public screen and frozen run defaults point to `inputs/`. |
| `src/gene_open_inverse/external_baselines_v1/evaluate.py` | SPED run default points to `inputs/runs/`. |
| `src/gene_open_inverse/external_baselines_v1/verify.py` | Witness input defaults point to `inputs/runs/`. |
| `src/gene_open_inverse/model/replogle_batch_disjoint_measurability_v1/config.json` | Absolute input paths replaced with `inputs/` placeholders. |
| `configs/candidate_knowledge_v1.json`, `configs/mapkg_candidate_adjudication_v1.json` | Absolute input paths replaced with `inputs/` placeholders. |

No split assignment, evaluation rule, random seed, metric, model parameter,
or recorded numerical result was changed.
