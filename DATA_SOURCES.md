# Data and knowledge sources

The paper uses public perturbation screens and public knowledge resources. Raw
single-cell matrices, processed atlases, and model checkpoints are too large for
this code package. The frozen derived input roles and checksums are in
`configs/asset_config.example.json` and `configs/frozen_input_checksums.json`.

| Input | Public source | File or version required by the code |
|---|---|---|
| K562 genome-scale CRISPRi | [Replogle et al. processed Perturb-seq release](https://plus.figshare.com/articles/dataset/_Mapping_information-rich_genotype-phenotype_landscapes_with_genome-scale_Perturb-seq_Replogle_et_al_2022_processed_Perturb-seq_datasets/20029387) | K562 **genome-scale**, raw single-cell H5AD; staged as `inputs/ReplogleWeissman2022_K562_gwps.h5ad` |
| RPE1 essential CRISPRi | [Same Replogle release](https://plus.figshare.com/articles/dataset/_Mapping_information-rich_genotype-phenotype_landscapes_with_genome-scale_Perturb-seq_Replogle_et_al_2022_processed_Perturb-seq_datasets/20029387) | RPE1 raw single-cell H5AD |
| HepG2 and Jurkat CRISPRi | [GEO GSE264667](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667) | `GSE264667_hepg2_raw_singlecell_01.h5ad` and `GSE264667_jurkat_raw_singlecell_01.h5ad`; exact expected byte sizes are in `four_context_v1/manifest.py` |
| Protein interactions | [STRING v12.0 downloads](https://version-12-0.string-db.org/cgi/download) | Physical-interaction links, with the human taxon and score handling specified by the feature builder |
| MAP-KG gene representation | [MAP official code](https://github.com/MAGIC-AI4Med/MAP) and [MAP-KG release](https://huggingface.co/datasets/RainGate/MAP-KG/tree/main) | Official frozen `mapkg_encoder_v3.pt` and the ESM2 gene embedding; checkpoint identity is recorded in `configs/mapkg_candidate_adjudication_v1.json` |
| Protein sequence representation | [ESM-2 models](https://github.com/facebookresearch/esm/blob/main/README.md) | `esm2_t48_15B_UR50D`, layer 48 mean residue embedding, as recorded in `configs/candidate_knowledge_v1.json` |
| Ancillary knowledge | [Reactome release 97](https://download.reactome.org/97/ReactomePathways.gmt.zip) and [NCBI human Gene Info](https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz) | Pathway membership and gene descriptions for knowledge-feature variants |

The K562 genome-scale screen and K562 essential screen are different releases;
substituting the essential screen would change the paper's candidate universe.
The processed matrices and features must be matched to the frozen hashes before
training or evaluation. The anonymous processed-data location and exact revision
are pending verification. No account-owned dataset link is part of this review
package.

The source modules under `src/gene_open_inverse/` retain the paper's processing
rules. The public-source table identifies inputs; it is not yet a verified
end-to-end regeneration recipe. The missing commands and frozen run records are
tracked in `RELEASE_STATUS.md`.
