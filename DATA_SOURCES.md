# Data sources

The companion [processed-data release](https://huggingface.co/datasets/Boom5426/VCDesign)
contains the frozen matrices and run artifacts used by this repository. It is
derived from the following public resources.

| Resource | Use in VCDesign-CED |
|---|---|
| [Replogle et al. Perturb-seq release](https://plus.figshare.com/articles/dataset/_Mapping_information-rich_genotype-phenotype_landscapes_with_genome-scale_Perturb-seq_Replogle_et_al_2022_processed_Perturb-seq_datasets/20029387) | K562 genome-scale and RPE1 essential CRISPRi screens |
| [GEO GSE264667](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE264667) | HepG2 and Jurkat CRISPRi screens |
| [STRING v12.0](https://version-12-0.string-db.org/cgi/download) | Static protein interaction features |
| [MAP and MAP-KG](https://github.com/MAGIC-AI4Med/MAP) | Frozen knowledge graph gene representation |
| [ESM-2](https://github.com/facebookresearch/esm) | Protein sequence representation |
| [Reactome release 97](https://download.reactome.org/97/ReactomePathways.gmt.zip) and [NCBI human Gene Info](https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz) | Pathway and gene-description feature variants |

Use the K562 genome-scale screen specified above. It differs from the K562
essential screen and defines the candidate universe used for the primary setup.
