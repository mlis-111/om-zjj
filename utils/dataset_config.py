"""
utils/dataset_config.py
数据集配置中心。
"""

DATASET_CONFIGS = {
    "anatomy": {
        "emb_dir":       "embeddings/anatomy",
        "src_entities":  "data/parsed/mouse.json",
        "tgt_entities":  "data/parsed/human.json",
        "src_hierarchy": "data/parsed/mouse_hierarchy.json",
        "tgt_hierarchy": "data/parsed/human_hierarchy.json",
        "src_owl":       "data/oaei/anatomy/mouse.owl",
        "tgt_owl":       "data/oaei/anatomy/human.owl",
        "reference":     "data/oaei/anatomy/reference.rdf",
        "ref_format":    "rdf",
        "psa_cache_dir": "data/parsed/psa_anatomy",
    },
    "omim-ordo": {
        "emb_dir":       "embeddings/omim-ordo",
        "src_entities":  "data/parsed/omim_ordo_src.json",
        "tgt_entities":  "data/parsed/omim_ordo_tgt.json",
        "src_hierarchy": "data/parsed/omim_ordo_src_hierarchy.json",
        "tgt_hierarchy": "data/parsed/omim_ordo_tgt_hierarchy.json",
        "src_owl":       "data/bio-ml-2024/omim-ordo/omim.owl",
        "tgt_owl":       "data/bio-ml-2024/omim-ordo/ordo.owl",
        "reference":     "data/bio-ml-2024/omim-ordo/refs_equiv/full.tsv",
        "ref_format":    "tsv",
        "psa_cache_dir": "data/parsed/psa_omim_ordo",
    },
    "ncit-doid": {
        "emb_dir":       "embeddings/ncit-doid",
        "src_entities":  "data/parsed/ncit_doid_src.json",
        "tgt_entities":  "data/parsed/ncit_doid_tgt.json",
        "src_hierarchy": "data/parsed/ncit_doid_src_hierarchy.json",
        "tgt_hierarchy": "data/parsed/ncit_doid_tgt_hierarchy.json",
        "src_owl":       "data/bio-ml-2024/ncit-doid/ncit.owl",
        "tgt_owl":       "data/bio-ml-2024/ncit-doid/doid.owl",
        "reference":     "data/bio-ml-2024/ncit-doid/refs_equiv/full.tsv",
        "ref_format":    "tsv",
        "psa_cache_dir": "data/parsed/psa_ncit_doid",
    },
    "snomed-fma": {
        "emb_dir":       "embeddings/snomed-fma",
        "src_entities":  "data/parsed/snomed_fma_src.json",
        "tgt_entities":  "data/parsed/snomed_fma_tgt.json",
        "src_hierarchy": "data/parsed/snomed_fma_src_hierarchy.json",
        "tgt_hierarchy": "data/parsed/snomed_fma_tgt_hierarchy.json",
        "src_owl":       "data/bio-ml-2024/snomed-fma.body/snomed.body.owl",
        "tgt_owl":       "data/bio-ml-2024/snomed-fma.body/fma.body.owl",
        "reference":     "data/bio-ml-2024/snomed-fma.body/refs_equiv/full.tsv",
        "ref_format":    "tsv",
        "psa_cache_dir": "data/parsed/psa_snomed_fma",
    },
    "snomed-ncit-pharm": {
        "emb_dir":       "embeddings/snomed-ncit-pharm",
        "src_entities":  "data/parsed/snomed_ncit_pharm_src.json",
        "tgt_entities":  "data/parsed/snomed_ncit_pharm_tgt.json",
        "src_hierarchy": "data/parsed/snomed_ncit_pharm_src_hierarchy.json",
        "tgt_hierarchy": "data/parsed/snomed_ncit_pharm_tgt_hierarchy.json",
        "src_owl":       "data/bio-ml-2024/snomed-ncit.pharm/snomed.pharm.owl",
        "tgt_owl":       "data/bio-ml-2024/snomed-ncit.pharm/ncit.pharm.owl",
        "reference":     "data/bio-ml-2024/snomed-ncit.pharm/refs_equiv/full.tsv",
        "ref_format":    "tsv",
        "psa_cache_dir": "data/parsed/psa_snomed_ncit_pharm",
    },
    "snomed-ncit-neoplas": {
        "emb_dir":       "embeddings/snomed-ncit-neoplas",
        "src_entities":  "data/parsed/snomed_ncit_neoplas_src.json",
        "tgt_entities":  "data/parsed/snomed_ncit_neoplas_tgt.json",
        "src_hierarchy": "data/parsed/snomed_ncit_neoplas_src_hierarchy.json",
        "tgt_hierarchy": "data/parsed/snomed_ncit_neoplas_tgt_hierarchy.json",
        "src_owl":       "data/bio-ml-2024/snomed-ncit.neoplas/snomed.neoplas.owl",
        "tgt_owl":       "data/bio-ml-2024/snomed-ncit.neoplas/ncit.neoplas.owl",
        "reference":     "data/bio-ml-2024/snomed-ncit.neoplas/refs_equiv/full.tsv",
        "ref_format":    "tsv",
        "psa_cache_dir": "data/parsed/psa_snomed_ncit_neoplas",
    },
}

ALL_DATASETS = list(DATASET_CONFIGS.keys())


def get_config(dataset_name: str) -> dict:
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"未知数据集: {dataset_name}，可选: {ALL_DATASETS}")
    return DATASET_CONFIGS[dataset_name].copy()