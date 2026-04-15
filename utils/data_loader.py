"""
utils/data_loader.py
加载所有预计算数据，供GP系统使用。

提供：
  - 18个LLM相似度矩阵（numpy array）
  - 实体索引（URI <-> 矩阵行/列）
  - 标准对齐reference（用于最终F1评估）
  - 源/目标本体层次关系（用于可靠性计算）
"""
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from rdflib import Graph, URIRef


# LLM类型分类
LLM_TYPE_MAP = {
    # 字符串类
    "biobert":        "string",
    "pubmedbert":     "string",
    "sapbert":        "string",
    "bioelectra":     "string",
    "scibert":        "string",
    "clinicalbert":   "string",
    # 结构类
    "biogpt":         "structure",
    "biogpt_large":   "structure",
    "biomistral":     "structure",
    "biomedlm":       "structure",
    "pmcllama":       "structure",
    # 语义类
    "distilbert":     "semantic",
    "tinybert":       "semantic",
    "umlsbert":       "semantic",
    "biolinkbert":    "semantic",
    "biomedbert":     "semantic",
    "pubmedbert_emb": "semantic",
    "e5base":         "semantic",
}

ALL_MODELS = list(LLM_TYPE_MAP.keys())


@dataclass
class HierarchyData:
    """单个本体的层次关系数据"""
    uri_to_idx:       Dict[str, int]
    idx_to_uri:       Dict[int, str]
    parents:          Dict[int, List[int]]   # is-a 父类
    children:         Dict[int, List[int]]   # is-a 子类
    part_of_parents:  Dict[int, List[int]]   # part-of 上位
    part_of_children: Dict[int, List[int]]   # part-of 下位

    def has_isa_relation(self, idx_a: int, idx_b: int) -> bool:
        return (
            idx_b in self.parents.get(idx_a, []) or
            idx_b in self.children.get(idx_a, [])
        )

    def has_partof_relation(self, idx_a: int, idx_b: int) -> bool:
        return (
            idx_b in self.part_of_parents.get(idx_a, []) or
            idx_b in self.part_of_children.get(idx_a, [])
        )

    def has_any_relation(self, idx_a: int, idx_b: int) -> bool:
        return (self.has_isa_relation(idx_a, idx_b) or
                self.has_partof_relation(idx_a, idx_b))


@dataclass
class OMData:
    """GP系统所需的全部数据"""
    sim_matrices:   Dict[str, np.ndarray]   # model_name -> [M, N]
    src_uri_to_idx: Dict[str, int]
    tgt_uri_to_idx: Dict[str, int]
    src_idx_to_uri: Dict[int, str]
    tgt_idx_to_uri: Dict[int, str]
    n_src:          int
    n_tgt:          int
    src_hierarchy:  HierarchyData
    tgt_hierarchy:  HierarchyData
    reference:      Set[Tuple[str, str]]    # (src_uri, tgt_uri)
    llm_type_map:   Dict[str, str] = field(
        default_factory=lambda: LLM_TYPE_MAP.copy()
    )


def load_hierarchy(json_path: str) -> HierarchyData:
    """从extract_hierarchy.py生成的JSON加载层次关系"""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    def str_keys_to_int(d: dict) -> Dict[int, List[int]]:
        return {int(k): v for k, v in d.items()}

    return HierarchyData(
        uri_to_idx=data["uri_to_idx"],
        idx_to_uri={int(k): v for k, v in data["idx_to_uri"].items()},
        parents=str_keys_to_int(data["parents"]),
        children=str_keys_to_int(data["children"]),
        part_of_parents=str_keys_to_int(data["part_of_parents"]),
        part_of_children=str_keys_to_int(data["part_of_children"]),
    )


def load_reference(ref_path: str,
                   src_uri_to_idx: Dict[str, int],
                   tgt_uri_to_idx: Dict[str, int],
                   ref_format: str = None) -> Set[Tuple[str, str]]:
    """
    加载标准对齐，返回 Set of (src_uri, tgt_uri)。
    自动根据文件扩展名判断格式（也可通过 ref_format 显式指定）。

    支持格式：
      rdf  - OAEI Anatomy 的 reference.rdf
      tsv  - Bio-ML 的 full.tsv（列：SrcEntity TgtEntity Score）
    """
    if ref_format is None:
        ref_format = "tsv" if ref_path.endswith(".tsv") else "rdf"

    if ref_format == "tsv":
        return _load_reference_tsv(ref_path, src_uri_to_idx, tgt_uri_to_idx)
    else:
        return _load_reference_rdf(ref_path, src_uri_to_idx, tgt_uri_to_idx)


def _load_reference_rdf(rdf_path: str,
                        src_uri_to_idx: Dict[str, int],
                        tgt_uri_to_idx: Dict[str, int]) -> Set[Tuple[str, str]]:
    """OAEI RDF格式解析（原有逻辑，完整保留）"""
    import re
    reference = set()

    with open(rdf_path, encoding="utf-8") as f:
        content = f.read()

    cell_pattern = re.compile(r'<Cell>(.*?)</Cell>', re.DOTALL)
    entity_pattern = re.compile(r'<entity(\d) rdf:resource="([^"]+)"')

    for cell_match in cell_pattern.finditer(content):
        cell_content = cell_match.group(1)
        entities = {}
        for e_match in entity_pattern.finditer(cell_content):
            entities[e_match.group(1)] = e_match.group(2)

        e1_uri = entities.get("1")
        e2_uri = entities.get("2")
        if not e1_uri or not e2_uri:
            continue

        if e1_uri in src_uri_to_idx and e2_uri in tgt_uri_to_idx:
            reference.add((e1_uri, e2_uri))
        elif e2_uri in src_uri_to_idx and e1_uri in tgt_uri_to_idx:
            reference.add((e2_uri, e1_uri))

    return reference


def _load_reference_tsv(tsv_path: str,
                        src_uri_to_idx: Dict[str, int],
                        tgt_uri_to_idx: Dict[str, int]) -> Set[Tuple[str, str]]:
    """Bio-ML TSV格式解析（列：SrcEntity TgtEntity Score，首行为header）"""
    import csv
    reference = set()

    with open(tsv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            src_uri = row.get("SrcEntity", "").strip()
            tgt_uri = row.get("TgtEntity", "").strip()
            if not src_uri or not tgt_uri:
                continue
            if src_uri in src_uri_to_idx and tgt_uri in tgt_uri_to_idx:
                reference.add((src_uri, tgt_uri))
            elif tgt_uri in src_uri_to_idx and src_uri in tgt_uri_to_idx:
                reference.add((tgt_uri, src_uri))

    return reference


def load_om_data(
    emb_dir:        str = "embeddings/anatomy",
    src_entities:   str = "data/parsed/mouse.json",
    tgt_entities:   str = "data/parsed/human.json",
    src_hierarchy:  str = "data/parsed/mouse_hierarchy.json",
    tgt_hierarchy:  str = "data/parsed/human_hierarchy.json",
    reference_path: str = "data/oaei/anatomy/reference.rdf",
    ref_format:     str = None,
    models:         Optional[List[str]] = None,
) -> OMData:
    """
    一次性加载GP系统所需的全部数据。
    需要在项目根目录（om-zjj/）下运行。
    """
    if models is None:
        models = ALL_MODELS

    # 1. 实体索引
    with open(src_entities, encoding="utf-8") as f:
        src_ents = json.load(f)
    with open(tgt_entities, encoding="utf-8") as f:
        tgt_ents = json.load(f)

    src_uri_to_idx = {e["uri"]: i for i, e in enumerate(src_ents)}
    tgt_uri_to_idx = {e["uri"]: i for i, e in enumerate(tgt_ents)}
    src_idx_to_uri = {i: e["uri"] for i, e in enumerate(src_ents)}
    tgt_idx_to_uri = {i: e["uri"] for i, e in enumerate(tgt_ents)}
    n_src, n_tgt = len(src_ents), len(tgt_ents)
    print(f"实体数量: src={n_src}, tgt={n_tgt}")

    # 2. 相似度矩阵
    sim_matrices = {}
    for model in models:
        npy_path = os.path.join(emb_dir, f"{model}_sim.npy")
        if not os.path.exists(npy_path):
            print(f"[WARNING] 找不到: {npy_path}，跳过")
            continue
        mat = np.load(npy_path).astype(np.float32)
        assert mat.shape == (n_src, n_tgt), \
            f"{model} 矩阵形状 {mat.shape} 与期望 ({n_src},{n_tgt}) 不符"
        sim_matrices[model] = mat
        print(f"  加载 {model}: shape={mat.shape}, "
              f"mean={mat.mean():.4f}, std={mat.std():.4f}")
    print(f"共加载 {len(sim_matrices)} 个相似度矩阵")

    # 3. 层次关系
    print("加载层次关系...")
    src_hier = load_hierarchy(src_hierarchy)
    tgt_hier = load_hierarchy(tgt_hierarchy)
    print(f"  src: {len(src_hier.uri_to_idx)}实体, "
          f"{sum(len(v) for v in src_hier.parents.values())}条is-a, "
          f"{sum(len(v) for v in src_hier.part_of_parents.values())}条part-of")
    print(f"  tgt: {len(tgt_hier.uri_to_idx)}实体, "
          f"{sum(len(v) for v in tgt_hier.parents.values())}条is-a, "
          f"{sum(len(v) for v in tgt_hier.part_of_parents.values())}条part-of")

    # 4. 标准对齐
    print("加载标准对齐...")
    reference = load_reference(reference_path, src_uri_to_idx, tgt_uri_to_idx, ref_format)
    print(f"  标准对齐数量: {len(reference)}")

    return OMData(
        sim_matrices=sim_matrices,
        src_uri_to_idx=src_uri_to_idx,
        tgt_uri_to_idx=tgt_uri_to_idx,
        src_idx_to_uri=src_idx_to_uri,
        tgt_idx_to_uri=tgt_idx_to_uri,
        n_src=n_src,
        n_tgt=n_tgt,
        src_hierarchy=src_hier,
        tgt_hierarchy=tgt_hier,
        reference=reference,
    )


if __name__ == "__main__":
    # 需要在 om-zjj/ 根目录下运行：python utils/data_loader.py
    data = load_om_data()
    print("\n=== 数据加载完成 ===")
    print(f"相似度矩阵: {list(data.sim_matrices.keys())}")
    print(f"标准对齐数量: {len(data.reference)}")

    # 抽查一条标准对齐
    sample = next(iter(data.reference))
    print(f"\n标准对齐样例: {sample}")
    src_idx = data.src_uri_to_idx.get(sample[0], -1)
    tgt_idx = data.tgt_uri_to_idx.get(sample[1], -1)
    if src_idx >= 0 and tgt_idx >= 0:
        print(f"  src_idx={src_idx}, tgt_idx={tgt_idx}")
        for m in ["biogpt", "sapbert", "biomistral"]:
            if m in data.sim_matrices:
                print(f"  {m}相似度: "
                      f"{data.sim_matrices[m][src_idx, tgt_idx]:.4f}")