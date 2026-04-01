"""
utils/psa_builder.py
构建部分标准对齐（Partial Standard Alignment, PSA）。

PSA是标准对齐的子集，用于GP进化过程中的近似适应度评估。
构建策略（参考DPGP-AML论文）：
  1. 在is-a层级图中，选取in-degree+out-degree最高的Top-δ个概念
  2. 用SMOA字符串相似度（完全匹配，阈值1.0）找初始锚点
  3. 分别在is-a图和part-of图中执行

PSA的作用：
  - 替代完整标准对齐（实际未知）进行适应度近似
  - 节点度数高的概念是本体中的关键概念，用于PSA能更好地引导搜索方向
"""
import json
from typing import Dict, List, Set, Tuple

import numpy as np

from utils.data_loader import OMData


# ============================================================
# 字符串相似度（SMOA简化版）
# ============================================================

def _smoa_similarity(s1: str, s2: str) -> float:
    """
    SMOA字符串相似度（简化版）。
    完全匹配返回1.0，否则基于公共子串计算相似度。
    PSA构建中只使用完全匹配（返回1.0）的情况。
    """
    s1, s2 = s1.lower().strip(), s2.lower().strip()
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    # 最长公共子串比例
    len1, len2 = len(s1), len(s2)
    max_common = 0
    for i in range(len1):
        for j in range(len2):
            k = 0
            while (i + k < len1 and j + k < len2 and
                   s1[i + k] == s2[j + k]):
                k += 1
            max_common = max(max_common, k)
    return 2.0 * max_common / (len1 + len2)


def _get_label_from_uri(uri: str,
                        entities_json: List[dict]) -> str:
    """根据URI从实体列表中获取label"""
    # 建立URI->label的快速查找（只建一次，调用方负责缓存）
    return ""


# ============================================================
# PSA构建
# ============================================================

def build_psa(data: OMData,
              src_entities: List[dict],
              tgt_entities: List[dict],
              delta: float = 0.3) -> Set[Tuple[str, str]]:
    """
    构建部分标准对齐（PSA）。

    Args:
        data:          OMData（包含层次关系和实体索引）
        src_entities:  源本体实体列表（来自01_parse_ontology.py的JSON）
        tgt_entities:  目标本体实体列表
        delta:         选取Top-δ比例的代表性概念，默认30%

    Returns:
        PSA：Set of (src_uri, tgt_uri)
    """
    src_hier = data.src_hierarchy
    tgt_hier = data.tgt_hierarchy

    # 建立URI->label映射
    src_uri_to_label = {e["uri"]: e["label"] for e in src_entities}
    tgt_uri_to_label = {e["uri"]: e["label"] for e in tgt_entities}

    # ---- Step 1：计算每个实体的度数（in-degree + out-degree）----
    def compute_degree(uri_to_idx: Dict[str, int],
                       parents: Dict[int, List[int]],
                       children: Dict[int, List[int]],
                       part_of_parents: Dict[int, List[int]],
                       part_of_children: Dict[int, List[int]]) -> Dict[int, int]:
        """计算每个实体的总度数（is-a + part-of，双向）"""
        degree = {}
        n = len(uri_to_idx)
        for idx in range(n):
            d = (len(parents.get(idx, [])) +
                 len(children.get(idx, [])) +
                 len(part_of_parents.get(idx, [])) +
                 len(part_of_children.get(idx, [])))
            degree[idx] = d
        return degree

    src_degree = compute_degree(
        src_hier.uri_to_idx,
        src_hier.parents, src_hier.children,
        src_hier.part_of_parents, src_hier.part_of_children,
    )
    tgt_degree = compute_degree(
        tgt_hier.uri_to_idx,
        tgt_hier.parents, tgt_hier.children,
        tgt_hier.part_of_parents, tgt_hier.part_of_children,
    )

    # ---- Step 2：选Top-δ代表性概念 ----
    n_src_top = max(1, int(len(src_hier.uri_to_idx) * delta))
    n_tgt_top = max(1, int(len(tgt_hier.uri_to_idx) * delta))

    src_top_idxs = sorted(
        src_degree.keys(), key=lambda i: src_degree[i], reverse=True
    )[:n_src_top]
    tgt_top_idxs = sorted(
        tgt_degree.keys(), key=lambda i: tgt_degree[i], reverse=True
    )[:n_tgt_top]

    # 转换回URI
    src_top_uris = [src_hier.idx_to_uri[i] for i in src_top_idxs
                    if i in src_hier.idx_to_uri]
    tgt_top_uris = [tgt_hier.idx_to_uri[i] for i in tgt_top_idxs
                    if i in tgt_hier.idx_to_uri]

    # ---- Step 3：SMOA完全匹配找锚点 ----
    psa = set()
    for src_uri in src_top_uris:
        src_label = src_uri_to_label.get(src_uri, "")
        if not src_label:
            continue
        for tgt_uri in tgt_top_uris:
            tgt_label = tgt_uri_to_label.get(tgt_uri, "")
            if not tgt_label:
                continue
            # 只取完全匹配（SMOA=1.0）
            if _smoa_similarity(src_label, tgt_label) == 1.0:
                psa.add((src_uri, tgt_uri))
                break  # 每个src只匹配第一个完全相同的tgt

    return psa


def build_psa_from_files(data: OMData,
                          src_json_path: str = "data/parsed/mouse.json",
                          tgt_json_path: str = "data/parsed/human.json",
                          delta: float = 0.3) -> Set[Tuple[str, str]]:
    """
    从文件加载实体列表并构建PSA（便捷接口）。
    """
    import json
    with open(src_json_path, encoding="utf-8") as f:
        src_entities = json.load(f)
    with open(tgt_json_path, encoding="utf-8") as f:
        tgt_entities = json.load(f)

    psa = build_psa(data, src_entities, tgt_entities, delta)
    print(f"PSA构建完成: {len(psa)} 个锚点对 (delta={delta})")
    return psa


if __name__ == "__main__":
    from utils.data_loader import load_om_data
    data = load_om_data()
    psa = build_psa_from_files(data)

    # 验证PSA质量：与标准对齐对比
    if data.reference:
        correct = len(psa & data.reference)
        precision = correct / len(psa) if psa else 0
        recall    = correct / len(data.reference) if data.reference else 0
        print(f"PSA质量验证:")
        print(f"  PSA大小: {len(psa)}")
        print(f"  与标准对齐交集: {correct}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
