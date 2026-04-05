"""
utils/augmented_psa.py
双关系双向样本增强（Innovation 3）。

流程：
  第一阶段：多模型交集
    对18个LLM，取相似度>=k的候选对，求所有模型的交集，
    得到高置信度锚点集合。

  第二阶段：双关系双向传播
    对每个锚点(e1,e2)，沿is-a和part-of关系
    向上（父类）和向下（子类）双向传播，深度<=3跳，
    生成扩展候选对，is-a和part-of的传播结果合并。

  第三阶段：SMOA+N-gram过滤
    对候选对用SMOA字符串相似度和2-gram相似度过滤，
    满足其中一个（OR关系）且阈值=1.0的保留，
    得到最终高置信度对齐子集（增强PSA）。
"""
import json
from collections import deque
from typing import Dict, List, Set, Tuple

import numpy as np

from utils.data_loader import OMData


# ============================================================
# 字符串相似度工具
# ============================================================

def _smoa_similarity(s1: str, s2: str) -> float:
    """SMOA字符串相似度（最长公共子串比例）"""
    s1, s2 = s1.lower().strip(), s2.lower().strip()
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    len1, len2 = len(s1), len(s2)
    max_common = 0
    for i in range(len1):
        for j in range(len2):
            k = 0
            while (i + k < len1 and j + k < len2 and
                   s1[i + k] == s2[j + k]):
                k += 1
            if k > max_common:
                max_common = k
    return 2.0 * max_common / (len1 + len2)


def _ngram_similarity(s1: str, s2: str, n: int = 2) -> float:
    """N-gram相似度（Jaccard系数）"""
    s1, s2 = s1.lower().strip(), s2.lower().strip()
    if s1 == s2:
        return 1.0
    if len(s1) < n or len(s2) < n:
        return 1.0 if s1 == s2 else 0.0

    def get_ngrams(s):
        return set(s[i:i+n] for i in range(len(s) - n + 1))

    g1, g2 = get_ngrams(s1), get_ngrams(s2)
    intersection = len(g1 & g2)
    union = len(g1 | g2)
    return intersection / union if union > 0 else 0.0


def _string_match(s1: str, s2: str, threshold: float = 1.0, n: int = 2) -> bool:
    """SMOA或N-gram满足阈值（OR关系）"""
    return (_smoa_similarity(s1, s2) >= threshold or
            _ngram_similarity(s1, s2, n) >= threshold)


# ============================================================
# 第一阶段：多模型交集
# ============================================================

def compute_intersection_anchors(sim_matrices: Dict[str, np.ndarray],
                                  k: float = 0.7) -> Set[Tuple[int, int]]:
    """
    对所有LLM相似度矩阵，取>=k的候选对求交集。

    Args:
        sim_matrices: model_name -> [M, N]矩阵
        k:            相似度阈值

    Returns:
        Set of (src_idx, tgt_idx)，所有模型都认为相似的实体对
    """
    intersection_mask = None
    for name, mat in sim_matrices.items():
        mask = mat >= k
        if intersection_mask is None:
            intersection_mask = mask
        else:
            intersection_mask = intersection_mask & mask

    if intersection_mask is None:
        return set()

    rows, cols = np.where(intersection_mask)
    anchors = set(zip(rows.tolist(), cols.tolist()))
    print(f"  第一阶段：{len(anchors)} 个高置信度锚点（k={k}）")
    return anchors


# ============================================================
# 第二阶段：双关系双向传播
# ============================================================

def _get_neighbors(idx: int,
                   hierarchy_parents: Dict[int, List[int]],
                   hierarchy_children: Dict[int, List[int]],
                   part_of_parents: Dict[int, List[int]],
                   part_of_children: Dict[int, List[int]]) -> List[int]:
    """获取一个实体在is-a和part-of关系下的所有直接邻居（双向）"""
    neighbors = []
    neighbors.extend(hierarchy_parents.get(idx, []))
    neighbors.extend(hierarchy_children.get(idx, []))
    neighbors.extend(part_of_parents.get(idx, []))
    neighbors.extend(part_of_children.get(idx, []))
    return neighbors


def _bfs_neighbors(start_idx: int,
                   hierarchy_parents: Dict[int, List[int]],
                   hierarchy_children: Dict[int, List[int]],
                   part_of_parents: Dict[int, List[int]],
                   part_of_children: Dict[int, List[int]],
                   max_depth: int = 3) -> Set[int]:
    """
    BFS遍历从start_idx出发，在is-a和part-of关系下
    双向传播max_depth跳内的所有实体索引（不含start_idx本身）。
    """
    visited = {start_idx}
    queue = deque([(start_idx, 0)])
    result = set()

    while queue:
        curr, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for neighbor in _get_neighbors(
            curr,
            hierarchy_parents, hierarchy_children,
            part_of_parents, part_of_children
        ):
            if neighbor not in visited:
                visited.add(neighbor)
                result.add(neighbor)
                queue.append((neighbor, depth + 1))

    return result


def propagate_anchors(anchors: Set[Tuple[int, int]],
                      data: OMData,
                      max_depth: int = 3) -> Set[Tuple[int, int]]:
    """
    对每个锚点(e1,e2)，沿is-a和part-of关系双向传播，
    生成扩展候选对。

    传播逻辑：
      对锚点(e1,e2)：
        - 找e1在src本体中的邻居集合N1（3跳内）
        - 找e2在tgt本体中的邻居集合N2（3跳内）
        - 生成候选对：N1×{e2} ∪ {e1}×N2 ∪ N1×N2

    Args:
        anchors:   第一阶段得到的锚点集合
        data:      OMData（包含层次关系）
        max_depth: 传播深度上限

    Returns:
        所有候选对（包含原始锚点）
    """
    src_hier = data.src_hierarchy
    tgt_hier = data.tgt_hierarchy

    candidates = set(anchors)  # 包含原始锚点

    for e1, e2 in anchors:
        # e1的邻居
        n1 = _bfs_neighbors(
            e1,
            src_hier.parents, src_hier.children,
            src_hier.part_of_parents, src_hier.part_of_children,
            max_depth
        )
        # e2的邻居
        n2 = _bfs_neighbors(
            e2,
            tgt_hier.parents, tgt_hier.children,
            tgt_hier.part_of_parents, tgt_hier.part_of_children,
            max_depth
        )

        # 生成候选对：邻居与原始实体的组合
        for nb1 in n1:
            candidates.add((nb1, e2))    # e1邻居 × e2
        for nb2 in n2:
            candidates.add((e1, nb2))    # e1 × e2邻居
        for nb1 in n1:
            for nb2 in n2:
                candidates.add((nb1, nb2))  # e1邻居 × e2邻居

    print(f"  第二阶段：传播后候选对数量 {len(candidates)}")
    return candidates


# ============================================================
# 第三阶段：SMOA + N-gram 过滤
# ============================================================

def filter_by_string_similarity(candidates: Set[Tuple[int, int]],
                                  src_idx_to_label: Dict[int, str],
                                  tgt_idx_to_label: Dict[int, str],
                                  threshold: float = 1.0,
                                  n: int = 2) -> Set[Tuple[int, int]]:
    """
    对候选对用SMOA和N-gram过滤（OR关系）。
    满足 SMOA>=threshold 或 N-gram>=threshold 的对保留。

    Args:
        candidates:        第二阶段生成的候选对
        src_idx_to_label:  src实体索引->标签
        tgt_idx_to_label:  tgt实体索引->标签
        threshold:         相似度阈值
        n:                 N-gram的N

    Returns:
        过滤后的高置信度对齐子集
    """
    result = set()
    for src_idx, tgt_idx in candidates:
        src_label = src_idx_to_label.get(src_idx, "")
        tgt_label = tgt_idx_to_label.get(tgt_idx, "")
        if not src_label or not tgt_label:
            continue
        if _string_match(src_label, tgt_label, threshold, n):
            result.add((src_idx, tgt_idx))

    print(f"  第三阶段：SMOA/N-gram过滤后 {len(result)} 个对齐对")
    return result


# ============================================================
# 主入口
# ============================================================

def build_augmented_psa(data: OMData,
                         src_entities: List[dict],
                         tgt_entities: List[dict],
                         k: float = 0.7,
                         max_depth: int = 3,
                         str_threshold: float = 1.0,
                         n_gram: int = 2) -> Set[Tuple[str, str]]:
    """
    构建增强PSA（双关系双向样本增强）。

    Args:
        data:           OMData
        src_entities:   源本体实体列表（含uri和label）
        tgt_entities:   目标本体实体列表
        k:              第一阶段相似度阈值
        max_depth:      第二阶段传播深度
        str_threshold:  第三阶段字符串相似度阈值
        n_gram:         N-gram的N

    Returns:
        增强PSA：Set of (src_uri, tgt_uri)
    """
    print("构建增强PSA（双关系双向样本增强）...")

    # 建立索引
    src_idx_to_uri   = {i: e["uri"]   for i, e in enumerate(src_entities)}
    src_idx_to_label = {i: e["label"] for i, e in enumerate(src_entities)}
    tgt_idx_to_uri   = {i: e["uri"]   for i, e in enumerate(tgt_entities)}
    tgt_idx_to_label = {i: e["label"] for i, e in enumerate(tgt_entities)}

    # 第一阶段：多模型交集
    anchors = compute_intersection_anchors(data.sim_matrices, k)

    if not anchors:
        print("  警告：第一阶段交集为空，尝试降低k值")
        return set()

    # 第二阶段：双向传播
    candidates = propagate_anchors(anchors, data, max_depth)

    # 第三阶段：字符串过滤（使用idx索引）
    filtered_idx = filter_by_string_similarity(
        candidates, src_idx_to_label, tgt_idx_to_label,
        str_threshold, n_gram
    )

    # 转换回URI对
    psa_uri = set()
    for src_idx, tgt_idx in filtered_idx:
        src_uri = src_idx_to_uri.get(src_idx)
        tgt_uri = tgt_idx_to_uri.get(tgt_idx)
        if src_uri and tgt_uri:
            psa_uri.add((src_uri, tgt_uri))

    print(f"增强PSA构建完成：{len(psa_uri)} 个高置信度对齐对")
    return psa_uri


def build_augmented_psa_from_files(
        data: OMData,
        src_json_path: str = "data/parsed/mouse.json",
        tgt_json_path: str = "data/parsed/human.json",
        k: float = 0.7,
        max_depth: int = 3,
        str_threshold: float = 1.0,
        n_gram: int = 2) -> Set[Tuple[str, str]]:
    """从文件加载实体列表并构建增强PSA（便捷接口）"""
    with open(src_json_path, encoding="utf-8") as f:
        src_entities = json.load(f)
    with open(tgt_json_path, encoding="utf-8") as f:
        tgt_entities = json.load(f)

    psa = build_augmented_psa(
        data, src_entities, tgt_entities,
        k, max_depth, str_threshold, n_gram
    )

    # 验证质量
    if data.reference:
        correct  = len(psa & data.reference)
        prec     = correct / len(psa) if psa else 0
        recall   = correct / len(data.reference)
        f1       = (2*prec*recall/(prec+recall)) if (prec+recall) > 0 else 0
        print(f"增强PSA质量验证:")
        print(f"  大小={len(psa)}, 正确={correct}")
        print(f"  Precision={prec:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

    return psa


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.data_loader import load_om_data

    data = load_om_data()
    psa  = build_augmented_psa_from_files(data)
