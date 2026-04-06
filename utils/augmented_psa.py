"""
utils/augmented_psa.py
双关系双向样本增强（Innovation 3）。

流程：
  第一阶段：多模型交集
    对18个LLM，取相似度>=k的候选对，求所有模型的交集。

  第二阶段：双关系双向传播
    对每个锚点(e1,e2)，沿is-a和part-of关系双向传播，深度<=3跳。

  第三阶段：SMOA+N-gram分层过滤
    Layer1：SMOA/N-gram >= 1.0，高精度锚点（P≈0.965）
    Layer2：SMOA/N-gram >= 0.75，去掉Layer1后的扩展锚点（P≈0.569）

输出：LayeredPSA，包含两层锚点及各自权重（w1=0.7, w2=0.3）。
"""
import os
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import numpy as np

from utils.data_loader import OMData


# ============================================================
# 分层PSA数据结构
# ============================================================

@dataclass
class LayeredPSA:
    """
    分层PSA。
    layer1: 高精度锚点
    layer2: 扩展锚点（不与layer1重叠）
    w1/w2:  各层权重，用于加权fitness计算
    """
    layer1: Set[Tuple[str, str]] = field(default_factory=set)
    layer2: Set[Tuple[str, str]] = field(default_factory=set)
    w1: float = 0.7
    w2: float = 0.3

    @property
    def all_pairs(self) -> Set[Tuple[str, str]]:
        return self.layer1 | self.layer2

    def __len__(self):
        return len(self.layer1) + len(self.layer2)

    def summary(self):
        print(f"  Layer1: {len(self.layer1)} 个 (w={self.w1})")
        print(f"  Layer2: {len(self.layer2)} 个 (w={self.w2})")
        print(f"  合计:   {len(self)} 个")


# ============================================================
# 字符串相似度工具
# ============================================================

def _smoa_similarity(s1: str, s2: str) -> float:
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
    s1, s2 = s1.lower().strip(), s2.lower().strip()
    if s1 == s2:
        return 1.0
    if len(s1) < n or len(s2) < n:
        return 1.0 if s1 == s2 else 0.0
    def get_ngrams(s):
        return set(s[i:i+n] for i in range(len(s) - n + 1))
    g1, g2 = get_ngrams(s1), get_ngrams(s2)
    union = len(g1 | g2)
    return len(g1 & g2) / union if union > 0 else 0.0


def _string_match(s1: str, s2: str, threshold: float, n: int = 2) -> bool:
    """SMOA或N-gram满足阈值（OR关系）"""
    return (_smoa_similarity(s1, s2) >= threshold or
            _ngram_similarity(s1, s2, n) >= threshold)


# ============================================================
# 第一阶段：多模型交集
# ============================================================

def compute_intersection_anchors(sim_matrices: Dict[str, np.ndarray],
                                  k: float = 0.7) -> Set[Tuple[int, int]]:
    intersection_mask = None
    for mat in sim_matrices.values():
        mask = mat >= k
        intersection_mask = mask if intersection_mask is None else (intersection_mask & mask)
    if intersection_mask is None:
        return set()
    rows, cols = np.where(intersection_mask)
    anchors = set(zip(rows.tolist(), cols.tolist()))
    print(f"  第一阶段：{len(anchors)} 个高置信度锚点（k={k}）")
    return anchors


# ============================================================
# 第二阶段：双关系双向传播
# ============================================================

def _bfs_neighbors(start_idx: int,
                   parents: Dict[int, List[int]],
                   children: Dict[int, List[int]],
                   part_of_parents: Dict[int, List[int]],
                   part_of_children: Dict[int, List[int]],
                   max_depth: int = 3) -> Set[int]:
    visited = {start_idx}
    queue   = deque([(start_idx, 0)])
    result  = set()
    while queue:
        curr, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for nb in (parents.get(curr, []) + children.get(curr, []) +
                   part_of_parents.get(curr, []) + part_of_children.get(curr, [])):
            if nb not in visited:
                visited.add(nb)
                result.add(nb)
                queue.append((nb, depth + 1))
    return result


def propagate_anchors(anchors: Set[Tuple[int, int]],
                      data: OMData,
                      max_depth: int = 3) -> Set[Tuple[int, int]]:
    src_hier   = data.src_hierarchy
    tgt_hier   = data.tgt_hierarchy
    candidates = set(anchors)

    for e1, e2 in anchors:
        n1 = _bfs_neighbors(e1, src_hier.parents, src_hier.children,
                             src_hier.part_of_parents, src_hier.part_of_children, max_depth)
        n2 = _bfs_neighbors(e2, tgt_hier.parents, tgt_hier.children,
                             tgt_hier.part_of_parents, tgt_hier.part_of_children, max_depth)
        for nb1 in n1:
            candidates.add((nb1, e2))
        for nb2 in n2:
            candidates.add((e1, nb2))
        for nb1 in n1:
            for nb2 in n2:
                candidates.add((nb1, nb2))

    print(f"  第二阶段：传播后候选对数量 {len(candidates)}")
    return candidates


# ============================================================
# 第三阶段：分层字符串过滤
# ============================================================

def filter_layered(candidates: Set[Tuple[int, int]],
                   src_idx_to_label: Dict[int, str],
                   tgt_idx_to_label: Dict[int, str],
                   src_idx_to_uri: Dict[int, str],
                   tgt_idx_to_uri: Dict[int, str],
                   thresh1: float = 1.0,
                   thresh2: float = 0.8,
                   n: int = 2) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    """
    分层过滤：
      Layer1：满足thresh1（高精度）
      Layer2：满足thresh2但不在Layer1（扩展）
    """
    layer1_idx = set()
    layer2_idx = set()

    for src_idx, tgt_idx in candidates:
        sl = src_idx_to_label.get(src_idx, "")
        tl = tgt_idx_to_label.get(tgt_idx, "")
        if not sl or not tl:
            continue
        if _string_match(sl, tl, thresh1, n):
            layer1_idx.add((src_idx, tgt_idx))
        elif _string_match(sl, tl, thresh2, n):
            layer2_idx.add((src_idx, tgt_idx))

    def to_uri(idx_set):
        result = set()
        for si, ti in idx_set:
            su = src_idx_to_uri.get(si)
            tu = tgt_idx_to_uri.get(ti)
            if su and tu:
                result.add((su, tu))
        return result

    layer1 = to_uri(layer1_idx)
    layer2 = to_uri(layer2_idx)
    print(f"  第三阶段：Layer1={len(layer1)}，Layer2={len(layer2)}")
    return layer1, layer2


# ============================================================
# 主入口
# ============================================================

def build_augmented_psa(data: OMData,
                         src_entities: List[dict],
                         tgt_entities: List[dict],
                         k: float = 0.7,
                         max_depth: int = 3,
                         thresh1: float = 1.0,
                         thresh2: float = 0.8,
                         n_gram: int = 2,
                         w1: float = 0.7,
                         w2: float = 0.3) -> LayeredPSA:
    """构建分层增强PSA"""
    print("构建增强PSA（双关系双向样本增强）...")

    src_idx_to_uri   = {i: e["uri"]   for i, e in enumerate(src_entities)}
    src_idx_to_label = {i: e["label"] for i, e in enumerate(src_entities)}
    tgt_idx_to_uri   = {i: e["uri"]   for i, e in enumerate(tgt_entities)}
    tgt_idx_to_label = {i: e["label"] for i, e in enumerate(tgt_entities)}

    anchors    = compute_intersection_anchors(data.sim_matrices, k)
    candidates = propagate_anchors(anchors, data, max_depth)
    layer1, layer2 = filter_layered(
        candidates,
        src_idx_to_label, tgt_idx_to_label,
        src_idx_to_uri, tgt_idx_to_uri,
        thresh1, thresh2, n_gram
    )

    psa = LayeredPSA(layer1=layer1, layer2=layer2, w1=w1, w2=w2)
    print("增强PSA构建完成：")
    psa.summary()
    return psa


def build_augmented_psa_from_files(
        data: OMData,
        src_json_path: str = "data/parsed/mouse.json",
        tgt_json_path: str = "data/parsed/human.json",
        k: float = 0.7,
        max_depth: int = 3,
        thresh1: float = 1.0,
        thresh2: float = 0.75,
        n_gram: int = 2,
        w1: float = 0.7,
        w2: float = 0.3,
        cache_path: str = "data/parsed/augmented_psa_cache.json") -> LayeredPSA:
    """从文件加载实体列表并构建分层增强PSA，支持缓存"""
    import hashlib

    # 用参数生成缓存key，参数变了自动重建
    cache_key = hashlib.md5(
        f"{k}_{max_depth}_{thresh1}_{thresh2}_{n_gram}_{w1}_{w2}".encode()
    ).hexdigest()[:8]

    # 尝试加载缓存
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("cache_key") == cache_key:
            layer1 = {tuple(p) for p in cached["layer1"]}
            layer2 = {tuple(p) for p in cached["layer2"]}
            psa = LayeredPSA(layer1=layer1, layer2=layer2, w1=w1, w2=w2)
            print(f"增强PSA从缓存加载：")
            psa.summary()
            return psa
        else:
            print("参数已变更，重新构建增强PSA...")

    # 缓存未命中，重新构建
    with open(src_json_path, encoding="utf-8") as f:
        src_entities = json.load(f)
    with open(tgt_json_path, encoding="utf-8") as f:
        tgt_entities = json.load(f)

    psa = build_augmented_psa(
        data, src_entities, tgt_entities,
        k, max_depth, thresh1, thresh2, n_gram, w1, w2
    )

    # 保存缓存
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        cache_data = {
            "cache_key": cache_key,
            "layer1": [list(p) for p in psa.layer1],
            "layer2": [list(p) for p in psa.layer2],
            "w1": w1, "w2": w2,
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False)
        print(f"增强PSA已缓存至: {cache_path}")

    if data.reference:
        for name, layer in [("Layer1", psa.layer1), ("Layer2", psa.layer2)]:
            correct = len(layer & data.reference)
            prec    = correct / len(layer) if layer else 0
            rec     = correct / len(data.reference)
            print(f"  {name}: P={prec:.4f}, R={rec:.4f}, correct={correct}")

    return psa