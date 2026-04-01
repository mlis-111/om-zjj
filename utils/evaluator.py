"""
utils/evaluator.py
计算本体匹配的评估指标。

提供两种评估：
  1. 标准评估（evaluate）：用完整标准对齐计算真实Precision/Recall/F1
  2. 近似评估（approximate_evaluate）：用PSA计算近似指标，用于GP适应度
"""
from typing import List, Set, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from utils.data_loader import OMData


def evaluate(alignment_matrix: np.ndarray,
             data: "OMData") -> Tuple[float, float, float]:
    """
    用完整标准对齐计算真实Precision/Recall/F1。
    用于最终评估，不用于GP进化中的适应度。

    Args:
        alignment_matrix: 0/1对齐矩阵 [M, N]
        data:             OMData（包含reference和实体索引）

    Returns:
        (precision, recall, f1)
    """
    if not data.reference:
        return 0.0, 0.0, 0.0

    # 将0/1矩阵转换为URI对集合
    predicted = set()
    rows, cols = np.where(alignment_matrix > 0.5)
    for r, c in zip(rows, cols):
        src_uri = data.src_idx_to_uri.get(int(r))
        tgt_uri = data.tgt_idx_to_uri.get(int(c))
        if src_uri and tgt_uri:
            predicted.add((src_uri, tgt_uri))

    return compute_f1(predicted, data.reference)


def compute_f1(predicted: Set[Tuple[str, str]],
               reference: Set[Tuple[str, str]]) -> Tuple[float, float, float]:
    """
    计算Precision/Recall/F1。

    Args:
        predicted: 预测的对应对集合
        reference: 标准对应对集合

    Returns:
        (precision, recall, f1)
    """
    if not predicted:
        return 0.0, 0.0, 0.0
    if not reference:
        return 0.0, 0.0, 0.0

    tp = len(predicted & reference)
    precision = tp / len(predicted)
    recall    = tp / len(reference)

    if precision + recall == 0:
        return precision, recall, 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def approximate_evaluate(alignment_matrix: np.ndarray,
                          sim_matrix: np.ndarray,
                          data: "OMData",
                          psa: Set[Tuple[str, str]]) -> Tuple[float, float, float]:
    """
    用PSA计算近似Precision/Recall/F1（用于GP适应度）。

    近似指标定义（参考MLHGP和DPGP-AML论文）：

    近似Recall：
      recalla = |Ap ∩ PSA| / |PSA|
      其中Ap是alignment中与PSA共享至少一个实体的子集

    近似Precision：
      precisiona = sqrt(avgSim(Ap) × correctness(Ap))
      avgSim = Ap∩PSA中对应对的平均相似度
      correctness = |Ap∩PSA| / |A|

    近似F1：
      fa = 2 × recalla × precisiona / (recalla + precisiona)

    Args:
        alignment_matrix: 0/1对齐矩阵 [M, N]
        sim_matrix:       组合树输出的综合相似度矩阵 [M, N]
        data:             OMData
        psa:              部分标准对齐

    Returns:
        (approx_precision, approx_recall, approx_f1)
    """
    if not psa:
        return 0.0, 0.0, 0.0

    # 将0/1矩阵转换为 {src_uri: tgt_uri} 映射
    rows, cols = np.where(alignment_matrix > 0.5)
    alignment_pairs = set()
    alignment_src_map = {}   # src_uri -> tgt_uri
    alignment_tgt_map = {}   # tgt_uri -> src_uri

    for r, c in zip(rows, cols):
        src_uri = data.src_idx_to_uri.get(int(r))
        tgt_uri = data.tgt_idx_to_uri.get(int(c))
        if src_uri and tgt_uri:
            alignment_pairs.add((src_uri, tgt_uri))
            alignment_src_map[src_uri] = tgt_uri
            alignment_tgt_map[tgt_uri] = src_uri

    if not alignment_pairs:
        return 0.0, 0.0, 0.0

    total_alignment = len(alignment_pairs)

    # 构建PSA的src/tgt URI集合
    psa_src_uris = {p[0] for p in psa}
    psa_tgt_uris = {p[1] for p in psa}

    # Ap：alignment中与PSA共享至少一个实体的子集
    ap = set()
    for src_uri, tgt_uri in alignment_pairs:
        if src_uri in psa_src_uris or tgt_uri in psa_tgt_uris:
            ap.add((src_uri, tgt_uri))

    # Ap ∩ PSA
    ap_intersect_psa = ap & psa
    n_correct = len(ap_intersect_psa)

    # 近似Recall
    recall_a = n_correct / len(psa)

    if n_correct == 0:
        return 0.0, recall_a, 0.0

    # 平均相似度（Ap∩PSA中对应对的相似度均值）
    sim_values = []
    for src_uri, tgt_uri in ap_intersect_psa:
        src_idx = data.src_uri_to_idx.get(src_uri)
        tgt_idx = data.tgt_uri_to_idx.get(tgt_uri)
        if src_idx is not None and tgt_idx is not None:
            sim_values.append(float(sim_matrix[src_idx, tgt_idx]))

    avg_sim = np.mean(sim_values) if sim_values else 0.0

    # correctness = |Ap∩PSA| / |A|
    correctness = n_correct / total_alignment

    # 近似Precision
    precision_a = float(np.sqrt(avg_sim * correctness))

    # 近似F1
    if precision_a + recall_a == 0:
        return precision_a, recall_a, 0.0

    f1_a = 2 * precision_a * recall_a / (precision_a + recall_a)
    return precision_a, recall_a, f1_a
