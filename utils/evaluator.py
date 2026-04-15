"""
utils/evaluator.py
评估指标计算。

提供两种评估：
  1. evaluate：用完整标准对齐计算真实Precision/Recall/F1（最终评估）
  2. approximate_evaluate：用增强PSA计算近似F1（GP适应度）
"""
from typing import Set, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from utils.data_loader import OMData


def compute_f1(predicted: Set[Tuple[str, str]],
               reference: Set[Tuple[str, str]]) -> Tuple[float, float, float]:
    """计算Precision/Recall/F1"""
    if not predicted or not reference:
        return 0.0, 0.0, 0.0
    tp        = len(predicted & reference)
    precision = tp / len(predicted)
    recall    = tp / len(reference)
    f1        = (2 * precision * recall / (precision + recall)
                 if precision + recall > 0 else 0.0)
    return precision, recall, f1


def evaluate(alignment_matrix: np.ndarray,
             data: "OMData") -> Tuple[float, float, float]:
    """
    用完整标准对齐计算真实Precision/Recall/F1。
    用于最终评估，不用于GP适应度。
    """
    if not data.reference:
        return 0.0, 0.0, 0.0

    rows, cols = np.where(alignment_matrix > 0.5)
    predicted = set()
    for r, c in zip(rows, cols):
        su = data.src_idx_to_uri.get(int(r))
        tu = data.tgt_idx_to_uri.get(int(c))
        if su and tu:
            predicted.add((su, tu))

    return compute_f1(predicted, data.reference)


def approximate_evaluate(alignment_matrix: np.ndarray,
                          sim_matrix: np.ndarray,
                          data: "OMData",
                          psa: Set[Tuple[str, str]]) -> Tuple[float, float, float]:
    """
    用增强PSA计算近似F1（用于GP适应度）。

    近似Recall：
      recall_a = |Ap ∩ PSA| / |PSA|
      其中 Ap 是 alignment 中与 PSA 共享至少一个实体的子集

    近似Precision：
      precision_a = sqrt(avgSim(Ap∩PSA) × correctness)
      correctness = |Ap∩PSA| / |A|

    Args:
        alignment_matrix: 0/1对齐矩阵 [M, N]
        sim_matrix:       组合树输出的综合相似度矩阵 [M, N]
        data:             OMData
        psa:              增强PSA，Set of (src_uri, tgt_uri)

    Returns:
        (approx_precision, approx_recall, approx_f1)
    """
    if not psa:
        return 0.0, 0.0, 0.0

    rows, cols = np.where(alignment_matrix > 0.5)
    alignment_pairs = set()
    for r, c in zip(rows, cols):
        su = data.src_idx_to_uri.get(int(r))
        tu = data.tgt_idx_to_uri.get(int(c))
        if su and tu:
            alignment_pairs.add((su, tu))

    if not alignment_pairs:
        return 0.0, 0.0, 0.0

    total_alignment = len(alignment_pairs)
    psa_src_uris    = {p[0] for p in psa}
    psa_tgt_uris    = {p[1] for p in psa}

    # Ap：alignment 中与 PSA 共享至少一个实体的子集
    ap = {(su, tu) for su, tu in alignment_pairs
          if su in psa_src_uris or tu in psa_tgt_uris}

    # 近似 Recall
    ap_correct = ap & psa
    n_correct  = len(ap_correct)
    recall_a   = n_correct / len(psa)

    if n_correct == 0:
        return 0.0, recall_a, 0.0

    # 平均相似度
    sim_values = []
    for su, tu in ap_correct:
        si = data.src_uri_to_idx.get(su)
        ti = data.tgt_uri_to_idx.get(tu)
        if si is not None and ti is not None:
            sim_values.append(float(sim_matrix[si, ti]))

    avg_sim     = float(np.mean(sim_values)) if sim_values else 0.0
    correctness = n_correct / total_alignment
    precision_a = float(np.sqrt(avg_sim * correctness))

    if precision_a + recall_a == 0:
        return precision_a, recall_a, 0.0

    f1_a = 2 * precision_a * recall_a / (precision_a + recall_a)
    return precision_a, recall_a, f1_a