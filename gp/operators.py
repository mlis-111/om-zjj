"""
gp/operators.py
双树的所有操作算子。

组合树算术操作（输入两个float矩阵，输出一个float矩阵）：
  add, sub, mul, div, max, min, avg

过滤树逻辑操作（输入两个0/1矩阵，输出一个0/1矩阵）：
  intersection, union, xor

过滤策略（输入一个float矩阵，输出一个0/1矩阵）：
  数值类：fixed_threshold, max_value, median, mean, var_mean, top_k
  推理类：kde, kmeans
  启发式：nde, stable_marriage, hungarian, random_hill_climbing
"""
import random
import warnings
from typing import Optional, TYPE_CHECKING

import numpy as np
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans

if TYPE_CHECKING:
    from utils.data_loader import OMData

# ============================================================
# 组合树算术操作
# ============================================================

def op_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.clip(a + b, 0.0, 1.0)

def op_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a - b)

def op_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b

def op_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    numerator   = np.minimum(a, b)
    denominator = np.maximum(a, b)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 1e-9, numerator / denominator, 0.0)

def op_max(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(a, b)

def op_min(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.minimum(a, b)

def op_avg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0

ARITHMETIC_OPS = {
    "add": op_add, "sub": op_sub, "mul": op_mul, "div": op_div,
    "max": op_max, "min": op_min, "avg": op_avg,
}

# ============================================================
# 过滤树逻辑操作
# ============================================================

def op_intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).astype(np.float32)

def op_union(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.clip(a + b, 0.0, 1.0).astype(np.float32)

def op_xor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a - b).astype(np.float32)

LOGICAL_OPS = {
    "intersection": op_intersection,
    "union":        op_union,
    "xor":          op_xor,
}

# ============================================================
# 过滤策略：数值类
# ============================================================

def filter_fixed_threshold(sim: np.ndarray, threshold: float = 0.7) -> np.ndarray:
    return (sim >= threshold).astype(np.float32)

def filter_max_value(sim: np.ndarray) -> np.ndarray:
    row_max  = sim.max(axis=1, keepdims=True)
    col_max  = sim.max(axis=0, keepdims=True)
    row_mask = (sim >= row_max) & (row_max > 0)
    col_mask = (sim >= col_max) & (col_max > 0)
    result   = np.zeros_like(sim, dtype=np.float32)
    result[row_mask & col_mask] = 1.0
    return result

def filter_median(sim: np.ndarray) -> np.ndarray:
    return (sim > np.median(sim, axis=1, keepdims=True)).astype(np.float32)

def filter_mean(sim: np.ndarray) -> np.ndarray:
    return (sim > sim.mean(axis=1, keepdims=True)).astype(np.float32)

def filter_var_mean(sim: np.ndarray) -> np.ndarray:
    means = sim.mean(axis=1, keepdims=True)
    stds  = sim.std(axis=1, keepdims=True)
    return (sim > means + stds).astype(np.float32)

def filter_top_k(sim: np.ndarray, k: int = 3) -> np.ndarray:
    result    = np.zeros_like(sim, dtype=np.float32)
    k         = min(k, sim.shape[1])
    top_k_idx = np.argpartition(sim, -k, axis=1)[:, -k:]
    result[np.arange(sim.shape[0])[:, None], top_k_idx] = 1.0
    return result

# ============================================================
# 过滤策略：推理类
# ============================================================

def filter_kde(sim: np.ndarray) -> np.ndarray:
    values = sim.flatten()
    values_sample = values[values > 0.01]
    if len(values_sample) < 10:
        return filter_mean(sim)
    if len(values_sample) > 50000:
        values_sample = values_sample[
            np.random.choice(len(values_sample), 50000, replace=False)
        ]
    try:
        kde       = gaussian_kde(values_sample)
        x         = np.linspace(values_sample.min(), values_sample.max(), 200)
        threshold = float(np.clip(x[np.argmin(kde(x))], 0.3, 0.9))
    except Exception:
        threshold = float(sim.mean())
    return (sim >= threshold).astype(np.float32)

def filter_kmeans(sim: np.ndarray, n_clusters: int = 2) -> np.ndarray:
    values = sim.flatten().reshape(-1, 1).astype(np.float64)
    sample = values[np.random.choice(len(values), min(10000, len(values)), replace=False)]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kmeans = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
            kmeans.fit(sample)
        centers        = kmeans.cluster_centers_.flatten()
        unique_centers = np.unique(np.round(centers, 6))
        threshold      = float(centers.max()) * 0.8 if len(unique_centers) >= n_clusters else float(sim.mean())
    except Exception:
        threshold = float(sim.mean())
    return (sim >= threshold).astype(np.float32)

# ============================================================
# 过滤策略：启发式类
# ============================================================

def filter_nde(sim: np.ndarray) -> np.ndarray:
    result = np.zeros_like(sim, dtype=np.float32)
    rows, cols = np.where(sim > 0)
    if len(rows) == 0:
        return result
    order       = np.argsort(-sim[rows, cols])
    rows, cols  = rows[order], cols[order]
    matched_src = set()
    matched_tgt = set()
    for i, j in zip(rows, cols):
        if i not in matched_src and j not in matched_tgt:
            result[i, j] = 1.0
            matched_src.add(i)
            matched_tgt.add(j)
    return result

def filter_stable_marriage(sim: np.ndarray) -> np.ndarray:
    result      = np.zeros_like(sim, dtype=np.float32)
    M, N        = sim.shape
    src_prefs   = [list(np.argsort(-sim[i])) for i in range(M)]
    tgt_rank    = np.argsort(-sim.T, axis=1)
    tgt_rank_map = [{tgt_rank[j][r]: r for r in range(M)} for j in range(N)]
    src_next    = [0] * M
    src_partner = [-1] * M
    tgt_partner = [-1] * N
    free_src    = list(range(M))
    while free_src:
        i = free_src.pop(0)
        if src_next[i] >= N:
            continue
        j = src_prefs[i][src_next[i]]
        src_next[i] += 1
        if tgt_partner[j] == -1:
            tgt_partner[j] = i
            src_partner[i] = j
        else:
            current = tgt_partner[j]
            if tgt_rank_map[j].get(i, M) < tgt_rank_map[j].get(current, M):
                tgt_partner[j] = i
                src_partner[i] = j
                src_partner[current] = -1
                free_src.append(current)
            else:
                free_src.append(i)
    for i, j in enumerate(src_partner):
        if j >= 0:
            result[i, j] = 1.0
    return result

def filter_hungarian(sim: np.ndarray) -> np.ndarray:
    from scipy.optimize import linear_sum_assignment
    M, N   = sim.shape
    size   = max(M, N)
    cost   = np.ones((size, size), dtype=np.float64)
    cost[:M, :N] = 1.0 - sim.astype(np.float64)
    row_ind, col_ind = linear_sum_assignment(cost)
    result = np.zeros_like(sim, dtype=np.float32)
    for r, c in zip(row_ind, col_ind):
        if r < M and c < N and sim[r, c] > 0:
            result[r, c] = 1.0
    return result

def filter_ant_colony(sim: np.ndarray,
                      n_ants: int = 10,
                      n_iterations: int = 20,
                      alpha: float = 1.0,
                      beta: float = 2.0,
                      rho: float = 0.1,
                      q: float = 1.0,
                      top_k_candidates: int = 50) -> np.ndarray:
    M, N          = sim.shape
    K             = min(top_k_candidates, N)
    top_k_indices = np.argsort(-sim, axis=1)[:, :K]
    heuristic     = sim.astype(np.float64) + 1e-6
    tau           = np.ones((M, N), dtype=np.float64) * 0.1
    best_alignment = None
    best_score     = -1.0
    for _ in range(n_iterations):
        all_alignments, all_scores = [], []
        for ant in range(n_ants):
            alignment   = np.zeros((M, N), dtype=np.float32)
            matched_tgt = set()
            for i in range(M):
                candidates = [j for j in top_k_indices[i] if j not in matched_tgt]
                if not candidates:
                    candidates = [j for j in range(N) if j not in matched_tgt]
                    if not candidates:
                        break
                cands    = np.array(candidates)
                probs    = (tau[i, cands] ** alpha) * (heuristic[i, cands] ** beta)
                prob_sum = probs.sum()
                j = int(np.random.choice(cands, p=probs/prob_sum)) if prob_sum > 1e-12 else int(np.random.choice(cands))
                alignment[i, j] = 1.0
                matched_tgt.add(j)
            score = float((alignment * sim).sum())
            all_alignments.append(alignment)
            all_scores.append(score)
            if score > best_score:
                best_score     = score
                best_alignment = alignment.copy()
        tau *= (1.0 - rho)
        best_idx = int(np.argmax(all_scores))
        if all_scores[best_idx] > 0:
            tau += (q / all_scores[best_idx]) * all_alignments[best_idx]
        tau = np.clip(tau, 1e-6, 10.0)
    return best_alignment if best_alignment is not None else filter_nde(sim)

def filter_random_hill_climbing(sim: np.ndarray, max_iter: int = 1000) -> np.ndarray:
    current       = filter_nde(sim)
    matched_pairs = list(zip(*np.where(current > 0)))
    for _ in range(max_iter):
        if len(matched_pairs) < 2:
            break
        idx_a = random.randint(0, len(matched_pairs) - 1)
        idx_b = random.randint(0, len(matched_pairs) - 1)
        if idx_a == idx_b:
            continue
        i_a, j_a = matched_pairs[idx_a]
        i_b, j_b = matched_pairs[idx_b]
        if j_a == j_b:
            continue
        if sim[i_a, j_b] + sim[i_b, j_a] > sim[i_a, j_a] + sim[i_b, j_b]:
            current[i_a, j_a] = current[i_b, j_b] = 0.0
            current[i_a, j_b] = current[i_b, j_a] = 1.0
            matched_pairs[idx_a] = (i_a, j_b)
            matched_pairs[idx_b] = (i_b, j_a)
    return current

# ============================================================
# 统一调用接口
# ============================================================

def apply_filter(method: str,
                 sim: np.ndarray,
                 data: Optional["OMData"] = None,
                 **kwargs) -> np.ndarray:
    dispatch = {
        "fixed_threshold":      lambda: filter_fixed_threshold(sim, threshold=kwargs.get("threshold", 0.7)),
        "max_value":            lambda: filter_max_value(sim),
        "median":               lambda: filter_median(sim),
        "mean":                 lambda: filter_mean(sim),
        "var_mean":             lambda: filter_var_mean(sim),
        "top_k":                lambda: filter_top_k(sim, k=kwargs.get("k", 3)),
        "kde":                  lambda: filter_kde(sim),
        "kmeans":               lambda: filter_kmeans(sim, n_clusters=kwargs.get("clusters", 2)),
        "nde":                  lambda: filter_nde(sim),
        "stable_marriage":      lambda: filter_stable_marriage(sim),
        "hungarian":            lambda: filter_hungarian(sim),
        "ant_colony":           lambda: filter_ant_colony(sim),
        "random_hill_climbing": lambda: filter_random_hill_climbing(sim),
    }
    if method not in dispatch:
        raise ValueError(f"未知的过滤方法: {method}")
    return dispatch[method]()


ALL_FILTER_METHODS = [
    "fixed_threshold", "max_value", "median", "mean", "var_mean", "top_k",
    "kde", "kmeans",
    "nde", "stable_marriage", "hungarian", "random_hill_climbing",
]

FILTER_DEFAULT_PARAMS = {
    "fixed_threshold":      {"threshold": 0.7},
    "max_value":            {},
    "median":               {},
    "mean":                 {},
    "var_mean":             {},
    "top_k":                {"k": 3},
    "kde":                  {},
    "kmeans":               {"clusters": 2},
    "nde":                  {},
    "stable_marriage":      {},
    "hungarian":            {},
    "ant_colony":           {},
    "random_hill_climbing": {},
}