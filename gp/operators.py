"""
gp/operators.py
双树的所有操作算子。

组合树算术操作（输入两个float矩阵，输出一个float矩阵）：
  add, sub, mul, div, max, min, avg

过滤树逻辑操作（输入两个0/1矩阵，输出一个0/1矩阵）：
  intersection, union, xor

过滤策略（输入一个float矩阵，输出一个0/1矩阵）：
  数值类：fixed_threshold, max_value, median, mean, var_mean, top_k
  推理类：structure_filter, kde, kmeans
  启发式：nde, stable_marriage, hungarian, ant_colony, random_hill_climbing
"""
import random
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
    """逐元素相加，结果裁剪到[0,1]"""
    return np.clip(a + b, 0.0, 1.0)


def op_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """逐元素相减取绝对值"""
    return np.abs(a - b)


def op_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """逐元素相乘"""
    return a * b


def op_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    逐元素相除：较小值除以较大值。
    避免除零，分母为0时结果为0。
    """
    numerator = np.minimum(a, b)
    denominator = np.maximum(a, b)
    return np.where(denominator > 1e-9, numerator / denominator, 0.0)


def op_max(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """逐元素取最大值"""
    return np.maximum(a, b)


def op_min(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """逐元素取最小值"""
    return np.minimum(a, b)


def op_avg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """逐元素取平均值"""
    return (a + b) / 2.0


# 算术操作映射表
ARITHMETIC_OPS = {
    "add": op_add,
    "sub": op_sub,
    "mul": op_mul,
    "div": op_div,
    "max": op_max,
    "min": op_min,
    "avg": op_avg,
}


# ============================================================
# 过滤树逻辑操作
# ============================================================

def op_intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """交集（AND）：两者都为1才为1"""
    return (a * b).astype(np.float32)


def op_union(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """并集（OR）：任一为1则为1"""
    return np.clip(a + b, 0.0, 1.0).astype(np.float32)


def op_xor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """异或（XOR）：恰好一个为1时才为1"""
    return np.abs(a - b).astype(np.float32)


# 逻辑操作映射表
LOGICAL_OPS = {
    "intersection": op_intersection,
    "union":        op_union,
    "xor":          op_xor,
}


# ============================================================
# 过滤策略：数值类
# ============================================================

def filter_fixed_threshold(sim: np.ndarray,
                            threshold: float = 0.7) -> np.ndarray:
    """固定阈值过滤：超过阈值的置1"""
    return (sim >= threshold).astype(np.float32)


def filter_max_value(sim: np.ndarray) -> np.ndarray:
    """
    每行最大值置1，其余置0（行方向1:1约束）。
    每列最大值也置1（列方向1:1约束）。
    取两者交集，确保双向1:1。
    """
    result = np.zeros_like(sim, dtype=np.float32)
    # 行方向：每行最大值
    row_max = sim.max(axis=1, keepdims=True)
    row_mask = (sim >= row_max) & (row_max > 0)
    # 列方向：每列最大值
    col_max = sim.max(axis=0, keepdims=True)
    col_mask = (sim >= col_max) & (col_max > 0)
    result[(row_mask & col_mask)] = 1.0
    return result


def filter_median(sim: np.ndarray) -> np.ndarray:
    """每行超过中位数的置1"""
    medians = np.median(sim, axis=1, keepdims=True)
    return (sim > medians).astype(np.float32)


def filter_mean(sim: np.ndarray) -> np.ndarray:
    """每行超过均值的置1"""
    means = sim.mean(axis=1, keepdims=True)
    return (sim > means).astype(np.float32)


def filter_var_mean(sim: np.ndarray) -> np.ndarray:
    """每行超过（均值 + 标准差）的置1"""
    means = sim.mean(axis=1, keepdims=True)
    stds  = sim.std(axis=1, keepdims=True)
    return (sim > means + stds).astype(np.float32)


def filter_top_k(sim: np.ndarray, k: int = 3) -> np.ndarray:
    """每行前k大值置1"""
    result = np.zeros_like(sim, dtype=np.float32)
    k = min(k, sim.shape[1])
    top_k_idx = np.argpartition(sim, -k, axis=1)[:, -k:]
    rows = np.arange(sim.shape[0])[:, None]
    result[rows, top_k_idx] = 1.0
    return result


# ============================================================
# 过滤策略：推理类
# ============================================================

def filter_structure(sim: np.ndarray,
                     data: "OMData",
                     threshold: float = 0.6) -> np.ndarray:
    """
    结构过滤：父子节点相似度约束（向量化实现）。

    对矩阵中每个位置(i, j)：
      若src本体中i有父节点p，且tgt本体中j有父节点q，
      但所有父节点对(p,q)的相似度都 < threshold，则(i, j)置0。

    优化：先用阈值得到候选集，再批量检查父节点约束。
    """
    result = (sim >= threshold).astype(np.float32)

    src_parents = data.src_hierarchy.parents
    tgt_parents = data.tgt_hierarchy.parents

    # 只处理result中值为1且双方都有父节点的位置
    candidate_rows, candidate_cols = np.where(result > 0)

    for i, j in zip(candidate_rows.tolist(), candidate_cols.tolist()):
        i_parents = src_parents.get(i, [])
        j_parents = tgt_parents.get(j, [])
        if not i_parents or not j_parents:
            continue

        # 用numpy批量计算所有父节点对的相似度
        ip = [p for p in i_parents if p < sim.shape[0]]
        jp = [q for q in j_parents if q < sim.shape[1]]
        if not ip or not jp:
            continue

        # 取父节点子矩阵的最大值
        parent_sim_max = sim[np.ix_(ip, jp)].max()
        if parent_sim_max < threshold:
            result[i, j] = 0.0

    return result


def filter_kde(sim: np.ndarray) -> np.ndarray:
    """
    核密度估计过滤。
    对整个矩阵的相似度值拟合KDE，找到密度局部最小值作为阈值，
    超过阈值的置1。
    """
    values = sim.flatten()
    # 去掉极端值加速KDE拟合
    values_sample = values[values > 0.01]
    if len(values_sample) < 10:
        return filter_mean(sim)

    # 采样加速（最多取5万个点）
    if len(values_sample) > 50000:
        idx = np.random.choice(len(values_sample), 50000, replace=False)
        values_sample = values_sample[idx]

    try:
        kde = gaussian_kde(values_sample)
        x = np.linspace(values_sample.min(), values_sample.max(), 200)
        density = kde(x)

        # 找密度局部最小值（谷底）作为阈值
        threshold = x[np.argmin(density)]
        # 阈值限制在合理范围内
        threshold = np.clip(threshold, 0.3, 0.9)
    except Exception:
        threshold = sim.mean()

    return (sim >= threshold).astype(np.float32)


def filter_kmeans(sim: np.ndarray, n_clusters: int = 2) -> np.ndarray:
    """
    K-Means聚类过滤。
    将相似度值聚为n_clusters类，高相似度簇置1。
    """
    values = sim.flatten().reshape(-1, 1).astype(np.float64)

    # 采样加速（最多取1万个点）
    if len(values) > 10000:
        idx = np.random.choice(len(values), 10000, replace=False)
        sample = values[idx]
    else:
        sample = values

    try:
        kmeans = KMeans(n_clusters=n_clusters, n_init=3, random_state=42)
        kmeans.fit(sample)
        # 高相似度簇的中心
        centers = kmeans.cluster_centers_.flatten()
        high_center = centers.max()
        # 阈值取高相似度簇中心的80%
        threshold = high_center * 0.8
    except Exception:
        threshold = sim.mean()

    return (sim >= threshold).astype(np.float32)


# ============================================================
# 过滤策略：启发式类
# ============================================================

def filter_nde(sim: np.ndarray) -> np.ndarray:
    """
    朴素降序提取（NDE）。
    按相似度降序排列所有候选对，贪心选择不冲突的1:1对应。
    """
    result = np.zeros_like(sim, dtype=np.float32)
    M, N = sim.shape

    # 获取所有(i,j,sim_val)并降序排列
    rows, cols = np.where(sim > 0)
    vals = sim[rows, cols]
    order = np.argsort(-vals)
    rows, cols = rows[order], cols[order]

    matched_src = set()
    matched_tgt = set()

    for i, j in zip(rows, cols):
        if i not in matched_src and j not in matched_tgt:
            result[i, j] = 1.0
            matched_src.add(i)
            matched_tgt.add(j)

    return result


def filter_stable_marriage(sim: np.ndarray) -> np.ndarray:
    """
    稳定婚姻算法（Gale-Shapley）。
    保证1:1稳定匹配，每个src实体最多匹配一个tgt实体。
    以src为求婚方（proposer），tgt为接受方（receiver）。
    """
    result = np.zeros_like(sim, dtype=np.float32)
    M, N = sim.shape

    # src偏好列表（按相似度降序排列的tgt索引）
    src_prefs = [list(np.argsort(-sim[i])) for i in range(M)]
    # tgt偏好：相似度越高越好
    # tgt_rank[j][i] = i在j的偏好中的排名（越小越好）
    tgt_rank = np.argsort(-sim.T, axis=1)  # [N, M]
    tgt_rank_map = [
        {tgt_rank[j][r]: r for r in range(M)}
        for j in range(N)
    ]

    # Gale-Shapley
    src_next = [0] * M           # 每个src下一个要求婚的tgt索引
    src_partner = [-1] * M       # src当前匹配的tgt
    tgt_partner = [-1] * N       # tgt当前匹配的src
    free_src = list(range(M))

    while free_src:
        i = free_src.pop(0)
        if src_next[i] >= N:
            continue
        j = src_prefs[i][src_next[i]]
        src_next[i] += 1

        if tgt_partner[j] == -1:
            # tgt j 未匹配，直接接受
            tgt_partner[j] = i
            src_partner[i] = j
        else:
            # tgt j 已匹配，比较偏好
            current = tgt_partner[j]
            if tgt_rank_map[j].get(i, M) < tgt_rank_map[j].get(current, M):
                # i 更受 j 青睐
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
    """
    匈牙利算法（最优1:1匹配）。
    将相似度转为代价（1-sim），用scipy.optimize.linear_sum_assignment求解。
    若矩阵非方阵，补全为方阵后求解，再截取有效部分。
    """
    from scipy.optimize import linear_sum_assignment

    M, N = sim.shape
    # 补全为方阵
    size = max(M, N)
    cost = np.ones((size, size), dtype=np.float64)
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
    """
    蚁群算法（完整实现，大矩阵优化版）。
    求解本体匹配问题的1:1最优对齐。

    大矩阵优化：每个src实体只从相似度Top-K的tgt候选中选择，
    而不是全部N个tgt，大幅减少计算量。

    Args:
        sim:               相似度矩阵 [M, N]
        n_ants:            蚂蚁数量
        n_iterations:      迭代次数
        alpha:             信息素重要程度
        beta:              启发因子重要程度
        rho:               信息素挥发率
        q:                 信息素强度常数
        top_k_candidates:  每个src只考虑相似度最高的K个tgt候选

    Returns:
        0/1对齐矩阵
    """
    M, N = sim.shape
    K = min(top_k_candidates, N)

    # 为每个src预计算Top-K候选tgt（按相似度降序）
    top_k_indices = np.argsort(-sim, axis=1)[:, :K]  # [M, K]

    heuristic = sim.astype(np.float64) + 1e-6
    tau = np.ones((M, N), dtype=np.float64) * 0.1

    best_alignment = None
    best_score = -1.0

    for iteration in range(n_iterations):
        all_alignments = []
        all_scores = []

        for ant in range(n_ants):
            alignment = np.zeros((M, N), dtype=np.float32)
            matched_tgt = set()

            for i in range(M):
                # 只从Top-K候选中选择未匹配的
                candidates = [j for j in top_k_indices[i] if j not in matched_tgt]
                if not candidates:
                    # 降级：从所有未匹配中随机选
                    all_available = [j for j in range(N) if j not in matched_tgt]
                    if not all_available:
                        break
                    candidates = all_available

                # 批量计算转移概率（向量化）
                cands = np.array(candidates)
                tau_vals = tau[i, cands] ** alpha
                heur_vals = heuristic[i, cands] ** beta
                probs = tau_vals * heur_vals
                prob_sum = probs.sum()

                if prob_sum < 1e-12:
                    j = int(np.random.choice(cands))
                else:
                    probs /= prob_sum
                    j = int(np.random.choice(cands, p=probs))

                alignment[i, j] = 1.0
                matched_tgt.add(j)

            score = float((alignment * sim).sum())
            all_alignments.append(alignment)
            all_scores.append(score)

            if score > best_score:
                best_score = score
                best_alignment = alignment.copy()

        # 信息素挥发
        tau *= (1.0 - rho)

        # 精英策略更新
        best_ant_idx = int(np.argmax(all_scores))
        best_ant_score = all_scores[best_ant_idx]
        if best_ant_score > 0:
            delta = q / best_ant_score
            tau += delta * all_alignments[best_ant_idx]

        tau = np.clip(tau, 1e-6, 10.0)

    return best_alignment if best_alignment is not None else filter_nde(sim)


def filter_random_hill_climbing(sim: np.ndarray,
                                 max_iter: int = 1000) -> np.ndarray:
    """
    随机爬山算法。
    从NDE初始解出发，通过随机交换操作进行局部搜索。
    """
    # 初始解：使用NDE
    current = filter_nde(sim)
    current_score = float((current * sim).sum())

    M, N = sim.shape
    matched_pairs = list(zip(*np.where(current > 0)))  # [(i,j), ...]

    for _ in range(max_iter):
        if len(matched_pairs) < 2:
            break

        # 随机选两个已匹配对，尝试交换tgt
        idx_a = random.randint(0, len(matched_pairs) - 1)
        idx_b = random.randint(0, len(matched_pairs) - 1)
        if idx_a == idx_b:
            continue

        i_a, j_a = matched_pairs[idx_a]
        i_b, j_b = matched_pairs[idx_b]

        if j_a == j_b:
            continue

        # 计算交换后的得分变化
        old_contrib = sim[i_a, j_a] + sim[i_b, j_b]
        new_contrib = sim[i_a, j_b] + sim[i_b, j_a]

        if new_contrib > old_contrib:
            # 接受改进
            current[i_a, j_a] = 0.0
            current[i_b, j_b] = 0.0
            current[i_a, j_b] = 1.0
            current[i_b, j_a] = 1.0
            matched_pairs[idx_a] = (i_a, j_b)
            matched_pairs[idx_b] = (i_b, j_a)
            current_score += (new_contrib - old_contrib)

    return current


# ============================================================
# 过滤策略统一调用接口
# ============================================================

def apply_filter(method: str,
                 sim: np.ndarray,
                 data: Optional["OMData"] = None,
                 **kwargs) -> np.ndarray:
    """
    统一的过滤策略调用接口。

    Args:
        method: 过滤方法名称
        sim:    相似度矩阵 [M, N]
        data:   OMData（structure_filter需要，其余可为None）
        **kwargs: 各方法的额外参数

    Returns:
        0/1对齐矩阵 [M, N]
    """
    if method == "fixed_threshold":
        return filter_fixed_threshold(
            sim, threshold=kwargs.get("threshold", 0.7)
        )
    elif method == "max_value":
        return filter_max_value(sim)
    elif method == "median":
        return filter_median(sim)
    elif method == "mean":
        return filter_mean(sim)
    elif method == "var_mean":
        return filter_var_mean(sim)
    elif method == "top_k":
        return filter_top_k(sim, k=kwargs.get("k", 3))
    elif method == "structure_filter":
        if data is None:
            raise ValueError("structure_filter 需要传入 data 参数")
        return filter_structure(
            sim, data, threshold=kwargs.get("threshold", 0.6)
        )
    elif method == "kde":
        return filter_kde(sim)
    elif method == "kmeans":
        return filter_kmeans(sim, n_clusters=kwargs.get("clusters", 2))
    elif method == "nde":
        return filter_nde(sim)
    elif method == "stable_marriage":
        return filter_stable_marriage(sim)
    elif method == "hungarian":
        return filter_hungarian(sim)
    elif method == "ant_colony":
        return filter_ant_colony(sim)
    elif method == "random_hill_climbing":
        return filter_random_hill_climbing(sim)
    else:
        raise ValueError(f"未知的过滤方法: {method}")


# 所有过滤方法名称列表（用于GP随机选择）
ALL_FILTER_METHODS = [
    "fixed_threshold", "max_value", "median", "mean", "var_mean", "top_k",
    "structure_filter", "kde", "kmeans",
    "nde", "stable_marriage", "hungarian", "ant_colony", "random_hill_climbing",
]

# 各方法默认参数
FILTER_DEFAULT_PARAMS = {
    "fixed_threshold":    {"threshold": 0.7},
    "max_value":          {},
    "median":             {},
    "mean":               {},
    "var_mean":           {},
    "top_k":              {"k": 3},
    "structure_filter":   {"threshold": 0.6},
    "kde":                {},
    "kmeans":             {"clusters": 2},
    "nde":                {},
    "stable_marriage":    {},
    "hungarian":          {},
    "ant_colony":         {},
    "random_hill_climbing": {},
}