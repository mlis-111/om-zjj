"""
gp/selection.py
三阶段锦标赛选择算子。

阶段说明：
  第一阶段（8→4）：按近似F1淘汰，保留高质量个体
  第二阶段（4→2）：按动态权重综合得分（多样性+可靠性）选择
                    余弦退火调整权重：早期偏多样性，后期偏可靠性
  第三阶段（2→1）：按复杂度淘汰，保留结构更简洁的个体，相等则随机

多样性（信息熵）：
  统计组合树中三类LLM（string/structure/semantic）的比例
  H = -sum(pi*ln(pi))，归一化到[0,1]（除以ln(3)）

可靠性：
  检测对齐中的逻辑冲突（一侧有层次关系但另一侧没有）
  reliability = 1 - (冲突数 / 总比较次数)

复杂度：
  组合树节点数 + 过滤树节点数
"""
import math
import random
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from gp.individual import Individual
    from utils.data_loader import OMData


def cosine_annealing_weights(gen: int, max_gen: int) -> Tuple[float, float]:
    """
    余弦退火动态权重。
    早期：w_div≈1（探索多样性）
    后期：w_rel≈1（收敛可靠性）

    Returns:
        (w_div, w_rel)，两者之和为1
    """
    progress = gen / max(max_gen - 1, 1)
    w_div = 0.5 * (1.0 + math.cos(math.pi * progress))
    w_rel = 1.0 - w_div
    return w_div, w_rel


def compute_diversity(individual: "Individual") -> float:
    """
    计算个体的LLM类型多样性（归一化信息熵）。

    Returns:
        diversity in [0, 1]
    """
    counts = {"string": 0, "structure": 0, "semantic": 0}
    for model_name in individual.combination_tree.get_leaf_models():
        llm_type = individual.llm_type_map.get(model_name, "string")
        counts[llm_type] = counts.get(llm_type, 0) + 1

    total = sum(counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for cnt in counts.values():
        if cnt > 0:
            p = cnt / total
            entropy -= p * math.log(p)

    max_entropy = math.log(3)
    return entropy / max_entropy if max_entropy > 0 else 0.0


def compute_reliability(individual: "Individual",
                        data: "OMData") -> float:
    """
    计算个体对齐结果的可靠性。

    逻辑冲突定义：
      对 c₁→c₂ 和 c₁'→c₂'，若 c₁与c₁'有层次关系 但 c₂与c₂'没有（或反之），
      则为冲突。

    Returns:
        reliability in [0, 1]
    """
    alignment = individual.get_alignment()  # List of (src_idx, tgt_idx)

    if len(alignment) < 2:
        return 1.0

    total_comparisons = 0
    conflict_count = 0
    src_hier = data.src_hierarchy
    tgt_hier = data.tgt_hierarchy

    n = len(alignment)
    for i in range(n):
        for j in range(i + 1, n):
            src_i, tgt_i = alignment[i]
            src_j, tgt_j = alignment[j]
            total_comparisons += 1
            src_has_rel = src_hier.has_any_relation(src_i, src_j)
            tgt_has_rel = tgt_hier.has_any_relation(tgt_i, tgt_j)
            if src_has_rel != tgt_has_rel:
                conflict_count += 1

    if total_comparisons == 0:
        return 1.0
    return 1.0 - (conflict_count / total_comparisons)


def compute_complexity(individual: "Individual") -> int:
    """
    计算个体复杂度。
    复杂度 = 组合树节点数 + 过滤树节点数
    """
    return (individual.combination_tree.node_count() +
            individual.filter_tree.node_count())


def three_stage_tournament(population: List["Individual"],
                           gen: int,
                           max_gen: int,
                           data: "OMData",
                           tournament_size: int = 8) -> "Individual":
    """
    三阶段锦标赛选择，返回一个父代个体。

    Args:
        population:      当前种群
        gen:             当前代数（从0开始）
        max_gen:         最大代数
        data:            OM数据（用于可靠性计算）
        tournament_size: 锦标赛大小，默认8

    Returns:
        选中的父代个体
    """
    candidates = random.sample(
        population, min(tournament_size, len(population))
    )

    # ---- 第一阶段（8→4）：按近似F1淘汰 ----
    random.shuffle(candidates)
    survivors_1 = []
    for i in range(0, len(candidates) - 1, 2):
        a, b = candidates[i], candidates[i + 1]
        survivors_1.append(a if a.fitness >= b.fitness else b)
    if len(candidates) % 2 == 1:
        survivors_1.append(candidates[-1])

    if len(survivors_1) == 1:
        return survivors_1[0]

    # ---- 第二阶段（4→2）：动态权重综合得分（多样性+可靠性）----
    w_div, w_rel = cosine_annealing_weights(gen, max_gen)
    random.shuffle(survivors_1)
    survivors_2 = []
    for i in range(0, len(survivors_1) - 1, 2):
        a, b = survivors_1[i], survivors_1[i + 1]
        score_a = w_div * compute_diversity(a) + w_rel * compute_reliability(a, data)
        score_b = w_div * compute_diversity(b) + w_rel * compute_reliability(b, data)
        survivors_2.append(a if score_a >= score_b else b)
    if len(survivors_1) % 2 == 1:
        survivors_2.append(survivors_1[-1])

    if len(survivors_2) == 1:
        return survivors_2[0]

    # ---- 第三阶段（2→1）：按复杂度淘汰，保留更简洁的 ----
    a, b = survivors_2[0], survivors_2[1]
    comp_a = compute_complexity(a)
    comp_b = compute_complexity(b)

    if comp_a < comp_b:
        return a
    elif comp_b < comp_a:
        return b
    else:
        return random.choice([a, b])


def elite_selection(population: List["Individual"],
                    elite_ratio: float = 0.1) -> List["Individual"]:
    """
    精英保留：选取适应度最高的前elite_ratio比例个体。
    """
    n_elite = max(1, int(len(population) * elite_ratio))
    sorted_pop = sorted(
        population, key=lambda ind: ind.fitness, reverse=True
    )
    return sorted_pop[:n_elite]