"""
gp/selection.py
三阶段锦标赛选择算子。

阶段说明：
  第一阶段（8→4）：按近似F1淘汰，保留高质量个体
  第二阶段（4→2）：余弦退火动态权重综合得分（多样性+可靠性）
                    早期偏多样性（探索），后期偏可靠性（收敛）
  第三阶段（2→1）：按复杂度淘汰，保留结构更简洁的个体，相等则随机

多样性（信息熵）：
  统计组合树中三类LLM（string/structure/semantic）的比例
  H = -sum(pi*ln(pi))，归一化到[0,1]（除以ln(3)）

可靠性（层次一致性，O(k×L)算法）：
  只检查"源侧有层次关系的对"：
  对每个匹配对(c1,c2)，遍历c1的直接超类super_c1：
    若super_c1也在匹配对里，检查super_c2是否是c2的超类
    若不是 → 冲突
  reliability = 1 - (冲突数 / 有层次关系的对数)

复杂度：
  组合树节点数 + 过滤树节点数
"""
import math
import random
from typing import Dict, List, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from gp.individual import Individual
    from utils.data_loader import OMData


# ============================================================
# 余弦退火动态权重
# ============================================================

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


# ============================================================
# 各阶段指标计算
# ============================================================

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
                        data: "OMData",
                        max_pairs: int = 500) -> float:
    """
    计算个体对齐结果的可靠性（层次一致性）。

    算法：O(k×L)，只检查源侧有层次关系的对。

    对每个匹配对(c1,c2)，遍历c1的所有直接超类super_c1：
      - 若super_c1也在匹配对里（即super_c1→super_c2存在），
        则检查目标侧：super_c2是否是c2的超类
      - 若不是 → 冲突

    分母只统计"有层次关系的对数"，语义更精准。

    Args:
        individual: GP个体（需已调用evaluate()）
        data:       OMData（包含层次关系）
        max_pairs:  对齐数量过多时随机采样上限

    Returns:
        reliability in [0, 1]
    """
    alignment = individual.get_alignment()

    if len(alignment) < 2:
        # print("  对齐对数不足，可靠性设为0.0")
        return 0.0

    src_hier = data.src_hierarchy
    tgt_hier = data.tgt_hierarchy

    # src_to_tgt用完整对齐构建，确保超类查找不遗漏
    src_to_tgt: Dict[int, int] = {c1: c2 for c1, c2 in alignment}

    # 只对外层循环采样，减少计算量
    sample = alignment if len(alignment) <= max_pairs else random.sample(alignment, max_pairs)

    tgt_parents_set: Dict[int, Set[int]] = {
        k: set(v) for k, v in tgt_hier.parents.items()
    }

    conflict_count = 0
    checked: Set[Tuple[int, int]] = set()

    for c1, c2 in sample:
        for super_c1 in src_hier.parents.get(c1, []):
            if super_c1 not in src_to_tgt:
                continue
            pair_key = (min(c1, super_c1), max(c1, super_c1))
            if pair_key in checked:
                continue
            checked.add(pair_key)
            super_c2 = src_to_tgt[super_c1]
            if super_c2 not in tgt_parents_set.get(c2, set()):
                conflict_count += 1

    total = len(checked)
    if total == 0:
        # print("  无层次关系的对，可靠性设为1.0")
        return 1.0

    # print(f"  检查了 {total} 个有层次关系的对，发现 {conflict_count} 个冲突, 可靠性 = {1.0 - (conflict_count / total):.4f}")
    return 1.0 - (conflict_count / total)


def compute_complexity(individual: "Individual") -> int:
    """
    计算个体复杂度。
    复杂度 = 组合树节点数 + 过滤树节点数
    """
    return (individual.combination_tree.node_count() +
            individual.filter_tree.node_count())


# ============================================================
# 三阶段锦标赛
# ============================================================

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
        data:            OM数据
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

    # ---- 第二阶段（4→2）：余弦退火动态权重（多样性+可靠性）----
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