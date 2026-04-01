"""
gp/fitness.py
适应度评估模块。

使用PSA（部分标准对齐）计算近似F1作为适应度。
适应度越高，个体越好。
"""
from typing import List, Set, Tuple, TYPE_CHECKING

import numpy as np

from utils.evaluator import approximate_evaluate

if TYPE_CHECKING:
    from gp.individual import Individual
    from utils.data_loader import OMData


def evaluate_individual(individual: "Individual",
                         data: "OMData",
                         psa: Set[Tuple[str, str]]) -> float:
    """
    计算单个个体的适应度（近似F1）。

    流程：
      1. 组合树输出综合相似度矩阵
      2. 过滤树输出0/1对齐矩阵
      3. 用PSA计算近似F1

    Args:
        individual: GP个体
        data:       OMData
        psa:        部分标准对齐

    Returns:
        fitness in [0, 1]
    """
    # 获取组合树输出的综合相似度矩阵
    sim_matrix = individual.combination_tree.evaluate(data.sim_matrices)

    # 获取过滤树输出的0/1对齐矩阵（evaluate内部有缓存）
    alignment_matrix = individual.filter_tree.evaluate(sim_matrix, data)

    # 缓存到个体（供get_alignment使用）
    individual._alignment_cache = alignment_matrix

    # 计算近似F1
    _, _, approx_f1 = approximate_evaluate(
        alignment_matrix, sim_matrix, data, psa
    )

    individual.fitness = approx_f1
    return approx_f1


def evaluate_population(population: List["Individual"],
                         data: "OMData",
                         psa: Set[Tuple[str, str]]) -> None:
    """
    批量评估种群中所有个体的适应度（原地修改fitness字段）。

    Args:
        population: 个体列表
        data:       OMData
        psa:        部分标准对齐
    """
    for ind in population:
        ind.invalidate_cache()
        evaluate_individual(ind, data, psa)
