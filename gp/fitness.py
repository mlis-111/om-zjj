"""
gp/fitness.py
适应度评估模块（带全局缓存）。

缓存机制：
  基于个体树结构的字符串哈希作为key，缓存对齐矩阵和近似F1。
  相同树结构的个体直接复用缓存结果，避免重复计算。
  缓存在整个GP运行期间有效，每次run_gp调用前应手动清空。
"""
from typing import Dict, List, Set, Tuple, TYPE_CHECKING

import numpy as np

from utils.evaluator import approximate_evaluate

if TYPE_CHECKING:
    from gp.individual import Individual, CombinationNode, FilterNode
    from utils.data_loader import OMData


# ============================================================
# 全局缓存：key=tree_hash, value=(alignment_matrix, approx_f1)
# ============================================================
_EVAL_CACHE: Dict[str, Tuple[np.ndarray, float]] = {}


def clear_cache():
    """清空全局评估缓存（每次run_gp开始前调用）"""
    global _EVAL_CACHE
    _EVAL_CACHE.clear()


def cache_size() -> int:
    return len(_EVAL_CACHE)


# ============================================================
# 树结构哈希
# ============================================================

def _hash_combination_tree(node) -> str:
    if node.is_leaf:
        return node.model_name
    return f"({node.op_name}:{_hash_combination_tree(node.left)},{_hash_combination_tree(node.right)})"


def _hash_filter_tree(node) -> str:
    if node.is_leaf:
        params_str = ",".join(f"{k}={v}" for k, v in sorted(node.filter_params.items()))
        return f"{node.filter_method}({params_str})"
    return f"({node.logic_op}:{_hash_filter_tree(node.left)},{_hash_filter_tree(node.right)})"


def tree_hash(individual) -> str:
    """生成个体双树结构的唯一哈希字符串"""
    return (f"C={_hash_combination_tree(individual.combination_tree)}"
            f"|F={_hash_filter_tree(individual.filter_tree)}")


# ============================================================
# 适应度评估
# ============================================================

def evaluate_individual(individual,
                         data,
                         psa: Set[Tuple[str, str]]) -> float:
    """
    计算单个个体的适应度（近似F1），优先从缓存读取。
    """
    key = tree_hash(individual)

    if key in _EVAL_CACHE:
        alignment_matrix, approx_f1 = _EVAL_CACHE[key]
        individual._alignment_cache = alignment_matrix
        individual.fitness = approx_f1
        return approx_f1

    # 缓存未命中：正常计算
    sim_matrix       = individual.combination_tree.evaluate(data.sim_matrices)
    alignment_matrix = individual.filter_tree.evaluate(sim_matrix, data)
    individual._alignment_cache = alignment_matrix

    _, _, approx_f1 = approximate_evaluate(
        alignment_matrix, sim_matrix, data, psa
    )
    individual.fitness = approx_f1

    _EVAL_CACHE[key] = (alignment_matrix, approx_f1)
    return approx_f1


def evaluate_population(population: List,
                         data,
                         psa: Set[Tuple[str, str]]) -> None:
    """
    批量评估种群适应度（原地修改fitness字段）。
    相同树结构的个体只计算一次。
    """
    for ind in population:
        ind.invalidate_cache()
        evaluate_individual(ind, data, psa)