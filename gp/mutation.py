"""
gp/mutation.py
自适应变异算子。

变异策略：
  在进化30代后，统计前40%个体的编码分布，
  根据主流编码方式动态调整变异概率（主流操作概率提升10%）。

  片段三（树深度）对树结构影响最大，初始变异概率最小。

变异操作：
  - 组合树：随机替换一个节点（叶节点换模型，内节点换算术操作）
  - 过滤树：随机替换一个节点（叶节点换过滤方法，内节点换逻辑操作）
  - 深度变异：随机调整树的深度（±1）
"""
import random
from collections import Counter
from copy import deepcopy
from typing import Dict, List, Optional, TYPE_CHECKING

from gp.operators import (
    ARITHMETIC_OPS, LOGICAL_OPS,
    ALL_FILTER_METHODS, FILTER_DEFAULT_PARAMS,
)
from utils.data_loader import ALL_MODELS

if TYPE_CHECKING:
    from gp.individual import Individual, CombinationNode, FilterNode


# ============================================================
# 编码分布统计（用于自适应变异）
# ============================================================

def _count_combination_ops(population: List["Individual"],
                             top_ratio: float = 0.4) -> Dict[str, int]:
    """统计种群前top_ratio个体中组合树算术操作的分布"""
    n_top = max(1, int(len(population) * top_ratio))
    top_pop = sorted(population, key=lambda x: x.fitness, reverse=True)[:n_top]
    counts = Counter()
    for ind in top_pop:
        for node in ind.combination_tree.collect_nodes():
            if not node.is_leaf and node.op_name:
                counts[node.op_name] += 1
    return dict(counts)


def _count_filter_methods(population: List["Individual"],
                           top_ratio: float = 0.4) -> Dict[str, int]:
    """统计种群前top_ratio个体中过滤树叶节点方法的分布"""
    n_top = max(1, int(len(population) * top_ratio))
    top_pop = sorted(population, key=lambda x: x.fitness, reverse=True)[:n_top]
    counts = Counter()
    for ind in top_pop:
        for node in ind.filter_tree.collect_nodes():
            if node.is_leaf and node.filter_method:
                counts[node.filter_method] += 1
    return dict(counts)


def _adaptive_weights(counts: Dict[str, int],
                       all_ops: List[str],
                       boost: float = 1.1) -> Dict[str, float]:
    """
    根据计数分布计算自适应权重。
    主流操作（最高频）的权重提升boost倍，其余保持1.0。
    """
    if not counts:
        return {op: 1.0 for op in all_ops}
    max_op = max(counts, key=counts.get)
    weights = {}
    for op in all_ops:
        weights[op] = boost if op == max_op else 1.0
    return weights


def _weighted_choice(options: List[str],
                      weights: Dict[str, float]) -> str:
    """按权重随机选择"""
    w = [weights.get(op, 1.0) for op in options]
    total = sum(w)
    r = random.random() * total
    cumsum = 0.0
    for op, wi in zip(options, w):
        cumsum += wi
        if r <= cumsum:
            return op
    return options[-1]


# ============================================================
# 节点级变异
# ============================================================

def _mutate_combination_node(node: "CombinationNode",
                              available_models: List[str],
                              op_weights: Dict[str, float]) -> None:
    """原地变异组合树的一个节点"""
    if node.is_leaf:
        # 叶节点：换一个不同的模型
        other_models = [m for m in available_models if m != node.model_name]
        if other_models:
            node.model_name = random.choice(other_models)
    else:
        # 内节点：换一个不同的算术操作
        other_ops = [op for op in ARITHMETIC_OPS if op != node.op_name]
        if other_ops:
            node.op_name = _weighted_choice(other_ops, op_weights)


def _mutate_filter_node(node: "FilterNode",
                         available_filters: List[str],
                         method_weights: Dict[str, float]) -> None:
    """原地变异过滤树的一个节点"""
    if node.is_leaf:
        # 叶节点：换一个不同的过滤方法
        other_methods = [m for m in available_filters if m != node.filter_method]
        if other_methods:
            new_method = _weighted_choice(other_methods, method_weights)
            node.filter_method = new_method
            node.filter_params = FILTER_DEFAULT_PARAMS.get(new_method, {}).copy()
    else:
        # 内节点：换一个不同的逻辑操作
        other_ops = [op for op in LOGICAL_OPS if op != node.logic_op]
        if other_ops:
            node.logic_op = random.choice(other_ops)


# ============================================================
# 个体级变异
# ============================================================

def mutate(individual: "Individual",
           mutation_rate: float = 0.1,
           available_models: List[str] = None,
           available_filters: List[str] = None,
           op_weights: Dict[str, float] = None,
           method_weights: Dict[str, float] = None,
           max_depth: int = 5,
           depth_mutation_rate: float = 0.02) -> "Individual":
    """
    对个体进行变异。

    变异概率说明：
      - 组合树节点变异概率：mutation_rate（默认0.1）
      - 过滤树节点变异概率：mutation_rate
      - 深度变异概率：depth_mutation_rate（默认0.02，最小）

    Args:
        individual:         待变异个体（原地修改后返回）
        mutation_rate:      节点变异概率
        available_models:   可用模型列表
        available_filters:  可用过滤方法列表
        op_weights:         算术操作自适应权重
        method_weights:     过滤方法自适应权重
        max_depth:          最大树深度
        depth_mutation_rate: 深度变异概率（影响最大，概率最小）

    Returns:
        变异后的个体（已清除缓存）
    """
    if available_models is None:
        available_models = ALL_MODELS
    if available_filters is None:
        available_filters = ALL_FILTER_METHODS
    if op_weights is None:
        op_weights = {op: 1.0 for op in ARITHMETIC_OPS}
    if method_weights is None:
        method_weights = {m: 1.0 for m in available_filters}

    mutated = False

    # ---- 组合树节点变异 ----
    c_nodes = individual.combination_tree.collect_nodes()
    for node in c_nodes:
        if random.random() < mutation_rate:
            _mutate_combination_node(node, available_models, op_weights)
            mutated = True

    # ---- 过滤树节点变异 ----
    f_nodes = individual.filter_tree.collect_nodes()
    for node in f_nodes:
        if random.random() < mutation_rate:
            _mutate_filter_node(node, available_filters, method_weights)
            mutated = True

    # ---- 深度变异（概率最小，影响最大）----
    if random.random() < depth_mutation_rate:
        _depth_mutation(individual, available_models, available_filters, max_depth)
        mutated = True

    if mutated:
        individual.invalidate_cache()

    return individual


def _depth_mutation(individual: "Individual",
                     available_models: List[str],
                     available_filters: List[str],
                     max_depth: int) -> None:
    """
    深度变异：随机增加或减少一层树深度。
    50%概率增加深度（在随机叶节点处扩展），50%概率减少深度（随机裁剪子树为叶节点）。
    """
    from gp.individual import CombinationNode, FilterNode
    from gp.operators import FILTER_DEFAULT_PARAMS

    # 随机选择变异组合树还是过滤树
    if random.random() < 0.5:
        # 组合树深度变异
        nodes = individual.combination_tree.collect_nodes()
        if random.random() < 0.5:
            # 增加深度：在随机叶节点处扩展为内节点
            leaves = [n for n in nodes if n.is_leaf]
            if leaves:
                target = random.choice(leaves)
                # 将叶节点变为内节点
                op = random.choice(list(ARITHMETIC_OPS.keys()))
                left_model  = target.model_name
                right_model = random.choice(available_models)
                target.op_name    = op
                target.model_name = None
                target.left  = CombinationNode(model_name=left_model)
                target.right = CombinationNode(model_name=right_model)
        else:
            # 减少深度：随机将内节点替换为叶节点
            inner_nodes = [n for n in nodes if not n.is_leaf]
            if inner_nodes:
                target = random.choice(inner_nodes)
                model = random.choice(available_models)
                target.op_name    = None
                target.model_name = model
                target.left       = None
                target.right      = None
    else:
        # 过滤树深度变异
        nodes = individual.filter_tree.collect_nodes()
        if random.random() < 0.5:
            # 增加深度
            leaves = [n for n in nodes if n.is_leaf]
            if leaves:
                target = random.choice(leaves)
                op     = random.choice(list(LOGICAL_OPS.keys()))
                left_method  = target.filter_method
                left_params  = target.filter_params.copy()
                right_method = random.choice(available_filters)
                right_params = FILTER_DEFAULT_PARAMS.get(right_method, {}).copy()
                target.logic_op      = op
                target.filter_method = None
                target.filter_params = {}
                target.left  = FilterNode(filter_method=left_method,
                                           filter_params=left_params)
                target.right = FilterNode(filter_method=right_method,
                                           filter_params=right_params)
        else:
            # 减少深度
            inner_nodes = [n for n in nodes if not n.is_leaf]
            if inner_nodes:
                target = random.choice(inner_nodes)
                method = random.choice(available_filters)
                params = FILTER_DEFAULT_PARAMS.get(method, {}).copy()
                target.logic_op      = None
                target.filter_method = method
                target.filter_params = params
                target.left          = None
                target.right         = None


# ============================================================
# 种群级自适应变异
# ============================================================

def adaptive_mutate_population(population: List["Individual"],
                                 gen: int,
                                 mutation_rate: float = 0.1,
                                 available_models: List[str] = None,
                                 available_filters: List[str] = None,
                                 max_depth: int = 5,
                                 adaptive_start_gen: int = 30) -> None:
    """
    对整个种群进行自适应变异（原地修改）。

    进化30代后开始统计编码分布，动态调整变异权重。
    片段三（深度变异）的概率固定为最小值0.02。

    Args:
        population:        种群
        gen:               当前代数
        mutation_rate:     基础节点变异概率
        available_models:  可用模型
        available_filters: 可用过滤方法
        max_depth:         最大树深度
        adaptive_start_gen: 开始自适应调整的代数（默认30）
    """
    if available_models is None:
        available_models = ALL_MODELS
    if available_filters is None:
        available_filters = ALL_FILTER_METHODS

    # 30代后启动自适应权重
    if gen >= adaptive_start_gen:
        op_counts     = _count_combination_ops(population)
        method_counts = _count_filter_methods(population)
        op_weights     = _adaptive_weights(op_counts,     list(ARITHMETIC_OPS.keys()))
        method_weights = _adaptive_weights(method_counts, available_filters)
    else:
        op_weights     = {op: 1.0 for op in ARITHMETIC_OPS}
        method_weights = {m: 1.0 for m in available_filters}

    for ind in population:
        mutate(
            ind,
            mutation_rate=mutation_rate,
            available_models=available_models,
            available_filters=available_filters,
            op_weights=op_weights,
            method_weights=method_weights,
            max_depth=max_depth,
            depth_mutation_rate=0.02,
        )
