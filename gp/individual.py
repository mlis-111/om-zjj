"""
gp/individual.py
双树个体定义：组合树（CombinationTree）+ 过滤树（FilterTree）。

组合树（CombinationTree）：
  - 叶节点：LLM模型名称（引用预计算的相似度矩阵）
  - 中间节点：算术操作（add/sub/mul/div/max/min/avg）
  - 根节点输出：float相似度矩阵 [M, N]

过滤树（FilterTree）：
  - 叶节点：过滤策略（对组合树输出的相似度矩阵应用过滤，得到0/1矩阵）
  - 中间节点：逻辑操作（intersection/union/xor）
  - 根节点输出：0/1对齐矩阵 [M, N]

编码方式（三段编码）：
  第一段：组合树（LLM类型+计算操作）
  第二段：过滤树（过滤操作+逻辑操作）
  第三段：两棵树的深度参数(a, b)
"""
import random
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from gp.operators import (
    ARITHMETIC_OPS, LOGICAL_OPS,
    ALL_FILTER_METHODS, FILTER_DEFAULT_PARAMS,
    apply_filter,
)
from utils.data_loader import LLM_TYPE_MAP, ALL_MODELS

if TYPE_CHECKING:
    from utils.data_loader import OMData


# ============================================================
# 树节点定义
# ============================================================

class CombinationNode:
    """
    组合树节点。
    - 叶节点：model_name 不为None，left/right为None
    - 中间节点：op_name 不为None，left/right为子节点
    """
    def __init__(self,
                 op_name: Optional[str] = None,
                 model_name: Optional[str] = None,
                 left: Optional["CombinationNode"] = None,
                 right: Optional["CombinationNode"] = None):
        self.op_name    = op_name      # 算术操作名
        self.model_name = model_name   # LLM模型名（叶节点）
        self.left       = left
        self.right      = right

    @property
    def is_leaf(self) -> bool:
        return self.model_name is not None

    def node_count(self) -> int:
        if self.is_leaf:
            return 1
        count = 1
        if self.left:
            count += self.left.node_count()
        if self.right:
            count += self.right.node_count()
        return count

    def get_leaf_models(self) -> List[str]:
        """返回所有叶节点的模型名称列表"""
        if self.is_leaf:
            return [self.model_name]
        models = []
        if self.left:
            models.extend(self.left.get_leaf_models())
        if self.right:
            models.extend(self.right.get_leaf_models())
        return models

    def evaluate(self, sim_matrices: Dict[str, np.ndarray]) -> np.ndarray:
        """递归计算组合树，返回float相似度矩阵"""
        if self.is_leaf:
            return sim_matrices[self.model_name].copy()
        left_val  = self.left.evaluate(sim_matrices)
        right_val = self.right.evaluate(sim_matrices)
        return ARITHMETIC_OPS[self.op_name](left_val, right_val)

    def collect_nodes(self) -> List["CombinationNode"]:
        """返回所有节点列表（包括自身）"""
        nodes = [self]
        if self.left:
            nodes.extend(self.left.collect_nodes())
        if self.right:
            nodes.extend(self.right.collect_nodes())
        return nodes


class FilterNode:
    """
    过滤树节点。
    - 叶节点：filter_method 不为None，left/right为None
    - 中间节点：logic_op 不为None，left/right为子节点
    """
    def __init__(self,
                 logic_op:      Optional[str] = None,
                 filter_method: Optional[str] = None,
                 filter_params: Optional[Dict] = None,
                 left:  Optional["FilterNode"] = None,
                 right: Optional["FilterNode"] = None):
        self.logic_op      = logic_op
        self.filter_method = filter_method
        self.filter_params = filter_params or {}
        self.left          = left
        self.right         = right

    @property
    def is_leaf(self) -> bool:
        return self.filter_method is not None

    def node_count(self) -> int:
        if self.is_leaf:
            return 1
        count = 1
        if self.left:
            count += self.left.node_count()
        if self.right:
            count += self.right.node_count()
        return count

    def evaluate(self, sim: np.ndarray,
                 data: "OMData") -> np.ndarray:
        """递归计算过滤树，返回0/1对齐矩阵"""
        if self.is_leaf:
            return apply_filter(
                self.filter_method, sim, data, **self.filter_params
            )
        left_val  = self.left.evaluate(sim, data)
        right_val = self.right.evaluate(sim, data)
        return LOGICAL_OPS[self.logic_op](left_val, right_val)

    def collect_nodes(self) -> List["FilterNode"]:
        """返回所有节点列表（包括自身）"""
        nodes = [self]
        if self.left:
            nodes.extend(self.left.collect_nodes())
        if self.right:
            nodes.extend(self.right.collect_nodes())
        return nodes


# ============================================================
# 随机树生成
# ============================================================

def _random_combination_tree(depth: int,
                              max_depth: int,
                              available_models: List[str]) -> CombinationNode:
    """
    随机生成组合树。
    depth=0或达到max_depth时强制生成叶节点。
    """
    if depth >= max_depth or (depth > 0 and random.random() < 0.3):
        # 叶节点：随机选一个模型
        model = random.choice(available_models)
        return CombinationNode(model_name=model)
    else:
        # 中间节点：随机选一个算术操作
        op = random.choice(list(ARITHMETIC_OPS.keys()))
        left  = _random_combination_tree(depth + 1, max_depth, available_models)
        right = _random_combination_tree(depth + 1, max_depth, available_models)
        return CombinationNode(op_name=op, left=left, right=right)


def _random_filter_tree(depth: int,
                        max_depth: int,
                        available_filters: List[str]) -> FilterNode:
    """
    随机生成过滤树。
    depth=0或达到max_depth时强制生成叶节点。
    """
    if depth >= max_depth or (depth > 0 and random.random() < 0.3):
        # 叶节点：随机选一个过滤方法
        method = random.choice(available_filters)
        params = FILTER_DEFAULT_PARAMS.get(method, {}).copy()
        return FilterNode(filter_method=method, filter_params=params)
    else:
        # 中间节点：随机选一个逻辑操作
        op = random.choice(list(LOGICAL_OPS.keys()))
        left  = _random_filter_tree(depth + 1, max_depth, available_filters)
        right = _random_filter_tree(depth + 1, max_depth, available_filters)
        return FilterNode(logic_op=op, left=left, right=right)


# ============================================================
# 个体定义
# ============================================================

class Individual:
    """
    GP个体：组合树 + 过滤树。

    Attributes:
        combination_tree: 组合树根节点
        filter_tree:      过滤树根节点
        fitness:          近似F1适应度（由fitness.py计算后赋值）
        llm_type_map:     模型名称->类型映射（用于多样性计算）
        _alignment_cache: 缓存当前对齐结果，避免重复计算
    """
    def __init__(self,
                 combination_tree: CombinationNode,
                 filter_tree:      FilterNode,
                 llm_type_map:     Dict[str, str] = None):
        self.combination_tree = combination_tree
        self.filter_tree      = filter_tree
        self.fitness: float   = 0.0
        self.llm_type_map     = llm_type_map or LLM_TYPE_MAP.copy()
        self._alignment_cache: Optional[np.ndarray] = None

    def evaluate(self, data: "OMData") -> np.ndarray:
        """
        解码个体，返回0/1对齐矩阵。
        结果会缓存，避免重复计算。
        """
        if self._alignment_cache is None:
            # Step1：组合树计算综合相似度矩阵
            sim = self.combination_tree.evaluate(data.sim_matrices)
            # Step2：过滤树生成0/1对齐矩阵
            self._alignment_cache = self.filter_tree.evaluate(sim, data)
        return self._alignment_cache

    def invalidate_cache(self):
        """修改个体结构后调用，清除缓存"""
        self._alignment_cache = None

    def get_alignment(self) -> List[Tuple[int, int]]:
        """
        返回当前对齐的实体对列表。
        注意：需要先调用evaluate()确保缓存存在。

        Returns:
            List of (src_idx, tgt_idx)
        """
        if self._alignment_cache is None:
            raise RuntimeError(
                "请先调用 evaluate(data) 再调用 get_alignment()"
            )
        rows, cols = np.where(self._alignment_cache > 0.5)
        return list(zip(rows.tolist(), cols.tolist()))

    def clone(self) -> "Individual":
        """深拷贝个体"""
        new_ind = Individual(
            combination_tree=deepcopy(self.combination_tree),
            filter_tree=deepcopy(self.filter_tree),
            llm_type_map=self.llm_type_map.copy(),
        )
        new_ind.fitness = self.fitness
        new_ind._alignment_cache = self._alignment_cache  # 复用缓存，无需重新计算
        return new_ind

    def __repr__(self) -> str:
        c_nodes = self.combination_tree.node_count()
        f_nodes = self.filter_tree.node_count()
        models  = self.combination_tree.get_leaf_models()
        return (f"Individual(fitness={self.fitness:.4f}, "
                f"c_nodes={c_nodes}, f_nodes={f_nodes}, "
                f"models={models})")


# ============================================================
# 种群初始化
# ============================================================

def create_individual(available_models: List[str],
                      available_filters: List[str],
                      min_depth: int = 2,
                      max_depth: int = 5,
                      llm_type_map: Dict[str, str] = None) -> Individual:
    """
    随机创建一个个体。

    Args:
        available_models:  可用的LLM模型名称列表
        available_filters: 可用的过滤方法列表
        min_depth:         最小树深度
        max_depth:         最大树深度
        llm_type_map:      模型类型映射

    Returns:
        随机初始化的Individual
    """
    # 随机选择深度（min_depth到max_depth之间）
    c_depth = random.randint(min_depth, max_depth)
    f_depth = random.randint(min_depth, max_depth)

    comb_tree   = _random_combination_tree(0, c_depth, available_models)
    filter_tree = _random_filter_tree(0, f_depth, available_filters)

    return Individual(
        combination_tree=comb_tree,
        filter_tree=filter_tree,
        llm_type_map=llm_type_map or LLM_TYPE_MAP.copy(),
    )


def create_population(size: int,
                      available_models: List[str],
                      available_filters: List[str],
                      min_depth: int = 2,
                      max_depth: int = 5,
                      llm_type_map: Dict[str, str] = None) -> List[Individual]:
    """
    创建初始种群。

    Args:
        size: 种群大小

    Returns:
        Individual列表
    """
    return [
        create_individual(
            available_models, available_filters,
            min_depth, max_depth, llm_type_map
        )
        for _ in range(size)
    ]
