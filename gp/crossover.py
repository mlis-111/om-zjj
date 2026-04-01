"""
gp/crossover.py
分段交叉算子。

编码方式（三段）：
  第一段：组合树（LLM类型 + 算术操作）
  第二段：过滤树（过滤操作 + 逻辑操作）
  第三段：树的深度参数(a, b)

交叉策略：
  片段与片段之间交叉，即随机选择1~3段进行互换。
  交叉后对子代做深度裁剪，确保不超过max_depth。
"""
import random
from copy import deepcopy
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from gp.individual import Individual, CombinationNode, FilterNode


def _clip_combination_tree(node: "CombinationNode",
                            max_depth: int,
                            current_depth: int = 0,
                            available_models: List[str] = None) -> "CombinationNode":
    """
    递归裁剪组合树，超过max_depth的子树替换为随机叶节点。
    """
    from gp.individual import CombinationNode
    if node.is_leaf:
        return node
    if current_depth >= max_depth - 1:
        # 超深，替换为叶节点
        model = random.choice(available_models) if available_models else node.get_leaf_models()[0]
        return CombinationNode(model_name=model)
    node.left  = _clip_combination_tree(node.left,  max_depth, current_depth + 1, available_models)
    node.right = _clip_combination_tree(node.right, max_depth, current_depth + 1, available_models)
    return node


def _clip_filter_tree(node: "FilterNode",
                      max_depth: int,
                      current_depth: int = 0,
                      available_filters: List[str] = None) -> "FilterNode":
    """
    递归裁剪过滤树，超过max_depth的子树替换为随机叶节点。
    """
    from gp.individual import FilterNode
    from gp.operators import FILTER_DEFAULT_PARAMS
    if node.is_leaf:
        return node
    if current_depth >= max_depth - 1:
        method = random.choice(available_filters) if available_filters else node.filter_method or "nde"
        params = FILTER_DEFAULT_PARAMS.get(method, {}).copy()
        return FilterNode(filter_method=method, filter_params=params)
    node.left  = _clip_filter_tree(node.left,  max_depth, current_depth + 1, available_filters)
    node.right = _clip_filter_tree(node.right, max_depth, current_depth + 1, available_filters)
    return node


def subtree_crossover(parent_a: "Individual",
                      parent_b: "Individual",
                      crossover_rate: float = 0.8,
                      max_depth: int = 5) -> Tuple["Individual", "Individual"]:
    """
    分段交叉：随机选择1~3段进行互换。

    三段编码：
      段1：组合树
      段2：过滤树
      段3：（隐式，通过树深度体现，这里不单独处理）

    Args:
        parent_a:      父代个体A
        parent_b:      父代个体B
        crossover_rate: 交叉概率
        max_depth:     最大树深度

    Returns:
        两个子代个体
    """
    if random.random() > crossover_rate:
        return parent_a.clone(), parent_b.clone()

    child_a = parent_a.clone()
    child_b = parent_b.clone()

    # 随机决定交叉哪些段（至少交叉一段）
    swap_combination = random.random() < 0.7   # 70%概率交叉组合树
    swap_filter      = random.random() < 0.7   # 70%概率交叉过滤树

    # 确保至少交叉一段
    if not swap_combination and not swap_filter:
        if random.random() < 0.5:
            swap_combination = True
        else:
            swap_filter = True

    # 获取可用模型和过滤方法（从父代中推断）
    available_models  = list(set(
        parent_a.combination_tree.get_leaf_models() +
        parent_b.combination_tree.get_leaf_models()
    ))
    available_filters = list(set(
        _get_filter_methods(parent_a.filter_tree) +
        _get_filter_methods(parent_b.filter_tree)
    ))

    if swap_combination:
        # 交换组合树的随机子树
        nodes_a = child_a.combination_tree.collect_nodes()
        nodes_b = child_b.combination_tree.collect_nodes()

        # 随机选择交叉点（非根节点优先，但也可以选根节点）
        point_a = random.choice(nodes_a)
        # 从b中找同类型节点（叶换叶，内节点换内节点）
        same_type_b = [n for n in nodes_b if n.is_leaf == point_a.is_leaf]
        if not same_type_b:
            same_type_b = nodes_b
        point_b = random.choice(same_type_b)

        # 交换节点内容
        _swap_combination_nodes(point_a, point_b)

        # 裁剪超深子树
        child_a.combination_tree = _clip_combination_tree(
            child_a.combination_tree, max_depth, 0, available_models
        )
        child_b.combination_tree = _clip_combination_tree(
            child_b.combination_tree, max_depth, 0, available_models
        )

    if swap_filter:
        # 交换过滤树的随机子树
        nodes_a = child_a.filter_tree.collect_nodes()
        nodes_b = child_b.filter_tree.collect_nodes()

        point_a = random.choice(nodes_a)
        same_type_b = [n for n in nodes_b if n.is_leaf == point_a.is_leaf]
        if not same_type_b:
            same_type_b = nodes_b
        point_b = random.choice(same_type_b)

        _swap_filter_nodes(point_a, point_b)

        child_a.filter_tree = _clip_filter_tree(
            child_a.filter_tree, max_depth, 0, available_filters
        )
        child_b.filter_tree = _clip_filter_tree(
            child_b.filter_tree, max_depth, 0, available_filters
        )

    child_a.invalidate_cache()
    child_b.invalidate_cache()

    return child_a, child_b


def _swap_combination_nodes(node_a: "CombinationNode",
                             node_b: "CombinationNode") -> None:
    """原地交换两个组合树节点的内容（不改变指针，改变内容）"""
    # 交换所有字段
    node_a.op_name,    node_b.op_name    = node_b.op_name,    node_a.op_name
    node_a.model_name, node_b.model_name = node_b.model_name, node_a.model_name
    node_a.left,       node_b.left       = node_b.left,       node_a.left
    node_a.right,      node_b.right      = node_b.right,      node_a.right


def _swap_filter_nodes(node_a: "FilterNode",
                       node_b: "FilterNode") -> None:
    """原地交换两个过滤树节点的内容"""
    node_a.logic_op,      node_b.logic_op      = node_b.logic_op,      node_a.logic_op
    node_a.filter_method, node_b.filter_method = node_b.filter_method, node_a.filter_method
    node_a.filter_params, node_b.filter_params = node_b.filter_params, node_a.filter_params
    node_a.left,          node_b.left          = node_b.left,          node_a.left
    node_a.right,         node_b.right         = node_b.right,         node_a.right


def _get_filter_methods(node: "FilterNode") -> List[str]:
    """收集过滤树中所有叶节点的过滤方法"""
    if node.is_leaf:
        return [node.filter_method]
    methods = []
    if node.left:
        methods.extend(_get_filter_methods(node.left))
    if node.right:
        methods.extend(_get_filter_methods(node.right))
    return methods
