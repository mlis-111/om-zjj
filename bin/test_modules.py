"""
test_modules.py
单元测试：验证所有确定性模块的输入输出是否符合预期。
跳过有随机性的模块（individual初始化、crossover、mutation、gp_engine）。

运行：
  cd /data/zjj/om-zjj
  python test_modules.py
"""
import sys
import os
import math
import traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

# ============================================================
# 测试工具
# ============================================================

PASS = 0
FAIL = 0

def check(name: str, condition: bool, msg: str = ""):
    global PASS, FAIL
    if condition:
        print(f"  [PASS] {name}")
        PASS += 1
    else:
        print(f"  [FAIL] {name}" + (f": {msg}" if msg else ""))
        FAIL += 1

def approx_eq(a, b, tol=1e-5):
    return abs(a - b) < tol


# ============================================================
# 1. operators.py：算术操作
# ============================================================

def test_arithmetic_ops():
    print("\n=== 1. 算术操作 ===")
    from gp.operators import op_add, op_sub, op_mul, op_div, op_max, op_min, op_avg

    a = np.array([[0.8, 0.2], [0.5, 0.5]], dtype=np.float32)
    b = np.array([[0.4, 0.6], [0.3, 0.7]], dtype=np.float32)

    # add：裁剪到[0,1]
    r = op_add(a, b)
    check("op_add clip", r[0,0] == 1.0 and approx_eq(r[0,1], 0.8))

    # sub：绝对值
    r = op_sub(a, b)
    check("op_sub abs", approx_eq(r[0,0], 0.4) and approx_eq(r[0,1], 0.4))

    # mul
    r = op_mul(a, b)
    check("op_mul", approx_eq(r[0,0], 0.32) and approx_eq(r[1,1], 0.35))

    # div：min/max
    r = op_div(a, b)
    check("op_div", approx_eq(r[0,0], 0.5))  # 0.4/0.8=0.5
    # div 除零
    z = np.zeros((2,2), dtype=np.float32)
    r = op_div(z, z)
    check("op_div zero", float(r.sum()) == 0.0)

    # max
    r = op_max(a, b)
    check("op_max", approx_eq(r[0,0], 0.8) and approx_eq(r[0,1], 0.6))

    # min
    r = op_min(a, b)
    check("op_min", approx_eq(r[0,0], 0.4) and approx_eq(r[0,1], 0.2))

    # avg
    r = op_avg(a, b)
    check("op_avg", approx_eq(r[0,0], 0.6) and approx_eq(r[0,1], 0.4))


# ============================================================
# 2. operators.py：逻辑操作
# ============================================================

def test_logical_ops():
    print("\n=== 2. 逻辑操作 ===")
    from gp.operators import op_intersection, op_union, op_xor

    a = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    b = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32)

    r = op_intersection(a, b)
    check("intersection", r[0,0]==1.0 and r[0,1]==0.0 and r[1,0]==0.0 and r[1,1]==1.0)

    r = op_union(a, b)
    check("union", r[0,0]==1.0 and r[0,1]==1.0 and r[1,0]==1.0 and r[1,1]==1.0)

    r = op_xor(a, b)
    check("xor", r[0,0]==0.0 and r[0,1]==1.0 and r[1,0]==1.0 and r[1,1]==0.0)


# ============================================================
# 3. operators.py：过滤策略（确定性部分）
# ============================================================

def test_filter_ops():
    print("\n=== 3. 过滤策略 ===")
    from gp.operators import (filter_fixed_threshold, filter_max_value,
                               filter_median, filter_mean, filter_var_mean,
                               filter_top_k, filter_nde)

    sim = np.array([
        [0.9, 0.3, 0.5],
        [0.2, 0.8, 0.4],
        [0.1, 0.6, 0.7],
    ], dtype=np.float32)

    # fixed_threshold=0.6
    r = filter_fixed_threshold(sim, threshold=0.6)
    check("fixed_threshold", r[0,0]==1.0 and r[0,1]==0.0 and r[1,1]==1.0)

    # max_value：行列双向最大
    r = filter_max_value(sim)
    check("max_value ones>=1", r.sum() >= 1)
    check("max_value (0,0)=1", r[0,0] == 1.0)  # 0.9是第0行最大且第0列最大

    # median：每行超过中位数
    r = filter_median(sim)
    # 第0行中位数=0.5，超过0.5的是0.9
    check("median row0", r[0,0]==1.0 and r[0,1]==0.0 and r[0,2]==0.0)

    # mean：每行超过均值
    r = filter_mean(sim)
    # 第0行均值=(0.9+0.3+0.5)/3=0.567，超过的只有0.9
    check("mean row0", r[0,0]==1.0 and r[0,1]==0.0)

    # var_mean：每行超过均值+std
    r = filter_var_mean(sim)
    check("var_mean shape", r.shape == sim.shape)
    check("var_mean range", r.min() >= 0 and r.max() <= 1)

    # top_k=1：每行只选最大值
    r = filter_top_k(sim, k=1)
    check("top_k=1 row0", r[0,0]==1.0 and r[0,1]==0.0 and r[0,2]==0.0)
    check("top_k=1 row1", r[1,0]==0.0 and r[1,1]==1.0)
    check("top_k=1 ones", int(r.sum()) == 3)

    # nde：贪心1:1
    r = filter_nde(sim)
    check("nde 1:1 src", int(r.sum(axis=1).max()) <= 1)
    check("nde 1:1 tgt", int(r.sum(axis=0).max()) <= 1)
    check("nde (0,0)=1", r[0,0] == 1.0)  # 0.9最大，先选


# ============================================================
# 4. evaluator.py
# ============================================================

def test_evaluator():
    print("\n=== 4. evaluator ===")
    from utils.evaluator import compute_f1, evaluate, approximate_evaluate
    from utils.augmented_psa import LayeredPSA

    # compute_f1
    pred = {("a","x"), ("b","y"), ("c","z")}
    ref  = {("a","x"), ("b","y"), ("d","w")}
    p, r, f1 = compute_f1(pred, ref)
    check("compute_f1 precision", approx_eq(p, 2/3))
    check("compute_f1 recall",    approx_eq(r, 2/3))
    check("compute_f1 f1",        approx_eq(f1, 2/3))

    # compute_f1 空集
    p, r, f1 = compute_f1(set(), ref)
    check("compute_f1 empty pred", p==0 and r==0 and f1==0)

    # approximate_evaluate with LayeredPSA
    # 构造简单的mock data
    class MockData:
        src_idx_to_uri = {0: "s0", 1: "s1", 2: "s2"}
        tgt_idx_to_uri = {0: "t0", 1: "t1", 2: "t2"}
        src_uri_to_idx = {"s0": 0, "s1": 1, "s2": 2}
        tgt_uri_to_idx = {"t0": 0, "t1": 1, "t2": 2}

    alignment = np.zeros((3, 3), dtype=np.float32)
    alignment[0, 0] = 1.0  # s0→t0
    alignment[1, 1] = 1.0  # s1→t1

    sim = np.array([
        [0.9, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.7],
    ], dtype=np.float32)

    layer1 = {("s0", "t0")}
    layer2 = {("s1", "t1")}
    psa = LayeredPSA(layer1=layer1, layer2=layer2, w1=0.7, w2=0.3)
    data = MockData()

    prec, rec, f1 = approximate_evaluate(alignment, sim, data, psa)
    check("approx_eval f1>0", f1 > 0)
    check("approx_eval rec>0", rec > 0)

    # 空对齐
    empty_align = np.zeros((3, 3), dtype=np.float32)
    prec, rec, f1 = approximate_evaluate(empty_align, sim, data, psa)
    check("approx_eval empty align", f1 == 0)


# ============================================================
# 5. augmented_psa.py：字符串相似度
# ============================================================

def test_string_similarity():
    print("\n=== 5. 字符串相似度 ===")
    from utils.augmented_psa import _smoa_similarity, _ngram_similarity, _string_match

    # SMOA
    check("smoa identical",  approx_eq(_smoa_similarity("heart", "heart"), 1.0))
    check("smoa empty",      approx_eq(_smoa_similarity("", "heart"), 0.0))
    check("smoa partial",    _smoa_similarity("heart muscle", "heart") > 0.5)
    check("smoa different",  _smoa_similarity("liver", "kidney") < 0.5)

    # N-gram
    check("ngram identical", approx_eq(_ngram_similarity("heart", "heart"), 1.0))
    check("ngram empty s1",  _ngram_similarity("a", "heart") < 0.5)
    check("ngram partial",   _ngram_similarity("heart muscle", "heart") > 0.3)

    # string_match OR关系
    check("match thresh=1.0 exact",    _string_match("heart", "heart", 1.0))
    check("match thresh=1.0 diff",     not _string_match("liver", "kidney", 1.0))
    check("match thresh=0.5 partial",  _string_match("heart muscle", "heart muscle", 0.5))


# ============================================================
# 6. augmented_psa.py：多模型交集
# ============================================================

def test_intersection_anchors():
    print("\n=== 6. 多模型交集 ===")
    from utils.augmented_psa import compute_intersection_anchors

    # 两个模型，取交集
    sim1 = np.array([[0.9, 0.2], [0.3, 0.8]], dtype=np.float32)
    sim2 = np.array([[0.8, 0.9], [0.1, 0.7]], dtype=np.float32)
    mats = {"m1": sim1, "m2": sim2}

    anchors = compute_intersection_anchors(mats, k=0.7)
    # k=0.7：m1中>=0.7的：(0,0),(1,1)；m2中>=0.7的：(0,1),(1,1)
    # 交集：只有(1,1)
    check("intersection anchors", anchors == {(1, 1)})

    # k很低，所有对都入选
    anchors_low = compute_intersection_anchors(mats, k=0.0)
    check("intersection k=0", len(anchors_low) == 4)


# ============================================================
# 7. selection.py：各指标计算
# ============================================================

def test_selection_metrics():
    print("\n=== 7. 选择算子指标 ===")
    from gp.selection import (compute_diversity, compute_reliability,
                               compute_complexity, cosine_annealing_weights)
    from gp.individual import Individual, CombinationNode, FilterNode
    from utils.data_loader import LLM_TYPE_MAP

    # 构造简单个体
    def make_individual(models, filter_method="nde"):
        if len(models) == 1:
            comb = CombinationNode(model_name=models[0])
        else:
            comb = CombinationNode(
                op_name="avg",
                left=CombinationNode(model_name=models[0]),
                right=CombinationNode(model_name=models[1]),
            )
        filt = FilterNode(filter_method=filter_method, filter_params={})
        return Individual(comb, filt, LLM_TYPE_MAP.copy())

    # diversity：单一类型=0，三类均匀=1
    ind_single = make_individual(["sapbert"])  # string类
    check("diversity single type", approx_eq(compute_diversity(ind_single), 0.0))

    # 构造三类各一个的个体
    comb3 = CombinationNode(
        op_name="avg",
        left=CombinationNode(
            op_name="avg",
            left=CombinationNode(model_name="sapbert"),    # string
            right=CombinationNode(model_name="biogpt"),    # structure
        ),
        right=CombinationNode(model_name="umlsbert"),      # semantic
    )
    filt3 = FilterNode(filter_method="nde", filter_params={})
    ind3 = Individual(comb3, filt3, LLM_TYPE_MAP.copy())
    check("diversity three types", approx_eq(compute_diversity(ind3), 1.0))

    # complexity
    ind2 = make_individual(["sapbert", "biogpt"])
    # comb=3节点(avg+2叶), filter=1节点
    check("complexity", compute_complexity(ind2) == 4)

    # cosine_annealing_weights
    w_div0, w_rel0 = cosine_annealing_weights(0, 10)
    w_divN, w_relN = cosine_annealing_weights(9, 10)
    check("cosine w_div early≈1", w_div0 > 0.9)
    check("cosine w_rel late≈1",  w_relN > 0.9)
    check("cosine sum=1 early",   approx_eq(w_div0 + w_rel0, 1.0))
    check("cosine sum=1 late",    approx_eq(w_divN + w_relN, 1.0))

    # compute_reliability with mock data
    from utils.data_loader import HierarchyData
    src_hier = HierarchyData(
        uri_to_idx={}, idx_to_uri={},
        parents={1: [0], 2: [0]},   # 0是1和2的父类
        children={0: [1, 2]},
        part_of_parents={}, part_of_children={},
    )
    tgt_hier = HierarchyData(
        uri_to_idx={}, idx_to_uri={},
        parents={10: [9], 20: [9]},  # 9是10和20的父类
        children={9: [10, 20]},
        part_of_parents={}, part_of_children={},
    )

    class MockData2:
        src_hierarchy = src_hier
        tgt_hierarchy = tgt_hier

    # 构造一致的对齐：0→9, 1→10（0是1的父，9是10的父，一致）
    ind_rel = make_individual(["sapbert"])
    ind_rel._alignment_cache = np.zeros((5, 25), dtype=np.float32)
    ind_rel._alignment_cache[0, 9]  = 1.0
    ind_rel._alignment_cache[1, 10] = 1.0

    r = compute_reliability(ind_rel, MockData2())
    check("reliability consistent", approx_eq(r, 1.0),
          f"expected 1.0, got {r}")

    # 构造冲突的对齐：0→9, 1→20（0是1的父，但20不是10的父，冲突）
    ind_conflict = make_individual(["sapbert"])
    ind_conflict._alignment_cache = np.zeros((5, 25), dtype=np.float32)
    ind_conflict._alignment_cache[0, 10] = 1.0  # 0→10
    ind_conflict._alignment_cache[1, 20] = 1.0  # 1→20，0是1的父但10不是20的父，冲突

    r2 = compute_reliability(ind_conflict, MockData2())
    check("reliability conflict", approx_eq(r2, 0.0),
          f"expected 0.0, got {r2}")


# ============================================================
# 8. fitness.py：tree_hash
# ============================================================

def test_tree_hash():
    print("\n=== 8. tree_hash ===")
    from gp.fitness import tree_hash
    from gp.individual import Individual, CombinationNode, FilterNode
    from utils.data_loader import LLM_TYPE_MAP

    def make_ind(model, fmethod):
        return Individual(
            CombinationNode(model_name=model),
            FilterNode(filter_method=fmethod, filter_params={}),
            LLM_TYPE_MAP.copy()
        )

    ind_a = make_ind("sapbert", "nde")
    ind_b = make_ind("sapbert", "nde")
    ind_c = make_ind("biogpt",  "nde")
    ind_d = make_ind("sapbert", "hungarian")

    check("hash same structure equal",    tree_hash(ind_a) == tree_hash(ind_b))
    check("hash diff model not equal",    tree_hash(ind_a) != tree_hash(ind_c))
    check("hash diff filter not equal",   tree_hash(ind_a) != tree_hash(ind_d))


# ============================================================
# 9. data_loader.py：HierarchyData关系查询
# ============================================================

def test_hierarchy_data():
    print("\n=== 9. HierarchyData关系查询 ===")
    from utils.data_loader import HierarchyData

    hier = HierarchyData(
        uri_to_idx={}, idx_to_uri={},
        parents={1: [0], 2: [0], 3: [1]},
        children={0: [1, 2], 1: [3]},
        part_of_parents={4: [0]},
        part_of_children={0: [4]},
    )

    # is-a关系
    check("isa parent",      hier.has_isa_relation(1, 0))   # 0是1的父
    check("isa child",       hier.has_isa_relation(0, 1))   # 1是0的子
    check("isa no relation", not hier.has_isa_relation(1, 2))  # 兄弟，无关系
    check("isa no relation transitive", not hier.has_isa_relation(3, 0))  # 祖孙，非直接

    # part-of关系
    check("partof parent",   hier.has_partof_relation(4, 0))
    check("partof child",    hier.has_partof_relation(0, 4))
    check("partof none",     not hier.has_partof_relation(1, 4))

    # any关系
    check("any isa",         hier.has_any_relation(1, 0))
    check("any partof",      hier.has_any_relation(4, 0))
    check("any none",        not hier.has_any_relation(1, 4))


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("GP本体匹配系统 - 单元测试")
    print("=" * 60)

    tests = [
        test_arithmetic_ops,
        test_logical_ops,
        test_filter_ops,
        test_evaluator,
        test_string_similarity,
        test_intersection_anchors,
        test_selection_metrics,
        test_tree_hash,
        test_hierarchy_data,
    ]

    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  [ERROR] {test.__name__}: {e}")
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"结果：{PASS} 通过，{FAIL} 失败")
    print("=" * 60)
    sys.exit(0 if FAIL == 0 else 1)
