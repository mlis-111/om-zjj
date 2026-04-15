"""
analyze_propagation.py
分析 BFS 传播生成的候选对的有效性。
"""
import json
import os
import sys
import random
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from utils.data_loader import load_om_data
from utils.dataset_config import get_config


def bfs_neighbors_by_type(start_idx, parents, children, max_depth):
    """返回 {neighbor_idx: min_depth}"""
    visited = {start_idx: 0}
    queue   = deque([(start_idx, 0)])
    while queue:
        curr, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for nb in parents.get(curr, []) + children.get(curr, []):
            if nb not in visited:
                visited[nb] = depth + 1
                queue.append((nb, depth + 1))
    del visited[start_idx]
    return visited


# ============================================================
# 分析1A：结构保留率（全量，小数据集用）
# ============================================================

def analyze_structural_preservation(data, max_depth=3):
    print("\n=== 分析1：结构保留率（全量）===")
    _run_preservation(data, list(data.reference), max_depth)


def analyze_structural_preservation_sampled(data, max_depth=3, sample_size=300):
    print(f"\n=== 分析1：结构保留率（采样 {sample_size} 对）===")
    ref = list(data.reference)
    if len(ref) > sample_size:
        ref = random.sample(ref, sample_size)
    _run_preservation(data, ref, max_depth)


def _run_preservation(data, ref_sample, max_depth):
    src_uri_to_idx = data.src_uri_to_idx
    tgt_uri_to_idx = data.tgt_uri_to_idx
    src_hier = data.src_hierarchy
    tgt_hier = data.tgt_hierarchy
    ref_set_full = {(src_uri_to_idx[s], tgt_uri_to_idx[t])
                    for s, t in data.reference
                    if s in src_uri_to_idx and t in tgt_uri_to_idx}

    total_candidates = 0
    total_hits       = 0
    cands_by_type    = defaultdict(int)
    hits_by_type     = defaultdict(int)

    for s_uri, t_uri in ref_sample:
        e1 = src_uri_to_idx.get(s_uri)
        e2 = tgt_uri_to_idx.get(t_uri)
        if e1 is None or e2 is None:
            continue

        n1_isa    = bfs_neighbors_by_type(e1, src_hier.parents,         src_hier.children,         max_depth)
        n1_partof = bfs_neighbors_by_type(e1, src_hier.part_of_parents, src_hier.part_of_children, max_depth)
        n2_isa    = bfs_neighbors_by_type(e2, tgt_hier.parents,         tgt_hier.children,         max_depth)
        n2_partof = bfs_neighbors_by_type(e2, tgt_hier.part_of_parents, tgt_hier.part_of_children, max_depth)

        for type_name, n1, n2 in [
            ("isa×isa",       n1_isa,    n2_isa),
            ("isa×partof",    n1_isa,    n2_partof),
            ("partof×isa",    n1_partof, n2_isa),
            ("partof×partof", n1_partof, n2_partof),
        ]:
            for nb1 in n1:
                for nb2 in n2:
                    cands_by_type[type_name] += 1
                    total_candidates += 1
                    if (nb1, nb2) in ref_set_full:
                        hits_by_type[type_name] += 1
                        total_hits += 1

    prec = total_hits / total_candidates if total_candidates else 0
    print(f"  采样对数: {len(ref_sample)}, 扩展候选总数: {total_candidates}")
    print(f"  命中: {total_hits},  全局 Precision={prec:.4f}")
    print()
    print(f"  {'类型':<16} {'候选数':>10} {'命中数':>8} {'Precision':>10}")
    print(f"  {'-'*48}")
    for t in ["isa×isa", "isa×partof", "partof×isa", "partof×partof"]:
        c = cands_by_type[t]
        h = hits_by_type[t]
        p = h / c if c else 0
        print(f"  {t:<16} {c:>10} {h:>8} {p:>10.4f}")


# ============================================================
# 分析2：按传播深度的 Precision 衰减
# ============================================================

def analyze_by_depth(data, anchors, max_depth=3):
    print("\n=== 分析2：按传播深度的 Precision 衰减 ===")

    src_hier = data.src_hierarchy
    tgt_hier = data.tgt_hierarchy
    src_uri_to_idx = data.src_uri_to_idx
    tgt_uri_to_idx = data.tgt_uri_to_idx
    ref_set = {(src_uri_to_idx[s], tgt_uri_to_idx[t])
               for s, t in data.reference
               if s in src_uri_to_idx and t in tgt_uri_to_idx}

    depth_cands = defaultdict(int)
    depth_hits  = defaultdict(int)

    for e1, e2 in anchors:
        depth_cands[(0, 0)] += 1
        if (e1, e2) in ref_set:
            depth_hits[(0, 0)] += 1

        n1_all = bfs_neighbors_by_type(
            e1,
            {k: v for d in [src_hier.parents, src_hier.part_of_parents] for k, v in d.items()},
            {k: v for d in [src_hier.children, src_hier.part_of_children] for k, v in d.items()},
            max_depth)
        n2_all = bfs_neighbors_by_type(
            e2,
            {k: v for d in [tgt_hier.parents, tgt_hier.part_of_parents] for k, v in d.items()},
            {k: v for d in [tgt_hier.children, tgt_hier.part_of_children] for k, v in d.items()},
            max_depth)

        for nb1, d1 in n1_all.items():
            depth_cands[(d1, 0)] += 1
            if (nb1, e2) in ref_set:
                depth_hits[(d1, 0)] += 1
        for nb2, d2 in n2_all.items():
            depth_cands[(0, d2)] += 1
            if (e1, nb2) in ref_set:
                depth_hits[(0, d2)] += 1
        for nb1, d1 in n1_all.items():
            for nb2, d2 in n2_all.items():
                depth_cands[(d1, d2)] += 1
                if (nb1, nb2) in ref_set:
                    depth_hits[(d1, d2)] += 1

    print(f"  {'src深度':>6} {'tgt深度':>6} {'候选数':>10} {'命中数':>8} {'Precision':>10}")
    print(f"  {'-'*46}")
    for key in sorted(depth_cands.keys()):
        d1, d2 = key
        c = depth_cands[key]
        h = depth_hits[key]
        p = h / c if c else 0
        print(f"  {d1:>6} {d2:>6} {c:>10} {h:>8} {p:>10.4f}")


# ============================================================
# 分析3：锚点 vs 传播候选 Precision 对比
# ============================================================

def analyze_anchor_vs_propagated(data, anchors, max_depth=3):
    print("\n=== 分析3：锚点 vs 传播候选 Precision 对比 ===")

    src_uri_to_idx = data.src_uri_to_idx
    tgt_uri_to_idx = data.tgt_uri_to_idx
    ref_set = {(src_uri_to_idx[s], tgt_uri_to_idx[t])
               for s, t in data.reference
               if s in src_uri_to_idx and t in tgt_uri_to_idx}

    anchor_hits = sum(1 for a in anchors if a in ref_set)
    anchor_prec = anchor_hits / len(anchors) if anchors else 0
    print(f"  锚点数量: {len(anchors)},  命中: {anchor_hits},  Precision={anchor_prec:.4f}")

    src_hier = data.src_hierarchy
    tgt_hier = data.tgt_hierarchy

    propagated = set()
    for e1, e2 in anchors:
        n1 = set(bfs_neighbors_by_type(e1, src_hier.parents, src_hier.children, max_depth)) | \
             set(bfs_neighbors_by_type(e1, src_hier.part_of_parents, src_hier.part_of_children, max_depth))
        n2 = set(bfs_neighbors_by_type(e2, tgt_hier.parents, tgt_hier.children, max_depth)) | \
             set(bfs_neighbors_by_type(e2, tgt_hier.part_of_parents, tgt_hier.part_of_children, max_depth))
        for nb1 in n1: propagated.add((nb1, e2))
        for nb2 in n2: propagated.add((e1, nb2))
        for nb1 in n1:
            for nb2 in n2: propagated.add((nb1, nb2))

    propagated -= anchors
    prop_hits  = sum(1 for p in propagated if p in ref_set)
    prop_prec  = prop_hits / len(propagated) if propagated else 0
    recall_gain = prop_hits / len(ref_set) if ref_set else 0
    ratio = prop_prec / anchor_prec if anchor_prec > 0 else 0
    print(f"  传播新增候选: {len(propagated)},  命中: {prop_hits},  "
          f"Precision={prop_prec:.4f},  召回增益={recall_gain:.4f}")
    print(f"  传播/锚点 Precision 比值: {ratio:.4f}  "
          f"({'有效' if ratio > 0.1 else '无效，建议 depth=0'})")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="omim-ordo")
    parser.add_argument("--k",     type=float, default=0.7)
    parser.add_argument("--depth", type=int,   default=3)
    args = parser.parse_args()

    print(f"\n数据集: {args.dataset}")
    cfg  = get_config(args.dataset)
    data = load_om_data(
        emb_dir=cfg["emb_dir"],
        src_entities=cfg["src_entities"],
        tgt_entities=cfg["tgt_entities"],
        src_hierarchy=cfg["src_hierarchy"],
        tgt_hierarchy=cfg["tgt_hierarchy"],
        reference_path=cfg["reference"],
        ref_format=cfg.get("ref_format"),
    )

    from utils.augmented_psa import compute_intersection_anchors
    anchors = compute_intersection_anchors(data.sim_matrices, args.k)

    # 大数据集（src×tgt > 5000万）分析1自动切换采样
    if data.n_src * data.n_tgt > 50_000_000:
        print(f"  [大数据集] src={data.n_src}×tgt={data.n_tgt}，分析1启用采样(300对)")
        analyze_structural_preservation_sampled(data, max_depth=args.depth, sample_size=300)
    else:
        analyze_structural_preservation(data, max_depth=args.depth)

    analyze_by_depth(data, anchors, max_depth=args.depth)
    analyze_anchor_vs_propagated(data, anchors, max_depth=args.depth)