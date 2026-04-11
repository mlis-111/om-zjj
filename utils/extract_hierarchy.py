"""
extract_hierarchy.py
提取本体的层次关系（is-a 和 part-of），输出为JSON文件。

用法:
  python extract_hierarchy.py \
    --owl data/oaei/anatomy/mouse.owl \
    --out data/parsed/mouse_hierarchy.json

说明：
- is-a 关系：直接来自 rdfs:subClassOf
- part-of 关系：来自 OWL限制结构
    <owl:Restriction>
        <owl:onProperty rdf:resource="...part_of"/>
        <owl:someValuesFrom rdf:resource="...TargetClass"/>
    </owl:Restriction>
  即 subClassOf some(part_of, TargetClass) 这种表达式
- OAEI Anatomy数据集的part-of属性URI：http://mouse.owl#UNDEFINED_part_of
"""
import argparse
import json
from collections import defaultdict
from rdflib import Graph, RDF, OWL, RDFS, URIRef

# part-of 属性URI列表（覆盖多种本体规范）
PART_OF_PROP_URIS = {
    "http://mouse.owl#UNDEFINED_part_of",            # OAEI Anatomy mouse
    "http://human.owl#UNDEFINED_part_of",            # OAEI Anatomy human
    "http://purl.obolibrary.org/obo/BFO_0000050",    # OBO Foundry part_of
    "http://purl.obolibrary.org/obo/RO_0001025",     # located_in
    "http://www.co-ode.org/ontologies/part_of",
    "http://purl.org/obo/owl/OBO_REL#part_of",
    "http://bioonto.de/owl/part_of",
}


def extract_hierarchy(owl_path: str) -> dict:
    """
    从OWL文件提取is-a和part-of层次关系。
    """
    g = Graph()
    g.parse(owl_path)

    # 收集所有Class URI
    uris = sorted([
        str(cls) for cls in g.subjects(RDF.type, OWL.Class)
        if isinstance(cls, URIRef)
    ])
    uri_to_idx = {uri: idx for idx, uri in enumerate(uris)}
    idx_to_uri = {str(idx): uri for idx, uri in enumerate(uris)}

    # ---- is-a 关系（rdfs:subClassOf 直接声明）----
    parents  = defaultdict(list)   # child_idx -> [parent_idx]
    children = defaultdict(list)   # parent_idx -> [child_idx]

    for s, _, o in g.triples((None, RDFS.subClassOf, None)):
        if not isinstance(s, URIRef) or not isinstance(o, URIRef):
            continue
        s_str, o_str = str(s), str(o)
        # 过滤OWL系统类和非本体命名空间
        if s_str not in uri_to_idx or o_str not in uri_to_idx:
            continue
        skip_prefixes = (
            "http://www.w3.org/",
            "http://www.geneontology.org/",
            "http://purl.obolibrary.org/obo/IAO",
        )
        if any(o_str.startswith(p) for p in skip_prefixes):
            continue
        s_idx = uri_to_idx[s_str]
        o_idx = uri_to_idx[o_str]
        if o_idx not in parents[s_idx]:
            parents[s_idx].append(o_idx)
        if s_idx not in children[o_idx]:
            children[o_idx].append(s_idx)

    # ---- part-of 关系（OWL限制结构解析）----
    # 结构：ClassA subClassOf [owl:Restriction onProperty part_of someValuesFrom ClassB]
    # 表示：ClassA part-of some ClassB
    part_of_parents  = defaultdict(list)   # child_idx -> [parent_idx]
    part_of_children = defaultdict(list)   # parent_idx -> [child_idx]

    part_of_uris = {URIRef(u) for u in PART_OF_PROP_URIS}

    # 找所有 owl:Restriction 节点
    for restriction in g.subjects(RDF.type, OWL.Restriction):
        # 检查 onProperty 是否是 part-of
        on_props = list(g.objects(restriction, OWL.onProperty))
        if not any(p in part_of_uris for p in on_props):
            continue

        # 找 someValuesFrom 指向的目标类
        targets = list(g.objects(restriction, OWL.someValuesFrom))
        if not targets:
            continue
        target = targets[0]
        if not isinstance(target, URIRef):
            continue
        target_str = str(target)
        if target_str not in uri_to_idx:
            continue
        tgt_idx = uri_to_idx[target_str]

        # 找使用这个限制的类（subClassOf 这个 restriction）
        for subj in g.subjects(RDFS.subClassOf, restriction):
            if not isinstance(subj, URIRef):
                continue
            subj_str = str(subj)
            if subj_str not in uri_to_idx:
                continue
            subj_idx = uri_to_idx[subj_str]

            # subj part-of target
            if tgt_idx not in part_of_parents[subj_idx]:
                part_of_parents[subj_idx].append(tgt_idx)
            if subj_idx not in part_of_children[tgt_idx]:
                part_of_children[tgt_idx].append(subj_idx)

    # 统计
    isa_edges    = sum(len(v) for v in parents.values())
    partof_edges = sum(len(v) for v in part_of_parents.values())
    print(f"  实体数量: {len(uris)}")
    print(f"  is-a 关系数: {isa_edges}")
    print(f"  part-of 关系数: {partof_edges}")

    return {
        "uri_to_idx":        uri_to_idx,
        "idx_to_uri":        idx_to_uri,
        "parents":           dict(parents),
        "children":          dict(children),
        "part_of_parents":   dict(part_of_parents),
        "part_of_children":  dict(part_of_children),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--owl", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    print(f"解析层次关系: {args.owl}")
    result = extract_hierarchy(args.owl)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)

    print(f"层次关系已保存: {args.out}")