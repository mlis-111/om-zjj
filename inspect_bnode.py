"""
inspect_bnode.py
深入检查 subClassOf->BNode 里的实际三元组结构，找出 part-of 的真实表达方式。

用法:
  python inspect_bnode.py --owl path/to/fma.body.owl
"""
import argparse
from collections import Counter
from rdflib import Graph, RDF, OWL, RDFS, URIRef, BNode

def inspect(owl_path: str):
    print(f"解析: {owl_path}")
    g = Graph()
    g.parse(owl_path)

    # 收集所有 subClassOf -> BNode
    bnode_pairs = []
    for s, _, bn in g.triples((None, RDFS.subClassOf, None)):
        if isinstance(s, URIRef) and isinstance(bn, BNode):
            bnode_pairs.append((s, bn))
    print(f"subClassOf URI->BNode 总数: {len(bnode_pairs)}")

    # 统计每个 BNode 里都有哪些谓词
    pred_counter = Counter()
    for _, bn in bnode_pairs:
        for p, o in g.predicate_objects(bn):
            pred_counter[str(p)] += 1

    print(f"\n这些 BNode 里出现的谓词（Top 20）:")
    for pred, cnt in pred_counter.most_common(20):
        label = pred.split("/")[-1].split("#")[-1]
        print(f"  {cnt:6d}  {label}  ({pred})")

    # 打印前 10 个 BNode 的完整内容
    print(f"\n前 10 个 BNode 的完整三元组:")
    seen = set()
    count = 0
    for s, bn in bnode_pairs:
        if bn in seen:
            continue
        seen.add(bn)
        print(f"\n  Class: ...{str(s)[-60:]}")
        print(f"  BNode triples:")
        for p, o in g.predicate_objects(bn):
            p_label = str(p).split("/")[-1].split("#")[-1]
            o_str = str(o)[-80:]
            print(f"    {p_label:30s}  {o_str}")
        count += 1
        if count >= 10:
            break

    # 专门找含 part_of / BFO_0000050 字样的谓词
    print(f"\n谓词中含 'part' 字样的:")
    for pred, cnt in pred_counter.items():
        if 'part' in pred.lower() or 'BFO_0000050' in pred:
            print(f"  {cnt:6d}  {pred}")

    # 检查是否有 rdf:type owl:Restriction
    restriction_count = sum(
        1 for _, bn in bnode_pairs
        if (bn, RDF.type, OWL.Restriction) in g
    )
    print(f"\n这些 BNode 中 rdf:type=owl:Restriction 的数量: {restriction_count}")

    # 检查 intersectionOf / unionOf
    inter_count = sum(1 for _, bn in bnode_pairs if list(g.objects(bn, OWL.intersectionOf)))
    union_count  = sum(1 for _, bn in bnode_pairs if list(g.objects(bn, OWL.unionOf)))
    print(f"含 owl:intersectionOf 的: {inter_count}")
    print(f"含 owl:unionOf 的: {union_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--owl", required=True)
    args = parser.parse_args()
    inspect(args.owl)
