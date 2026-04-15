"""
diagnose_partof.py
诊断Bio-ML本体中 part-of 关系的实际表达方式，找出当前代码漏掉的关系。

用法:
  python diagnose_partof.py --owl path/to/ontology.owl
"""
import argparse
from collections import defaultdict, Counter
from rdflib import Graph, RDF, OWL, RDFS, URIRef, BNode

BFO_PART_OF = URIRef("http://purl.obolibrary.org/obo/BFO_0000050")

KNOWN_PART_OF_URIS = {
    URIRef("http://mouse.owl#UNDEFINED_part_of"),
    URIRef("http://human.owl#UNDEFINED_part_of"),
    URIRef("http://purl.obolibrary.org/obo/BFO_0000050"),
    URIRef("http://purl.obolibrary.org/obo/RO_0001025"),
    URIRef("http://www.co-ode.org/ontologies/part_of"),
    URIRef("http://purl.org/obo/owl/OBO_REL#part_of"),
    URIRef("http://bioonto.de/owl/part_of"),
}


def diagnose(owl_path: str):
    print(f"\n解析: {owl_path}")
    g = Graph()
    g.parse(owl_path)
    print(f"总三元组数: {len(g)}")

    # 收集所有 Class URI
    all_classes = {
        str(c) for c in g.subjects(RDF.type, OWL.Class)
        if isinstance(c, URIRef)
    }
    print(f"OWL Class 数量: {len(all_classes)}")

    # ===== 方式1: 当前代码使用的 Restriction 结构 =====
    restriction_count = 0
    for restriction in g.subjects(RDF.type, OWL.Restriction):
        on_props = list(g.objects(restriction, OWL.onProperty))
        if any(p in KNOWN_PART_OF_URIS for p in on_props):
            targets = list(g.objects(restriction, OWL.someValuesFrom))
            if targets and isinstance(targets[0], URIRef):
                for subj in g.subjects(RDFS.subClassOf, restriction):
                    if isinstance(subj, URIRef):
                        restriction_count += 1
    print(f"\n[方式1] Restriction(someValuesFrom) 提取到的 part-of 关系: {restriction_count}")

    # ===== 方式2: 直接 triple: ClassA BFO_0000050 ClassB =====
    direct_count = 0
    for s, _, o in g.triples((None, BFO_PART_OF, None)):
        if isinstance(s, URIRef) and isinstance(o, URIRef):
            direct_count += 1
    print(f"[方式2] 直接 triple (ClassA BFO_0000050 ClassB): {direct_count}")

    # ===== 方式3: subClassOf blank node，blank node 含 part-of 直接指向目标 =====
    # 有些 OBO 本体把 part-of 写成: ClassA subClassOf _:bn, _:bn BFO_0000050 ClassB
    bnode_direct_count = 0
    for s, _, bn in g.triples((None, RDFS.subClassOf, None)):
        if isinstance(s, URIRef) and isinstance(bn, BNode):
            # 检查这个 blank node 是否直接含有 part-of triple
            for po_uri in KNOWN_PART_OF_URIS:
                targets = list(g.objects(bn, po_uri))
                for t in targets:
                    if isinstance(t, URIRef):
                        bnode_direct_count += 1
    print(f"[方式3] subClassOf blank_node，blank node 直接含 part-of: {bnode_direct_count}")

    # ===== 方式4: 扫描所有含 part-of URI 的 triple（不限结构）=====
    print(f"\n[方式4] 扫描所有含 part-of 相关 URI 的 triple:")
    for po_uri in KNOWN_PART_OF_URIS:
        triples = list(g.triples((None, po_uri, None)))
        if triples:
            # 分类
            type_counter = Counter()
            for s, _, o in triples:
                s_type = "URIRef" if isinstance(s, URIRef) else "BNode" if isinstance(s, BNode) else "Literal"
                o_type = "URIRef" if isinstance(o, URIRef) else "BNode" if isinstance(o, BNode) else "Literal"
                type_counter[f"s={s_type}, o={o_type}"] += 1
            uri_label = str(po_uri).split("/")[-1].split("#")[-1]
            print(f"  {uri_label} ({len(triples)} triples):")
            for k, v in type_counter.items():
                print(f"    {k}: {v}")
            # 打印前3个样例
            print(f"  样例:")
            for i, (s, _, o) in enumerate(triples[:3]):
                print(f"    {str(s)[-50:]}  ->  {str(o)[-50:]}")

    # ===== 方式5: owl:annotatedProperty = subClassOf，含 part-of 的公理 =====
    axiom_count = 0
    for ax in g.subjects(RDF.type, OWL.Axiom):
        ann_prop = list(g.objects(ax, OWL.annotatedProperty))
        if RDFS.subClassOf not in ann_prop:
            continue
        for po_uri in KNOWN_PART_OF_URIS:
            if list(g.objects(ax, po_uri)):
                axiom_count += 1
                break
    print(f"\n[方式5] owl:Axiom 里含 part-of annotation 的条目: {axiom_count}")

    # ===== 统计 subClassOf 总数用于对比 =====
    isa_direct = sum(1 for s, _, o in g.triples((None, RDFS.subClassOf, None))
                     if isinstance(s, URIRef) and isinstance(o, URIRef))
    isa_bnode = sum(1 for s, _, o in g.triples((None, RDFS.subClassOf, None))
                    if isinstance(s, URIRef) and isinstance(o, BNode))
    print(f"\n[参考] subClassOf 直接URI->URI: {isa_direct}")
    print(f"[参考] subClassOf URI->BNode (通常是 Restriction): {isa_bnode}")

    print("\n" + "="*60)
    total_missed = direct_count + bnode_direct_count
    print(f"当前代码能提取: {restriction_count}")
    print(f"可能漏掉 (方式2+3): {total_missed}")
    if restriction_count + total_missed > 0:
        miss_rate = total_missed / (restriction_count + total_missed) * 100
        print(f"漏掉比例估计: {miss_rate:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--owl", required=True, help="OWL文件路径")
    args = parser.parse_args()
    diagnose(args.owl)
