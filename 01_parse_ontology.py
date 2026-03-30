"""
01_parse_ontology.py
用法: python 01_parse_ontology.py --owl data/oaei/anatomy/mouse.owl --out data/parsed/mouse.json
"""
import argparse
import json
from rdflib import Graph, RDF, OWL, RDFS, URIRef

# 常用同义词属性的URI（OBO本体常见）
SYNONYM_PROPS = [
    "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym",
    "http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym",
    "http://www.geneontology.org/formats/oboInOwl#hasBroadSynonym",
    "http://www.geneontology.org/formats/oboInOwl#hasNarrowSynonym",
]
DEFINITION_PROPS = [
    "http://purl.obolibrary.org/obo/IAO_0000115",  # obo definition
    str(RDFS.comment),
]


def parse_owl(owl_path: str) -> list[dict]:
    """
    解析OWL文件，返回实体列表。
    每个实体格式：
    {
        "uri": "http://...",
        "label": "Heart",
        "synonyms": ["Cardiac", ...],
        "definition": "...",
        "text": "Heart Cardiac ... definition..."   # 最终送入LLM的拼接文本
    }
    """
    g = Graph()
    g.parse(owl_path)

    entities = []
    for cls in g.subjects(RDF.type, OWL.Class):
        if not isinstance(cls, URIRef):
            continue  # 跳过匿名类

        uri = str(cls)

        # 提取label（取第一个英文label）
        labels = [
            str(o) for s, p, o in g.triples((cls, RDFS.label, None))
            if not hasattr(o, 'language') or o.language in (None, 'en')
        ]
        label = labels[0] if labels else uri.split("/")[-1].split("#")[-1]

        # 提取同义词
        synonyms = []
        for prop_uri in SYNONYM_PROPS:
            for _, _, o in g.triples((cls, URIRef(prop_uri), None)):
                synonyms.append(str(o))

        # 提取定义/注释
        definition = ""
        for prop_uri in DEFINITION_PROPS:
            defs = [str(o) for _, _, o in g.triples((cls, URIRef(prop_uri), None))]
            if defs:
                definition = defs[0]
                break

        # 拼接文本：label放最前，权重最高
        parts = [label] + synonyms
        if definition:
            parts.append(definition)
        text = " [SEP] ".join(parts)

        entities.append({
            "uri": uri,
            "label": label,
            "synonyms": synonyms,
            "definition": definition,
            "text": text
        })

    return entities


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--owl", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    entities = parse_owl(args.owl)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)
    print(f"解析完成，共 {len(entities)} 个实体 -> {args.out}")
