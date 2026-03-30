# 相似度矩阵预计算实现方案

## 环境要求

```bash
pip install transformers  owlready2 rdflib numpy tqdm accelerate bitsandbytes
pip torch
```

---

## 目录结构

```
├── data/
│   └── oaei/
│       ├── anatomy/
│       │   ├── mouse.owl
│       │   └── human.owl
│       └── conference/
│           ├── cmt.owl
│           ├── conference.owl
│           └── ...（其余14个owl文件）
├── embeddings/
│   ├── anatomy/
│   │   ├── biobert_sim.npy
│   │   ├── biogpt_sim.npy
│   │   ├── biomistral_sim.npy
│   │   ├── biobert_src_emb.npy
│   │   ├── biobert_tgt_emb.npy
│   │   └── entity_index.json
│   └── conference/
│       └── cmt_conference/
│           └── ...（同上）
├── 01_parse_ontology.py
├── 02_compute_embeddings.py
├── 03_compute_similarity.py
└── run_all.sh
```

---

## Step 1：本体解析 `01_parse_ontology.py`

**目标：** 从OWL文件中提取所有Class实体及其文本字段，保存为JSON。

```python
"""
01_parse_ontology.py
用法: python 01_parse_ontology.py --owl data/oaei/anatomy/mouse.owl --out data/parsed/mouse.json
"""
import argparse
import json
from rdflib import Graph, RDF, OWL, RDFS, URIRef, Namespace

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
```

**运行：**
```bash
python 01_parse_ontology.py --owl data/oaei/anatomy/mouse.owl --out data/parsed/mouse.json
python 01_parse_ontology.py --owl data/oaei/anatomy/human.owl --out data/parsed/human.json
```

**解析完成后务必检查：**
- 打印几条实体看label和synonyms是否正常提取
- 确认实体数量接近论文报告的数值（AMA≈2744，NCI≈3304）
- 检查是否存在label为纯数字ID的实体（说明label字段缺失，需要调整提取逻辑）

---

## Step 2：计算Embedding `02_compute_embeddings.py`

**目标：** 对每个实体的text字段用指定LLM编码，输出[N, dim]的embedding矩阵，保存为npy。

```python
"""
02_compute_embeddings.py
用法:
  python 02_compute_embeddings.py \
    --entities data/parsed/mouse.json \
    --model biobert \
    --out embeddings/anatomy/biobert_src_emb.npy \
    --gpu 0
"""
import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModel,           # BioBERT
    AutoModelForCausalLM,               # BioGPT, BioMistral
    BitsAndBytesConfig
)

# 模型名称映射（可换成本地路径）
MODEL_MAP = {
    "biobert":    "dmis-lab/biobert-v1.1",
    "biogpt":     "microsoft/biogpt",
    "biomistral": "BioMistral/BioMistral-7B",
}

# 各模型embedding维度（用于验证）
DIM_MAP = {
    "biobert":    768,
    "biogpt":     1024,
    "biomistral": 4096,
}


def load_model(model_name: str, gpu_id: int):
    """
    加载模型和tokenizer。
    BioMistral-7B使用4bit量化降低显存占用（约8GB）。
    """
    model_path = MODEL_MAP[model_name]
    device = f"cuda:{gpu_id}"

    if model_name == "biomistral":
        # 4bit量化，单卡8GB以内
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map={"": gpu_id},
        )
    elif model_name == "biogpt":
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        model.eval()
    else:
        # BioBERT：标准BERT加载
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path).to(device)
        model.eval()

    return tokenizer, model, device


def get_embeddings_bert(texts: list[str], tokenizer, model, device,
                        batch_size: int = 64, max_length: int = 128) -> np.ndarray:
    """
    BioBERT：取[CLS] token的hidden state作为sentence embedding。
    """
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="BioBERT encoding"):
        batch = texts[i: i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # [CLS] token embedding
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [batch, 768]
        all_embs.append(cls_emb.cpu().float().numpy())
    return np.vstack(all_embs)  # [N, 768]


def get_embeddings_causal(texts: list[str], tokenizer, model, device,
                          batch_size: int = 16, max_length: int = 128) -> np.ndarray:
    """
    BioGPT / BioMistral：取最后一个token的hidden state作为sentence embedding。
    生成式模型没有[CLS]，用最后位置的hidden state近似sentence representation。
    """
    all_embs = []
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for i in tqdm(range(0, len(texts), batch_size), desc="Causal LM encoding"):
        batch = texts[i: i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True
            )
        # 取最后一层hidden states，选每个序列实际最后一个token
        hidden = outputs.hidden_states[-1]   # [batch, seq_len, dim]
        # attention_mask找每条序列最后一个有效位置
        lengths = inputs["attention_mask"].sum(dim=1) - 1  # [batch]
        last_emb = hidden[torch.arange(len(batch)), lengths, :]  # [batch, dim]
        all_embs.append(last_emb.cpu().float().numpy())
    return np.vstack(all_embs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", required=True, help="实体JSON文件路径")
    parser.add_argument("--model", required=True, choices=["biobert", "biogpt", "biomistral"])
    parser.add_argument("--out", required=True, help="输出npy路径")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    # 读取实体文本
    with open(args.entities, encoding="utf-8") as f:
        entities = json.load(f)
    texts = [e["text"] for e in entities]
    print(f"实体数量: {len(texts)}，使用GPU: {args.gpu}，模型: {args.model}")

    # 加载模型
    tokenizer, model, device = load_model(args.model, args.gpu)

    # 计算embedding
    if args.model == "biobert":
        embs = get_embeddings_bert(texts, tokenizer, model, device)
    else:
        embs = get_embeddings_causal(texts, tokenizer, model, device)

    # 验证维度
    expected_dim = DIM_MAP[args.model]
    assert embs.shape == (len(texts), expected_dim), \
        f"维度异常: {embs.shape}，期望 ({len(texts)}, {expected_dim})"

    # 保存
    np.save(args.out, embs)
    print(f"Embedding已保存: {args.out}，shape={embs.shape}")
```

---

## Step 3：计算相似度矩阵 `03_compute_similarity.py`

**目标：** 给定源/目标embedding矩阵，用矩阵乘法批量计算余弦相似度，输出[M, N]矩阵和entity_index.json。

```python
"""
03_compute_similarity.py
用法:
  python 03_compute_similarity.py \
    --src_emb embeddings/anatomy/biobert_src_emb.npy \
    --tgt_emb embeddings/anatomy/biobert_tgt_emb.npy \
    --src_entities data/parsed/mouse.json \
    --tgt_entities data/parsed/human.json \
    --out embeddings/anatomy/biobert_sim.npy \
    --index embeddings/anatomy/entity_index.json
"""
import argparse
import json
import numpy as np


def cosine_sim_matrix(src: np.ndarray, tgt: np.ndarray,
                      chunk_size: int = 512) -> np.ndarray:
    """
    计算余弦相似度矩阵。
    先L2归一化，再分块矩阵乘法，避免[M*N]一次性占用过多内存。
    src: [M, dim]
    tgt: [N, dim]
    return: [M, N]，值域[0,1]（余弦相似度经过归一化后非负）
    """
    # L2归一化
    src_norm = src / (np.linalg.norm(src, axis=1, keepdims=True) + 1e-9)
    tgt_norm = tgt / (np.linalg.norm(tgt, axis=1, keepdims=True) + 1e-9)

    M = src_norm.shape[0]
    N = tgt_norm.shape[0]
    sim = np.zeros((M, N), dtype=np.float32)

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        # [chunk, dim] @ [dim, N] -> [chunk, N]
        sim[start:end] = src_norm[start:end] @ tgt_norm.T

    # 余弦相似度可能因浮点误差略超[-1,1]，裁剪到[0,1]
    sim = np.clip(sim, 0.0, 1.0)
    return sim


def build_entity_index(src_entities: list, tgt_entities: list) -> dict:
    """
    构建URI到矩阵行/列索引的映射，保存为JSON。
    格式：
    {
        "src": {"http://...uri1": 0, "http://...uri2": 1, ...},
        "tgt": {"http://...uri1": 0, ...}
    }
    """
    return {
        "src": {e["uri"]: i for i, e in enumerate(src_entities)},
        "tgt": {e["uri"]: i for i, e in enumerate(tgt_entities)},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_emb",      required=True)
    parser.add_argument("--tgt_emb",      required=True)
    parser.add_argument("--src_entities", required=True)
    parser.add_argument("--tgt_entities", required=True)
    parser.add_argument("--out",          required=True, help="相似度矩阵npy输出路径")
    parser.add_argument("--index",        required=True, help="entity_index.json输出路径")
    args = parser.parse_args()

    src_emb = np.load(args.src_emb)
    tgt_emb = np.load(args.tgt_emb)
    print(f"src embedding shape: {src_emb.shape}")
    print(f"tgt embedding shape: {tgt_emb.shape}")

    sim = cosine_sim_matrix(src_emb, tgt_emb)
    print(f"相似度矩阵 shape: {sim.shape}")
    print(f"统计信息 -> min: {sim.min():.4f}, max: {sim.max():.4f}, "
          f"mean: {sim.mean():.4f}, std: {sim.std():.4f}")

    # 分布检查：如果mean>0.9或mean<0.05，说明embedding提取有问题
    if sim.mean() > 0.9:
        print("[WARNING] 相似度均值过高（>0.9），检查embedding提取方式是否正确")
    if sim.mean() < 0.05:
        print("[WARNING] 相似度均值过低（<0.05），检查文本拼接或模型加载是否正常")

    np.save(args.out, sim)
    print(f"相似度矩阵已保存: {args.out}")

    # 保存entity index
    with open(args.src_entities, encoding="utf-8") as f:
        src_entities = json.load(f)
    with open(args.tgt_entities, encoding="utf-8") as f:
        tgt_entities = json.load(f)

    index = build_entity_index(src_entities, tgt_entities)
    with open(args.index, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    print(f"Entity index已保存: {args.index}")
```

---

## Step 4：一键运行脚本 `run_all.sh`

8张4090分配任务：GPU 0跑BioBERT源，GPU 1跑BioBERT目标，GPU 2跑BioGPT源，以此类推，并行加速。

```bash
#!/bin/bash
set -e

PARSED=data/parsed
EMB=embeddings/anatomy

mkdir -p $EMB

echo "=== Step 1: 解析本体 ==="
python 01_parse_ontology.py --owl data/oaei/anatomy/mouse.owl --out $PARSED/mouse.json
python 01_parse_ontology.py --owl data/oaei/anatomy/human.owl  --out $PARSED/human.json

echo "=== Step 2: 计算Embedding（并行，分配不同GPU）==="

# BioBERT（GPU 0, 1）
python 02_compute_embeddings.py --entities $PARSED/mouse.json --model biobert \
    --out $EMB/biobert_src_emb.npy --gpu 0 &

python 02_compute_embeddings.py --entities $PARSED/human.json --model biobert \
    --out $EMB/biobert_tgt_emb.npy --gpu 1 &

# BioGPT（GPU 2, 3）
python 02_compute_embeddings.py --entities $PARSED/mouse.json --model biogpt \
    --out $EMB/biogpt_src_emb.npy --gpu 2 &

python 02_compute_embeddings.py --entities $PARSED/human.json --model biogpt \
    --out $EMB/biogpt_tgt_emb.npy --gpu 3 &

# BioMistral-7B（GPU 4, 5）—— 4bit量化，单卡约8GB
python 02_compute_embeddings.py --entities $PARSED/mouse.json --model biomistral \
    --out $EMB/biomistral_src_emb.npy --gpu 4 &

python 02_compute_embeddings.py --entities $PARSED/human.json --model biomistral \
    --out $EMB/biomistral_tgt_emb.npy --gpu 5 &

wait
echo "所有Embedding计算完成"

echo "=== Step 3: 计算相似度矩阵 ==="

python 03_compute_similarity.py \
    --src_emb $EMB/biobert_src_emb.npy \
    --tgt_emb $EMB/biobert_tgt_emb.npy \
    --src_entities $PARSED/mouse.json \
    --tgt_entities $PARSED/human.json \
    --out $EMB/biobert_sim.npy \
    --index $EMB/entity_index.json

python 03_compute_similarity.py \
    --src_emb $EMB/biogpt_src_emb.npy \
    --tgt_emb $EMB/biogpt_tgt_emb.npy \
    --src_entities $PARSED/mouse.json \
    --tgt_entities $PARSED/human.json \
    --out $EMB/biogpt_sim.npy \
    --index $EMB/entity_index.json   # index已存在，覆盖写无影响

python 03_compute_similarity.py \
    --src_emb $EMB/biomistral_src_emb.npy \
    --tgt_emb $EMB/biomistral_tgt_emb.npy \
    --src_entities $PARSED/mouse.json \
    --tgt_entities $PARSED/human.json \
    --out $EMB/biomistral_sim.npy \
    --index $EMB/entity_index.json

echo "=== 全部完成 ==="
echo "输出文件："
ls -lh $EMB/
```

---

## 结果验证清单

运行完成后依次确认以下几项，有任何一项异常就需要排查，不要继续往下做实验：

| 检查项 | 正常范围 | 说明 |
|--------|----------|------|
| 实体数量（Anatomy） | src≈2744，tgt≈3304 | 与论文一致 |
| 相似度矩阵shape | [2744, 3304] | 行=源，列=目标 |
| 相似度均值 | 0.1 ~ 0.7 | 太高说明embedding退化，太低说明模型未正确加载 |
| 相似度最大值 | ≈1.0 | 至少有几对完全匹配的实体 |
| entity_index.json | URI可正常查到索引 | 随机抽几个URI验证 |
| 三个矩阵shape一致 | 完全相同 | 三个LLM的矩阵必须对应同一套实体顺序 |

---

## 后续GP实验的加载方式

预计算完成后，GP实验启动时只需：

```python
import numpy as np
import json

# 加载三个相似度矩阵（叶节点）
sim_matrices = {
    "biobert":    np.load("embeddings/anatomy/biobert_sim.npy"),
    "biogpt":     np.load("embeddings/anatomy/biogpt_sim.npy"),
    "biomistral": np.load("embeddings/anatomy/biomistral_sim.npy"),
}

# 加载实体索引（用于最终还原对齐结果）
with open("embeddings/anatomy/entity_index.json") as f:
    entity_index = json.load(f)

# GP个体的叶节点直接引用 sim_matrices[llm_name]，无需再调用LLM
```

这样整个GP进化过程完全不涉及LLM推理，单次适应度评估只是矩阵算术操作，速度可以做到毫秒级。