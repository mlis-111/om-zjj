import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

"""
02_compute_embeddings.py
用法:
  python 02_compute_embeddings.py \
    --entities data/parsed/mouse.json \
    --model biobert \
    --out embeddings/anatomy/biobert_src_emb.npy \
    --gpu 0

模型分类：
  字符串类：biobert / pubmedbert / sapbert / bioelectra / scibert / clinicalbert
  结构类：  biogpt / biomistral
  语义类：  distilbert / bluebert / tinybert / biolinkbert / umlsbert
"""
import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# 模型名称映射
MODEL_MAP = {
    # 字符串类
    "biobert":      "dmis-lab/biobert-v1.1",
    "pubmedbert":   "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "sapbert":      "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    "bioelectra":   "kamalkraj/bioelectra-base-discriminator-pubmed",
    "scibert":      "allenai/scibert_scivocab_uncased",
    "clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
    # 结构类
    "biogpt":       "microsoft/biogpt",
    "biomistral":   "BioMistral/BioMistral-7B",
    # 语义类
    "distilbert":   "distilbert-base-uncased",
    "bluebert":     "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",
    "tinybert":     "huawei-noah/TinyBERT_General_4L_312D",
    "biolinkbert":  "michiyasunaga/BioLinkBERT-base",
    "umlsbert":     "GanjinZero/UMLSBert_ENG",
}

# 各模型embedding维度（用于验证）
DIM_MAP = {
    "biobert":      768,
    "pubmedbert":   768,
    "sapbert":      768,
    "bioelectra":   768,
    "scibert":      768,
    "clinicalbert": 768,
    "biogpt":       1024,
    "biomistral":   4096,
    "distilbert":   768,
    "bluebert":     768,
    "tinybert":     312,
    "biolinkbert":  768,
    "umlsbert":     768,
}

# 生成式模型列表（取最后token）
CAUSAL_MODELS = {"biogpt", "biomistral"}

# 需要4bit量化的大模型（7B+）
QUANTIZE_MODELS = {"biomistral"}


def load_model(model_name: str, gpu_id: int):
    """
    加载模型和tokenizer。
    - BERT类编码器：AutoModel，取[CLS]
    - 生成式小模型（biogpt）：AutoModelForCausalLM，取最后token
    - 生成式大模型（biomistral）：AutoModelForCausalLM + 4bit量化
    """
    model_path = MODEL_MAP[model_name]
    device = f"cuda:{gpu_id}"

    if model_name in QUANTIZE_MODELS:
        # 4bit量化，显存约8GB
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map={"": gpu_id},
        )
    elif model_name in CAUSAL_MODELS:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        model.eval()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path).to(device)
        model.eval()

    return tokenizer, model, device


def get_embeddings_bert(texts: list, tokenizer, model, device,
                        batch_size: int = 64, max_length: int = 128) -> np.ndarray:
    """
    BERT类模型：取[CLS] token的hidden state作为sentence embedding。
    适用于：biobert, pubmedbert, sapbert, bioelectra, scibert, clinicalbert,
            distilbert, bluebert, tinybert, biolinkbert, umlsbert
    """
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"BERT encoding"):
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
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [batch, dim]
        all_embs.append(cls_emb.cpu().float().numpy())
    return np.vstack(all_embs)


def get_embeddings_causal(texts: list, tokenizer, model, device,
                          batch_size: int = 16, max_length: int = 128) -> np.ndarray:
    """
    生成式模型：取最后一个有效token的hidden state作为sentence embedding。
    适用于：biogpt, biomistral
    """
    all_embs = []
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for i in tqdm(range(0, len(texts), batch_size), desc=f"Causal LM encoding"):
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
                output_hidden_states=True,
            )
        hidden = outputs.hidden_states[-1]                          # [batch, seq_len, dim]
        lengths = inputs["attention_mask"].sum(dim=1) - 1           # [batch]
        last_emb = hidden[torch.arange(len(batch)), lengths, :]     # [batch, dim]
        all_embs.append(last_emb.cpu().float().numpy())
    return np.vstack(all_embs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", required=True, help="实体JSON文件路径")
    parser.add_argument("--model", required=True, choices=list(MODEL_MAP.keys()))
    parser.add_argument("--out",     required=True, help="输出npy路径")
    parser.add_argument("--gpu",     type=int, default=0)
    args = parser.parse_args()

    with open(args.entities, encoding="utf-8") as f:
        entities = json.load(f)
    texts = [e["text"] for e in entities]
    print(f"实体数量: {len(texts)}，使用GPU: {args.gpu}，模型: {args.model}")

    tokenizer, model, device = load_model(args.model, args.gpu)

    if args.model in CAUSAL_MODELS:
        embs = get_embeddings_causal(texts, tokenizer, model, device)
    else:
        embs = get_embeddings_bert(texts, tokenizer, model, device)

    expected_dim = DIM_MAP[args.model]
    assert embs.shape == (len(texts), expected_dim), \
        f"维度异常: {embs.shape}，期望 ({len(texts)}, {expected_dim})"

    np.save(args.out, embs)
    print(f"Embedding已保存: {args.out}，shape={embs.shape}")