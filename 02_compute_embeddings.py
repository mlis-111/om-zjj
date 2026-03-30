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
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", required=True, help="实体JSON文件路径")
    parser.add_argument("--model", required=True, choices=["biobert", "biogpt", "biomistral"])
    parser.add_argument("--out", required=True, help="输出npy路径")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

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
