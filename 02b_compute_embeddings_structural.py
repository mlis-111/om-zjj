import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

"""
02b_compute_embeddings_structural.py
在 02_compute_embeddings.py 基础上，补充四个结构类模型：
  - biogpt_large   microsoft/BioGPT-Large         ~1.5GB  hidden=1536
  - biomistral     BioMistral/BioMistral-7B        ~14GB   hidden=4096  4bit量化
  - meditron       epfl-llm/meditron-7b            ~14GB   hidden=4096  4bit量化
  - medalpaca      medalpaca/medalpaca-7b           ~14GB   hidden=4096  4bit量化

用法:
  python 02b_compute_embeddings_structural.py \
    --entities data/parsed/mouse.json \
    --model biogpt_large \
    --out embeddings/anatomy/biogpt_large_src_emb.npy \
    --gpu 0
"""

import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# 模型名称映射
MODEL_MAP = {
    "biogpt_large": "microsoft/BioGPT-Large",
    "biomistral":   "BioMistral/BioMistral-7B",
    "meditron":     "epfl-llm/meditron-7b",
    "medalpaca":    "medalpaca/medalpaca-7b",
}

# embedding维度
DIM_MAP = {
    "biogpt_large": 1536,
    "biomistral":   4096,
    "meditron":     4096,
    "medalpaca":    4096,
}

# 需要4bit量化的大模型
QUANTIZE_MODELS = {"biomistral", "meditron", "medalpaca"}


def load_model(model_name: str, gpu_id: int):
    """
    加载模型和tokenizer。
    biogpt_large：直接加载，不需要量化（1.5B参数，显存约6GB）
    biomistral / meditron / medalpaca：4bit量化（7B参数，显存约8GB）
    """
    model_path = MODEL_MAP[model_name]
    device = f"cuda:{gpu_id}"

    if model_name in QUANTIZE_MODELS:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map={"": gpu_id},
        )
    else:
        # biogpt_large：直接加载到GPU
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        model.eval()

    return tokenizer, model, device


def get_embeddings_causal(texts: list, tokenizer, model, device,
                          batch_size: int = 16, max_length: int = 128) -> np.ndarray:
    """
    生成式模型：取最后一个有效token的hidden state作为sentence embedding。
    所有结构类模型（BioGPT-Large / BioMistral / Meditron / MedAlpaca）均使用此方法。
    """
    all_embs = []
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for i in tqdm(range(0, len(texts), batch_size), desc=f"encoding"):
        batch = texts[i: i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
            )
        # 取最后一层hidden states的最后一个有效token
        hidden = outputs.hidden_states[-1]                          # [batch, seq_len, dim]
        lengths = inputs["attention_mask"].sum(dim=1) - 1           # [batch]
        last_emb = hidden[torch.arange(len(batch)), lengths, :]     # [batch, dim]
        all_embs.append(last_emb.cpu().float().numpy())

    return np.vstack(all_embs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", required=True, help="实体JSON文件路径")
    parser.add_argument("--model",    required=True, choices=list(MODEL_MAP.keys()))
    parser.add_argument("--out",      required=True, help="输出npy路径")
    parser.add_argument("--gpu",      type=int, default=0)
    args = parser.parse_args()

    with open(args.entities, encoding="utf-8") as f:
        entities = json.load(f)
    texts = [e["text"] for e in entities]
    print(f"实体数量: {len(texts)}，使用GPU: {args.gpu}，模型: {args.model}")

    tokenizer, model, device = load_model(args.model, args.gpu)
    embs = get_embeddings_causal(texts, tokenizer, model, device)

    expected_dim = DIM_MAP[args.model]
    assert embs.shape == (len(texts), expected_dim), \
        f"维度异常: {embs.shape}，期望 ({len(texts)}, {expected_dim})"

    np.save(args.out, embs)
    print(f"Embedding已保存: {args.out}，shape={embs.shape}")
