#!/bin/bash
set -e

# export HF_ENDPOINT=https://hf-mirror.com

PARSED=data/parsed
EMB=embeddings/anatomy

mkdir -p $PARSED
mkdir -p $EMB

echo "=== Step 1: 解析本体 ==="
python preprocess/01_parse_ontology.py --owl data/oaei/anatomy/mouse.owl --out $PARSED/mouse.json
python preprocess/01_parse_ontology.py --owl data/oaei/anatomy/human.owl  --out $PARSED/human.json

echo "=== Step 2: 计算Embedding ==="
# 共19个模型，GPU 0和GPU 1并行
# GPU 0：字符串类6个 + 语义类7个（全是BERT级别，快）
# GPU 1：结构类6个（biogpt/biogpt_large是小模型，4个7B模型串行跑）
# 两组并行，组内串行

# ---- GPU 0：字符串类 + 语义类 ----
(
  # 字符串类
  for model in biobert pubmedbert sapbert bioelectra scibert clinicalbert; do
    echo "[GPU 0] ${model} src..."
    python preprocess/02_compute_embeddings.py --entities $PARSED/mouse.json --model $model \
        --out $EMB/${model}_src_emb.npy --gpu 0
    echo "[GPU 0] ${model} tgt..."
    python preprocess/02_compute_embeddings.py --entities $PARSED/human.json --model $model \
        --out $EMB/${model}_tgt_emb.npy --gpu 0
  done

  # 语义类
  for model in distilbert tinybert umlsbert biolinkbert biomedbert pubmedbert_emb e5base; do
    echo "[GPU 0] ${model} src..."
    python preprocess/02_compute_embeddings.py --entities $PARSED/mouse.json --model $model \
        --out $EMB/${model}_src_emb.npy --gpu 0
    echo "[GPU 0] ${model} tgt..."
    python preprocess/02_compute_embeddings.py --entities $PARSED/human.json --model $model \
        --out $EMB/${model}_tgt_emb.npy --gpu 0
  done
) &

# ---- GPU 1：结构类 ----
(
  # 小生成式模型（biogpt/biogpt_large）
  for model in biogpt biogpt_large; do
    echo "[GPU 1] ${model} src..."
    python preprocess/02_compute_embeddings.py --entities $PARSED/mouse.json --model $model \
        --out $EMB/${model}_src_emb.npy --gpu 1
    echo "[GPU 1] ${model} tgt..."
    python preprocess/02_compute_embeddings.py --entities $PARSED/human.json --model $model \
        --out $EMB/${model}_tgt_emb.npy --gpu 1
  done

  # 7B大模型（4bit量化，串行跑避免显存冲突）
  for model in biomistral biomedgpt meditron medalpaca; do
    echo "[GPU 1] ${model} src..."
    python preprocess/02_compute_embeddings.py --entities $PARSED/mouse.json --model $model \
        --out $EMB/${model}_src_emb.npy --gpu 1
    echo "[GPU 1] ${model} tgt..."
    python preprocess/02_compute_embeddings.py --entities $PARSED/human.json --model $model \
        --out $EMB/${model}_tgt_emb.npy --gpu 1
  done
) &

echo "=== 补跑 biomedlm + pmcllama ==="

# GPU 1：biomedlm先跑（~5GB，fp16），再跑pmcllama（13B，4bit量化）
(
  echo "[GPU 1] biomedlm src..."
  python preprocess/02_compute_embeddings.py --entities $PARSED/mouse.json --model biomedlm \
      --out $EMB/biomedlm_src_emb.npy --gpu 1
  echo "[GPU 1] biomedlm tgt..."
  python preprocess/02_compute_embeddings.py --entities $PARSED/human.json --model biomedlm \
      --out $EMB/biomedlm_tgt_emb.npy --gpu 1

  echo "[GPU 1] pmcllama src..."
  python preprocess/02_compute_embeddings.py --entities $PARSED/mouse.json --model pmcllama \
      --out $EMB/pmcllama_src_emb.npy --gpu 0
  echo "[GPU 1] pmcllama tgt..."
  python preprocess/02_compute_embeddings.py --entities $PARSED/human.json --model pmcllama \
      --out $EMB/pmcllama_tgt_emb.npy --gpu 0
)

wait
echo "所有Embedding计算完成"

echo "=== Step 3: 计算相似度矩阵 ==="
ALL_MODELS="biobert pubmedbert sapbert bioelectra scibert clinicalbert \
            biogpt biogpt_large biomistral biomedlm pmcllama \
            distilbert tinybert umlsbert biolinkbert biomedbert pubmedbert_emb e5base"

for model in $ALL_MODELS; do
    echo "计算 ${model} 相似度矩阵..."
    python preprocess/03_compute_similarity.py \
        --src_emb $EMB/${model}_src_emb.npy \
        --tgt_emb $EMB/${model}_tgt_emb.npy \
        --src_entities $PARSED/mouse.json \
        --tgt_entities $PARSED/human.json \
        --out     $EMB/${model}_sim.npy \
        --index   $EMB/entity_index.json
done

echo "=== 全部完成 ==="
echo "输出文件："
ls -lh $EMB/