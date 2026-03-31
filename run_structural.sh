#!/bin/bash
set -e

export HF_ENDPOINT=https://hf-mirror.com

PARSED=data/parsed
EMB=embeddings/anatomy

mkdir -p $EMB

echo "=== 结构类补充模型 Embedding 计算 ==="
echo "共4个模型：biogpt_large / biomistral / meditron / medalpaca"
echo "GPU 0：biogpt_large（1.5B，约6GB显存）"
echo "GPU 1：biomistral / meditron / medalpaca（7B x3，4bit量化，约8GB显存，串行）"
echo ""

# GPU 0：biogpt_large
(
  echo "[GPU 0] biogpt_large src..."
  python 02b_compute_embeddings_structural.py \
      --entities $PARSED/mouse.json --model biogpt_large \
      --out $EMB/biogpt_large_src_emb.npy --gpu 0

  echo "[GPU 0] biogpt_large tgt..."
  python 02b_compute_embeddings_structural.py \
      --entities $PARSED/human.json --model biogpt_large \
      --out $EMB/biogpt_large_tgt_emb.npy --gpu 0
) &

# GPU 1：三个7B模型串行
(
  echo "[GPU 1] biomistral src..."
  python 02b_compute_embeddings_structural.py \
      --entities $PARSED/mouse.json --model biomistral \
      --out $EMB/biomistral_src_emb.npy --gpu 1

  echo "[GPU 1] biomistral tgt..."
  python 02b_compute_embeddings_structural.py \
      --entities $PARSED/human.json --model biomistral \
      --out $EMB/biomistral_tgt_emb.npy --gpu 1

  echo "[GPU 1] meditron src..."
  python 02b_compute_embeddings_structural.py \
      --entities $PARSED/mouse.json --model meditron \
      --out $EMB/meditron_src_emb.npy --gpu 1

  echo "[GPU 1] meditron tgt..."
  python 02b_compute_embeddings_structural.py \
      --entities $PARSED/human.json --model meditron \
      --out $EMB/meditron_tgt_emb.npy --gpu 1

  echo "[GPU 1] medalpaca src..."
  python 02b_compute_embeddings_structural.py \
      --entities $PARSED/mouse.json --model medalpaca \
      --out $EMB/medalpaca_src_emb.npy --gpu 1

  echo "[GPU 1] medalpaca tgt..."
  python 02b_compute_embeddings_structural.py \
      --entities $PARSED/human.json --model medalpaca \
      --out $EMB/medalpaca_tgt_emb.npy --gpu 1
) &

wait
echo "所有结构类Embedding计算完成"

echo "=== 计算相似度矩阵 ==="
for model in biogpt_large biomistral meditron medalpaca; do
    echo "计算 ${model} 相似度矩阵..."
    python 03_compute_similarity.py \
        --src_emb $EMB/${model}_src_emb.npy \
        --tgt_emb $EMB/${model}_tgt_emb.npy \
        --src_entities $PARSED/mouse.json \
        --tgt_entities $PARSED/human.json \
        --out     $EMB/${model}_sim.npy \
        --index   $EMB/entity_index.json
done

echo "=== 全部完成 ==="
ls -lh $EMB/biogpt_large_* $EMB/biomistral_* $EMB/meditron_* $EMB/medalpaca_*
