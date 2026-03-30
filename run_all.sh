#!/bin/bash
set -e

export HF_ENDPOINT=https://hf-mirror.com

PARSED=data/parsed
EMB=embeddings/anatomy

mkdir -p $PARSED
mkdir -p $EMB

echo "=== Step 1: 解析本体 ==="
python 01_parse_ontology.py --owl data/oaei/anatomy/mouse.owl --out $PARSED/mouse.json
python 01_parse_ontology.py --owl data/oaei/anatomy/human.owl  --out $PARSED/human.json

echo "=== Step 2: 计算Embedding ==="
# 共13个模型，只有GPU 0和GPU 1
# GPU 0：字符串类（biobert/pubmedbert/sapbert/bioelectra/scibert/clinicalbert）+ 语义类部分
# GPU 1：结构类（biogpt/biomistral）+ 语义类其余
# biomistral单独占GPU 1，跑完后再跑其他
# 两组并行，组内串行

# GPU 0：字符串类6个 + 语义类3个（distilbert/tinybert/umlsbert）
(
  echo "[GPU 0] biobert src..."
  python 02_compute_embeddings.py --entities $PARSED/mouse.json --model biobert \
      --out $EMB/biobert_src_emb.npy --gpu 0
  echo "[GPU 0] biobert tgt..."
  python 02_compute_embeddings.py --entities $PARSED/human.json --model biobert \
      --out $EMB/biobert_tgt_emb.npy --gpu 0

  echo "[GPU 0] pubmedbert src..."
  python 02_compute_embeddings.py --entities $PARSED/mouse.json --model pubmedbert \
      --out $EMB/pubmedbert_src_emb.npy --gpu 0
  echo "[GPU 0] pubmedbert tgt..."
  python 02_compute_embeddings.py --entities $PARSED/human.json --model pubmedbert \
      --out $EMB/pubmedbert_tgt_emb.npy --gpu 0

  echo "[GPU 0] sapbert src..."
  python 02_compute_embeddings.py --entities $PARSED/mouse.json --model sapbert \
      --out $EMB/sapbert_src_emb.npy --gpu 0
  echo "[GPU 0] sapbert tgt..."
  python 02_compute_embeddings.py --entities $PARSED/human.json --model sapbert \
      --out $EMB/sapbert_tgt_emb.npy --gpu 0

  echo "[GPU 0] bioelectra src..."
  python 02_compute_embeddings.py --entities $PARSED/mouse.json --model bioelectra \
      --out $EMB/bioelectra_src_emb.npy --gpu 0
  echo "[GPU 0] bioelectra tgt..."
  python 02_compute_embeddings.py --entities $PARSED/human.json --model bioelectra \
      --out $EMB/bioelectra_tgt_emb.npy --gpu 0

  echo "[GPU 0] scibert src..."
  python 02_compute_embeddings.py --entities $PARSED/mouse.json --model scibert \
      --out $EMB/scibert_src_emb.npy --gpu 0
  echo "[GPU 0] scibert tgt..."
  python 02_compute_embeddings.py --entities $PARSED/human.json --model scibert \
      --out $EMB/scibert_tgt_emb.npy --gpu 0

  echo "[GPU 0] clinicalbert src..."
  python 02_compute_embeddings.py --entities $PARSED/mouse.json --model clinicalbert \
      --out $EMB/clinicalbert_src_emb.npy --gpu 0
  echo "[GPU 0] clinicalbert tgt..."
  python 02_compute_embeddings.py --entities $PARSED/human.json --model clinicalbert \
      --out $EMB/clinicalbert_tgt_emb.npy --gpu 0

  echo "[GPU 0] distilbert src..."
  python 02_compute_embeddings.py --entities $PARSED/mouse.json --model distilbert \
      --out $EMB/distilbert_src_emb.npy --gpu 0
  echo "[GPU 0] distilbert tgt..."
  python 02_compute_embeddings.py --entities $PARSED/human.json --model distilbert \
      --out $EMB/distilbert_tgt_emb.npy --gpu 0

  echo "[GPU 0] tinybert src..."
  python 02_compute_embeddings.py --entities $PARSED/mouse.json --model tinybert \
      --out $EMB/tinybert_src_emb.npy --gpu 0
  echo "[GPU 0] tinybert tgt..."
  python 02_compute_embeddings.py --entities $PARSED/human.json --model tinybert \
      --out $EMB/tinybert_tgt_emb.npy --gpu 0

  echo "[GPU 0] umlsbert src..."
  python 02_compute_embeddings.py --entities $PARSED/mouse.json --model umlsbert \
      --out $EMB/umlsbert_src_emb.npy --gpu 0
  echo "[GPU 0] umlsbert tgt..."
  python 02_compute_embeddings.py --entities $PARSED/human.json --model umlsbert \
      --out $EMB/umlsbert_tgt_emb.npy --gpu 0
) &

# GPU 1：结构类2个（biogpt/biomistral）+ 语义类2个（bluebert/biolinkbert）
# biomistral是7B模型放最后跑，避免显存影响前面的模型
(
  echo "[GPU 1] biogpt src..."
  python 02_compute_embeddings.py --entities $PARSED/mouse.json --model biogpt \
      --out $EMB/biogpt_src_emb.npy --gpu 1
  echo "[GPU 1] biogpt tgt..."
  python 02_compute_embeddings.py --entities $PARSED/human.json --model biogpt \
      --out $EMB/biogpt_tgt_emb.npy --gpu 1

  echo "[GPU 1] bluebert src..."
  python 02_compute_embeddings.py --entities $PARSED/mouse.json --model bluebert \
      --out $EMB/bluebert_src_emb.npy --gpu 1
  echo "[GPU 1] bluebert tgt..."
  python 02_compute_embeddings.py --entities $PARSED/human.json --model bluebert \
      --out $EMB/bluebert_tgt_emb.npy --gpu 1

  echo "[GPU 1] biolinkbert src..."
  python 02_compute_embeddings.py --entities $PARSED/mouse.json --model biolinkbert \
      --out $EMB/biolinkbert_src_emb.npy --gpu 1
  echo "[GPU 1] biolinkbert tgt..."
  python 02_compute_embeddings.py --entities $PARSED/human.json --model biolinkbert \
      --out $EMB/biolinkbert_tgt_emb.npy --gpu 1

  echo "[GPU 1] biomistral src..."
  python 02_compute_embeddings.py --entities $PARSED/mouse.json --model biomistral \
      --out $EMB/biomistral_src_emb.npy --gpu 1
  echo "[GPU 1] biomistral tgt..."
  python 02_compute_embeddings.py --entities $PARSED/human.json --model biomistral \
      --out $EMB/biomistral_tgt_emb.npy --gpu 1
) &

wait
echo "所有Embedding计算完成"

echo "=== Step 3: 计算相似度矩阵 ==="
ALL_MODELS="biobert pubmedbert sapbert bioelectra scibert clinicalbert biogpt biomistral distilbert bluebert tinybert biolinkbert umlsbert"

for model in $ALL_MODELS; do
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
echo "输出文件："
ls -lh $EMB/