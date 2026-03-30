#!/bin/bash
set -e

PARSED=data/parsed
EMB=embeddings/anatomy

mkdir -p $PARSED
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
    --index $EMB/entity_index.json

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
