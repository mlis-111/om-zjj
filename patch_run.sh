#!/bin/bash
# patch_run.sh
# 补跑biomedlm和pmcllama
# 等biomistral跑完后再执行本脚本

set -e
export HF_ENDPOINT=https://hf-mirror.com

PARSED=data/parsed
EMB=embeddings/anatomy

echo "=== 补跑 biomedlm + pmcllama ==="

# GPU 1：biomedlm先跑（~5GB，fp16），再跑pmcllama（13B，4bit量化）
(
  echo "[GPU 1] biomedlm src..."
  python 02_compute_embeddings.py --entities $PARSED/mouse.json --model biomedlm \
      --out $EMB/biomedlm_src_emb.npy --gpu 1
  echo "[GPU 1] biomedlm tgt..."
  python 02_compute_embeddings.py --entities $PARSED/human.json --model biomedlm \
      --out $EMB/biomedlm_tgt_emb.npy --gpu 1

  echo "[GPU 1] pmcllama src..."
  python 02_compute_embeddings.py --entities $PARSED/mouse.json --model pmcllama \
      --out $EMB/pmcllama_src_emb.npy --gpu 0
  echo "[GPU 1] pmcllama tgt..."
  python 02_compute_embeddings.py --entities $PARSED/human.json --model pmcllama \
      --out $EMB/pmcllama_tgt_emb.npy --gpu 0
)

echo "Embedding完成"

echo "=== 计算相似度矩阵 ==="
for model in biomedlm pmcllama; do
    if [ -f "$EMB/${model}_src_emb.npy" ] && [ -f "$EMB/${model}_tgt_emb.npy" ]; then
        echo "计算 ${model} 相似度矩阵..."
        python 03_compute_similarity.py \
            --src_emb $EMB/${model}_src_emb.npy \
            --tgt_emb $EMB/${model}_tgt_emb.npy \
            --src_entities $PARSED/mouse.json \
            --tgt_entities $PARSED/human.json \
            --out     $EMB/${model}_sim.npy \
            --index   $EMB/entity_index.json
    fi
done

echo "=== 完成 ==="
ls -lh $EMB/*.npy | awk '{print $5, $9}'