#!/bin/bash
# run_embed_all_bioml.sh
# 为所有 Bio-ML 2024 数据集完整跑预处理 + embedding 流水线
# omim-ordo 的 embedding 若已在跑，会自动跳过已存在的 sim.npy
#
# 用法: bash run_embed_all_bioml.sh [GPU0] [GPU1]
# 示例: bash run_embed_all_bioml.sh 0 1

GPU0=${1:-0}
GPU1=${2:-1}

MODELS=(
    biobert pubmedbert sapbert bioelectra scibert clinicalbert
    biogpt biogpt_large biomistral biomedlm pmcllama
    distilbert tinybert umlsbert biolinkbert biomedbert pubmedbert_emb e5base
)

# 数据集列表: "名称|src_owl|tgt_owl|src_json|tgt_json|src_hier|tgt_hier|emb_dir"
DATASETS=(
    "omim-ordo|data/bio-ml-2024/omim-ordo/omim.owl|data/bio-ml-2024/omim-ordo/ordo.owl|data/parsed/omim_ordo_src.json|data/parsed/omim_ordo_tgt.json|data/parsed/omim_ordo_src_hierarchy.json|data/parsed/omim_ordo_tgt_hierarchy.json|embeddings/omim-ordo"
    "ncit-doid|data/bio-ml-2024/ncit-doid/ncit.owl|data/bio-ml-2024/ncit-doid/doid.owl|data/parsed/ncit_doid_src.json|data/parsed/ncit_doid_tgt.json|data/parsed/ncit_doid_src_hierarchy.json|data/parsed/ncit_doid_tgt_hierarchy.json|embeddings/ncit-doid"
    "snomed-fma|data/bio-ml-2024/snomed-fma.body/snomed.body.owl|data/bio-ml-2024/snomed-fma.body/fma.body.owl|data/parsed/snomed_fma_src.json|data/parsed/snomed_fma_tgt.json|data/parsed/snomed_fma_src_hierarchy.json|data/parsed/snomed_fma_tgt_hierarchy.json|embeddings/snomed-fma"
    "snomed-ncit-pharm|data/bio-ml-2024/snomed-ncit.pharm/snomed.pharm.owl|data/bio-ml-2024/snomed-ncit.pharm/ncit.pharm.owl|data/parsed/snomed_ncit_pharm_src.json|data/parsed/snomed_ncit_pharm_tgt.json|data/parsed/snomed_ncit_pharm_src_hierarchy.json|data/parsed/snomed_ncit_pharm_tgt_hierarchy.json|embeddings/snomed-ncit-pharm"
    "snomed-ncit-neoplas|data/bio-ml-2024/snomed-ncit.neoplas/snomed.neoplas.owl|data/bio-ml-2024/snomed-ncit.neoplas/ncit.neoplas.owl|data/parsed/snomed_ncit_neoplas_src.json|data/parsed/snomed_ncit_neoplas_tgt.json|data/parsed/snomed_ncit_neoplas_src_hierarchy.json|data/parsed/snomed_ncit_neoplas_tgt_hierarchy.json|embeddings/snomed-ncit-neoplas"
)

mkdir -p logs

echo "========================================"
echo "Bio-ML 2024 全数据集预处理 + Embedding"
echo "GPU0=$GPU0  GPU1=$GPU1"
echo "========================================"

for ENTRY in "${DATASETS[@]}"; do
    IFS='|' read -r NAME SRC_OWL TGT_OWL SRC_JSON TGT_JSON SRC_HIER TGT_HIER EMB_DIR <<< "$ENTRY"

    echo ""
    echo "###################################################"
    echo "# 数据集: $NAME"
    echo "###################################################"
    mkdir -p "$EMB_DIR"

    # ── Step 1: 解析本体（含 text 字段）──────────────────────────
    # 如果 json 已存在且有 text 字段则跳过
    if python3 -c "
import json, sys
try:
    d = json.load(open('$SRC_JSON'))
    sys.exit(0 if d and 'text' in d[0] else 1)
except: sys.exit(1)
" 2>/dev/null; then
        echo "[Step1] $NAME src JSON 已存在，跳过解析"
    else
        echo "[Step1] 解析 $NAME src..."
        python preprocess/01_parse_ontology.py --owl "$SRC_OWL" --out "$SRC_JSON"
    fi

    if python3 -c "
import json, sys
try:
    d = json.load(open('$TGT_JSON'))
    sys.exit(0 if d and 'text' in d[0] else 1)
except: sys.exit(1)
" 2>/dev/null; then
        echo "[Step1] $NAME tgt JSON 已存在，跳过解析"
    else
        echo "[Step1] 解析 $NAME tgt..."
        python preprocess/01_parse_ontology.py --owl "$TGT_OWL" --out "$TGT_JSON"
    fi

    # ── Step 2: 提取层次关系（若不存在）─────────────────────────
    if [ ! -f "$SRC_HIER" ]; then
        echo "[Step2] 提取 $NAME src 层次关系..."
        python utils/extract_hierarchy.py --owl "$SRC_OWL" --out "$SRC_HIER"
    else
        echo "[Step2] $NAME src 层次关系已存在，跳过"
    fi

    if [ ! -f "$TGT_HIER" ]; then
        echo "[Step2] 提取 $NAME tgt 层次关系..."
        python utils/extract_hierarchy.py --owl "$TGT_OWL" --out "$TGT_HIER"
    else
        echo "[Step2] $NAME tgt 层次关系已存在，跳过"
    fi

    # ── Step 3+4: 18个模型 embedding + 相似度 ───────────────────
    echo "[Step3] 开始计算 $NAME 的 embedding 和相似度..."
    for MODEL in "${MODELS[@]}"; do
        SRC_EMB="$EMB_DIR/${MODEL}_src_emb.npy"
        TGT_EMB="$EMB_DIR/${MODEL}_tgt_emb.npy"
        SIM_OUT="$EMB_DIR/${MODEL}_sim.npy"
        INDEX_OUT="$EMB_DIR/entity_index.json"

        # 已存在则跳过
        if [ -f "$SIM_OUT" ]; then
            echo "  [跳过] $MODEL sim 已存在"
            continue
        fi

        echo "  ── $MODEL ──"
        python preprocess/02_compute_embeddings.py \
            --entities "$SRC_JSON" --model "$MODEL" --out "$SRC_EMB" --gpu "$GPU0"
        python preprocess/02_compute_embeddings.py \
            --entities "$TGT_JSON" --model "$MODEL" --out "$TGT_EMB" --gpu "$GPU1"
        python preprocess/03_compute_similarity.py \
            --src_emb "$SRC_EMB" --tgt_emb "$TGT_EMB" \
            --src_entities "$SRC_JSON" --tgt_entities "$TGT_JSON" \
            --out "$SIM_OUT" --index "$INDEX_OUT"

        echo "  ✓ $MODEL -> $SIM_OUT"
    done

    echo "### $NAME 完成 ###"
done

echo ""
echo "========================================"
echo "全部数据集处理完成"
echo "========================================"