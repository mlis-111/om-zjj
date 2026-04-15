#!/bin/bash
# 对所有已有数据集跑传播有效性分析
DATASETS=(omim-ordo ncit-doid snomed-fma snomed-ncit-pharm snomed-ncit-neoplas anatomy)

for DS in "${DATASETS[@]}"; do
    echo ""
    echo "############################################################"
    echo "# $DS"
    echo "############################################################"
    python analyze_propagation.py --dataset "$DS" --depth 3
done
