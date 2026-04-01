# GP-based Ontology Matching System

基于遗传编程的本体匹配系统，使用双树结构（组合树+过滤树）自动进化最优匹配策略。

## 快速开始

### 1. 环境配置

```bash
pip install transformers owlready2 rdflib numpy scipy scikit-learn tqdm accelerate bitsandbytes torch
```

### 2. 下载模型

```bash
bash model_download.sh
```

支持的模型：BioBERT, BioGPT, BioMistral, PubMedBERT, SapBERT

### 3. 数据准备

```bash
# 解析本体文件
python 01_parse_ontology.py --owl data/oaei/anatomy/mouse.owl --out data/parsed/mouse.json
python 01_parse_ontology.py --owl data/oaei/anatomy/human.owl --out data/parsed/human.json

# 计算嵌入向量
python 02_compute_embeddings.py \
  --src data/parsed/mouse.json \
  --tgt data/parsed/human.json \
  --out embeddings/anatomy

# 计算相似度矩阵
python 03_compute_similarity.py --emb_dir embeddings/anatomy
```

### 4. 运行匹配

```bash
# 调试模式（快速验证）
python main.py --mode debug

# 正式运行
python main.py --mode full

# 自定义参数
python main.py --population_size 50 --max_generations 20 --delta 0.3
```

## 项目结构

```
├── main.py                    # 主入口
├── 01_parse_ontology.py       # 本体解析
├── 02_compute_embeddings.py   # 嵌入计算
├── 03_compute_similarity.py   # 相似度计算
├── model_download.sh          # 模型下载脚本
├── gp/                        # GP核心模块
│   ├── gp_engine.py          # GP进化引擎
│   ├── individual.py         # 个体定义（双树结构）
│   ├── operators.py          # 算子库（算术+逻辑+过滤）
│   ├── crossover.py          # 交叉操作
│   ├── mutation.py           # 变异操作
│   ├── selection.py          # 选择策略
│   └── fitness.py            # 适应度函数
├── utils/                     # 工具模块
│   ├── data_loader.py        # 数据加载
│   ├── psa_builder.py        # PSA构建
│   ├── evaluator.py          # 评估指标
│   └── extract_hierarchy.py  # 层次结构提取
├── data/                      # 数据目录
│   ├── oaei/                 # OAEI数据集
│   └── parsed/               # 解析后的JSON
├── embeddings/                # 嵌入向量和相似度矩阵
└── results/                   # 实验结果
```

## 核心概念

### 双树结构

每个个体由两棵树组成：

1. **组合树**：组合多个模型的相似度矩阵
   - 叶节点：预训练模型（BioBERT, BioGPT等）
   - 内部节点：算术操作（add, mul, max, avg等）

2. **过滤树**：过滤候选匹配对
   - 叶节点：过滤策略（threshold, top_k, stable_marriage等）
   - 内部节点：逻辑操作（intersection, union, xor）

### PSA（Probably Similar Alignment）

预筛选机制，减少搜索空间：
- 对每个源实体，保留相似度最高的 top-δ 目标实体
- 默认 δ=0.3（保留前30%）

### GP进化流程

1. 初始化种群（随机生成双树个体）
2. 评估适应度（F1-score）
3. 选择（锦标赛选择）
4. 交叉（子树交换）
5. 变异（节点替换、子树生成）
6. 精英保留
7. 重复2-6直到达到最大代数

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | debug | 运行模式：debug(10个体/5代) 或 full(50个体/20代) |
| `--population_size` | 10/50 | 种群大小 |
| `--max_generations` | 5/20 | 最大进化代数 |
| `--crossover_rate` | 0.8 | 交叉概率 |
| `--mutation_rate` | 0.1 | 变异概率 |
| `--elite_ratio` | 0.1 | 精英保留比例 |
| `--tournament_size` | 8 | 锦标赛选择大小 |
| `--min_depth` | 2 | 树最小深度 |
| `--max_depth` | 5 | 树最大深度 |
| `--delta` | 0.3 | PSA的top-δ比例 |

## 输出结果

结果保存在 `results/<run_id>/results.json`：

```json
{
  "run_id": "20260401_121530",
  "mode": "full",
  "final_precision": 0.8523,
  "final_recall": 0.7891,
  "final_f1": 0.8194,
  "best_fitness": 0.8194,
  "best_models": ["biobert", "sapbert"],
  "generation_logs": [...]
}
```

## 算子库

### 组合树算术操作
`add`, `sub`, `mul`, `div`, `max`, `min`, `avg`

### 过滤树逻辑操作
`intersection`, `union`, `xor`

### 过滤策略

**数值类**：`fixed_threshold`, `max_value`, `median`, `mean`, `var_mean`, `top_k`

**推理类**：`structure_filter`, `kde`, `kmeans`

**启发式**：`nde`, `stable_marriage`, `hungarian`, `ant_colony`, `random_hill_climbing`

## 数据集

支持 OAEI（Ontology Alignment Evaluation Initiative）数据集：
- Anatomy（解剖学）：mouse.owl ↔ human.owl
- Conference（会议）：16个会议本体的两两匹配

## 引用

如果使用本项目，请引用相关论文。

## License

MIT
