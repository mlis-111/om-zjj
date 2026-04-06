"""
main.py
GP本体匹配系统入口。

用法:
  python main.py --mode debug
  python main.py --mode full
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gp.gp_engine import run_gp
from gp.operators import ALL_FILTER_METHODS
from utils.data_loader import load_om_data, ALL_MODELS
from utils.augmented_psa import build_augmented_psa_from_files
from utils.evaluator import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="GP本体匹配系统")
    parser.add_argument("--mode", default="debug", choices=["debug", "full"])

    # 数据路径
    parser.add_argument("--emb_dir",       default="embeddings/anatomy")
    parser.add_argument("--src_entities",  default="data/parsed/mouse.json")
    parser.add_argument("--tgt_entities",  default="data/parsed/human.json")
    parser.add_argument("--src_hierarchy", default="data/parsed/mouse_hierarchy.json")
    parser.add_argument("--tgt_hierarchy", default="data/parsed/human_hierarchy.json")
    parser.add_argument("--reference",     default="data/oaei/anatomy/reference.rdf")

    # GP参数
    parser.add_argument("--population_size",  type=int,   default=None)
    parser.add_argument("--max_generations",  type=int,   default=None)
    parser.add_argument("--crossover_rate",   type=float, default=0.8)
    parser.add_argument("--mutation_rate",    type=float, default=0.1)
    parser.add_argument("--elite_ratio",      type=float, default=0.1)
    parser.add_argument("--tournament_size",  type=int,   default=8)
    parser.add_argument("--min_depth",        type=int,   default=2)
    parser.add_argument("--max_depth",        type=int,   default=5)

    # 增强PSA参数
    parser.add_argument("--psa_k",        type=float, default=0.7,
                        help="第一阶段相似度阈值")
    parser.add_argument("--psa_depth",    type=int,   default=3,
                        help="第二阶段传播深度")
    parser.add_argument("--psa_thresh1",  type=float, default=1.0,
                        help="Layer1字符串相似度阈值")
    parser.add_argument("--psa_thresh2",  type=float, default=0.8,
                        help="Layer2字符串相似度阈值")
    parser.add_argument("--psa_w1",       type=float, default=0.8,
                        help="Layer1权重")
    parser.add_argument("--psa_w2",       type=float, default=0.2,
                        help="Layer2权重")

    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--run_id",     default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "debug":
        pop_size = args.population_size or 10
        max_gen  = args.max_generations or 5
        print("[DEBUG模式] population=10, generations=5")
    else:
        pop_size = args.population_size or 50
        max_gen  = args.max_generations or 20
        print("[FULL模式] population=50, generations=20")

    run_id     = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    print("\n=== 加载数据 ===")
    data = load_om_data(
        emb_dir=args.emb_dir,
        src_entities=args.src_entities,
        tgt_entities=args.tgt_entities,
        src_hierarchy=args.src_hierarchy,
        tgt_hierarchy=args.tgt_hierarchy,
        reference_path=args.reference,
    )

    # 构建增强PSA
    print("\n=== 构建增强PSA ===")
    psa = build_augmented_psa_from_files(
        data,
        src_json_path=args.src_entities,
        tgt_json_path=args.tgt_entities,
        k=args.psa_k,
        max_depth=args.psa_depth,
        thresh1=args.psa_thresh1,
        thresh2=args.psa_thresh2,
        n_gram=2,
        w1=args.psa_w1,
        w2=args.psa_w2,
    )

    # 运行GP
    print("\n=== 开始GP进化 ===")
    best_individual, logs = run_gp(
        data=data,
        psa=psa,
        population_size=pop_size,
        max_generations=max_gen,
        crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate,
        elite_ratio=args.elite_ratio,
        tournament_size=args.tournament_size,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        available_models=ALL_MODELS,
        available_filters=ALL_FILTER_METHODS,
        verbose=True,
    )

    # 最终评估
    print("\n=== 最终评估 ===")
    best_alignment    = best_individual.evaluate(data)
    precision, recall, f1 = evaluate(best_alignment, data)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")

    # 保存结果
    results = {
        "run_id":          run_id,
        "mode":            args.mode,
        "population_size": pop_size,
        "max_generations": max_gen,
        "psa_config": {
            "k": args.psa_k, "depth": args.psa_depth,
            "thresh1": args.psa_thresh1, "thresh2": args.psa_thresh2,
            "w1": args.psa_w1, "w2": args.psa_w2,
            "layer1_size": len(psa.layer1), "layer2_size": len(psa.layer2),
        },
        "final_precision": precision,
        "final_recall":    recall,
        "final_f1":        f1,
        "best_fitness":    best_individual.fitness,
        "best_models":     best_individual.combination_tree.get_leaf_models(),
        "generation_logs": logs,
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {results_path}")
    return results


if __name__ == "__main__":
    main()