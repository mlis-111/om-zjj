"""
main.py
GP本体匹配系统入口。

用法:
  # 调试模式（快速验证流程）
  python main.py --mode debug

  # 正式运行
  python main.py --mode full

  # 自定义参数
  python main.py --population_size 50 --max_generations 20
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
from utils.psa_builder import build_psa_from_files
from utils.evaluator import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="GP本体匹配系统")

    parser.add_argument("--mode", type=str, default="debug",
                        choices=["debug", "full"],
                        help="运行模式：debug(快速验证) / full(正式实验)")

    # 数据路径
    parser.add_argument("--emb_dir",       default="embeddings/anatomy")
    parser.add_argument("--src_entities",  default="data/parsed/mouse.json")
    parser.add_argument("--tgt_entities",  default="data/parsed/human.json")
    parser.add_argument("--src_hierarchy", default="data/parsed/mouse_hierarchy.json")
    parser.add_argument("--tgt_hierarchy", default="data/parsed/human_hierarchy.json")
    parser.add_argument("--reference",     default="data/oaei/anatomy/reference.rdf")

    # GP参数（可覆盖mode的默认值）
    parser.add_argument("--population_size",  type=int,   default=None)
    parser.add_argument("--max_generations",  type=int,   default=None)
    parser.add_argument("--crossover_rate",   type=float, default=0.8)
    parser.add_argument("--mutation_rate",    type=float, default=0.1)
    parser.add_argument("--elite_ratio",      type=float, default=0.1)
    parser.add_argument("--tournament_size",  type=int,   default=8)
    parser.add_argument("--min_depth",        type=int,   default=2)
    parser.add_argument("--max_depth",        type=int,   default=5)
    parser.add_argument("--delta",            type=float, default=0.3,
                        help="PSA构建的Top-δ比例")

    # 输出
    parser.add_argument("--output_dir", default="results",
                        help="结果输出目录")
    parser.add_argument("--run_id",     default=None,
                        help="实验ID，默认使用时间戳")

    return parser.parse_args()


def main():
    args = parse_args()

    # 根据mode设置默认参数
    if args.mode == "debug":
        pop_size = args.population_size or 10
        max_gen  = args.max_generations or 5
        print("[DEBUG模式] population=10, generations=5")
    else:
        pop_size = args.population_size or 50
        max_gen  = args.max_generations or 20
        print("[FULL模式] population=50, generations=20")

    # 输出目录
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)

    # ---- 加载数据 ----
    print("\n=== 加载数据 ===")
    data = load_om_data(
        emb_dir=args.emb_dir,
        src_entities=args.src_entities,
        tgt_entities=args.tgt_entities,
        src_hierarchy=args.src_hierarchy,
        tgt_hierarchy=args.tgt_hierarchy,
        reference_path=args.reference,
    )

    # ---- 构建PSA ----
    print("\n=== 构建PSA ===")
    psa = build_psa_from_files(
        data,
        src_json_path=args.src_entities,
        tgt_json_path=args.tgt_entities,
        delta=args.delta,
    )

    # ---- 运行GP ----
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

    # ---- 最终评估 ----
    print("\n=== 最终评估 ===")
    best_alignment = best_individual.evaluate(data)
    precision, recall, f1 = evaluate(best_alignment, data)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")

    # ---- 保存结果 ----
    results = {
        "run_id":          run_id,
        "mode":            args.mode,
        "population_size": pop_size,
        "max_generations": max_gen,
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
