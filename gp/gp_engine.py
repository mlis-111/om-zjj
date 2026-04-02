"""
gp/gp_engine.py
GP主进化循环。
"""
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from gp.crossover import subtree_crossover
from gp.fitness import evaluate_population, evaluate_individual, clear_cache, cache_size
from gp.individual import Individual, create_population
from gp.mutation import adaptive_mutate_population
from gp.selection import three_stage_tournament, elite_selection
from gp.operators import ALL_FILTER_METHODS
from utils.data_loader import ALL_MODELS, OMData
from utils.evaluator import evaluate


def run_gp(data: OMData,
           psa: Set[Tuple[str, str]],
           population_size: int = 50,
           max_generations: int = 20,
           crossover_rate: float = 0.8,
           mutation_rate: float = 0.1,
           elite_ratio: float = 0.1,
           tournament_size: int = 8,
           min_depth: int = 2,
           max_depth: int = 5,
           available_models: List[str] = None,
           available_filters: List[str] = None,
           verbose: bool = True) -> Tuple[Individual, List[Dict]]:
    """
    运行GP进化，返回最优个体和每代日志。

    Args:
        data:             OMData
        psa:              部分标准对齐
        population_size:  种群大小
        max_generations:  最大代数
        crossover_rate:   交叉概率
        mutation_rate:    变异概率
        elite_ratio:      精英保留比例
        tournament_size:  锦标赛大小
        min_depth:        最小树深度
        max_depth:        最大树深度
        available_models: 可用模型列表
        available_filters:可用过滤方法列表
        verbose:          是否打印进化日志

    Returns:
        (best_individual, generation_logs)
    """
    if available_models is None:
        available_models = ALL_MODELS
    if available_filters is None:
        available_filters = ALL_FILTER_METHODS

    n_elite = max(1, int(population_size * elite_ratio))

    # 清空上一次实验的缓存
    clear_cache()

    if verbose:
        print(f"\n{'='*50}")
        print(f"GP进化开始")
        print(f"种群大小={population_size}, 最大代数={max_generations}")
        print(f"模型数={len(available_models)}, 过滤方法数={len(available_filters)}")
        print(f"PSA大小={len(psa)}")
        print(f"{'='*50}\n")

    # 完全随机初始化种群
    population = create_population(
        population_size, available_models, available_filters,
        min_depth, max_depth
    )

    t_start = time.time()
    evaluate_population(population, data, psa)

    best_individual = max(population, key=lambda x: x.fitness).clone()
    generation_logs = []

    for gen in range(max_generations):
        t_gen = time.time()

        # 精英保留
        elites      = elite_selection(population, elite_ratio)
        elite_copies = [e.clone() for e in elites]

        # 交叉产生子代
        new_population = []
        n_offspring    = population_size - n_elite
        while len(new_population) < n_offspring:
            parent_a = three_stage_tournament(
                population, gen, max_generations, data, tournament_size
            )
            parent_b = three_stage_tournament(
                population, gen, max_generations, data, tournament_size
            )
            child_a, child_b = subtree_crossover(
                parent_a, parent_b, crossover_rate, max_depth
            )
            new_population.append(child_a)
            if len(new_population) < n_offspring:
                new_population.append(child_b)

        # 自适应变异
        adaptive_mutate_population(
            new_population, gen, mutation_rate,
            available_models, available_filters, max_depth
        )

        # 评估新个体（缓存加速）
        evaluate_population(new_population, data, psa)

        # 合并精英和新种群
        population = elite_copies + new_population

        # 更新全局最优
        gen_best = max(population, key=lambda x: x.fitness)
        if gen_best.fitness > best_individual.fitness:
            best_individual = gen_best.clone()

        # 统计
        fitnesses = [ind.fitness for ind in population]
        gen_log = {
            "gen":      gen + 1,
            "best":     best_individual.fitness,
            "gen_best": gen_best.fitness,
            "mean":     float(np.mean(fitnesses)),
            "std":      float(np.std(fitnesses)),
            "time":     time.time() - t_gen,
        }
        generation_logs.append(gen_log)

        if verbose:
            print(f"Gen {gen+1:3d}/{max_generations} | "
                  f"best={best_individual.fitness:.4f} | "
                  f"gen_best={gen_best.fitness:.4f} | "
                  f"mean={gen_log['mean']:.4f} | "
                  f"cache={cache_size()} | "
                  f"time={gen_log['time']:.1f}s")

    total_time = time.time() - t_start
    if verbose:
        print(f"\n进化完成，总耗时: {total_time:.1f}s")
        print(f"最优个体: {best_individual}")
        best_alignment = best_individual.evaluate(data)
        p, r, f1 = evaluate(best_alignment, data)
        print(f"\n最终评估结果（完整标准对齐）:")
        print(f"  Precision: {p:.4f}")
        print(f"  Recall:    {r:.4f}")
        print(f"  F1:        {f1:.4f}")

    return best_individual, generation_logs