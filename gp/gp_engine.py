"""
gp/gp_engine.py  —— 带详细计时的版本
在每一代内对以下阶段单独计时：
  1. elite_selection      精英保留
  2. tournament+crossover 锦标赛选择 + 交叉
  3. mutation             自适应变异
  4. evaluate_new         新子代评估
  5. merge+update         合并 & 更新全局最优
并在每代日志中输出各阶段耗时，进化结束后打印汇总表。
"""
import time
from collections import defaultdict
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


# ──────────────────────────────────────────────────────────────
# 辅助：简洁计时上下文
# ──────────────────────────────────────────────────────────────
class _Timer:
    """用法：with _Timer() as t: ...; elapsed = t.elapsed"""
    def __enter__(self):
        self._start = time.perf_counter()
        return self
    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start


def _fmt(seconds: float) -> str:
    """把秒数格式化成易读字符串"""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    return f"{seconds:.2f}s"


# ──────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────
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

    if available_models is None:
        available_models = ALL_MODELS
    if available_filters is None:
        available_filters = ALL_FILTER_METHODS

    n_elite = max(1, int(population_size * elite_ratio))
    clear_cache()

    if verbose:
        print(f"\n{'='*60}")
        print(f"GP进化开始（含详细计时）")
        print(f"种群大小={population_size}, 最大代数={max_generations}")
        print(f"模型数={len(available_models)}, 过滤方法数={len(available_filters)}")
        print(f"PSA大小={len(psa)}")
        print(f"{'='*60}\n")

    # ── 初始化 ──────────────────────────────────────────────────
    with _Timer() as t_init:
        population = create_population(
            population_size, available_models, available_filters,
            min_depth, max_depth
        )
    if verbose:
        print(f"[初始化种群]  {_fmt(t_init.elapsed)}")

    with _Timer() as t_eval0:
        evaluate_population(population, data, psa)
    if verbose:
        print(f"[第0代评估]   {_fmt(t_eval0.elapsed)}\n")

    best_individual = max(population, key=lambda x: x.fitness).clone()
    generation_logs: List[Dict] = []

    # 跨代累计计时（用于最终汇总）
    stage_totals: Dict[str, float] = defaultdict(float)

    t_gp_start = time.perf_counter()

    for gen in range(max_generations):
        stage_times: Dict[str, float] = {}

        # ── 1. 精英保留 ─────────────────────────────────────────
        with _Timer() as t:
            elites       = elite_selection(population, elite_ratio)
            elite_copies = [e.clone() for e in elites]
        stage_times["elite"]    = t.elapsed
        stage_totals["elite"]  += t.elapsed

        # ── 2. 锦标赛选择 + 交叉 ────────────────────────────────
        new_population = []
        n_offspring    = population_size - n_elite
        with _Timer() as t:
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
        stage_times["tournament+crossover"]    = t.elapsed
        stage_totals["tournament+crossover"]  += t.elapsed

        # ── 3. 自适应变异 ────────────────────────────────────────
        with _Timer() as t:
            adaptive_mutate_population(
                new_population, gen, mutation_rate,
                available_models, available_filters, max_depth
            )
        stage_times["mutation"]    = t.elapsed
        stage_totals["mutation"]  += t.elapsed

        # ── 4. 评估新子代 ────────────────────────────────────────
        with _Timer() as t:
            evaluate_population(new_population, data, psa)
        stage_times["evaluate_new"]    = t.elapsed
        stage_totals["evaluate_new"]  += t.elapsed

        # ── 5. 合并 & 更新全局最优 ───────────────────────────────
        with _Timer() as t:
            population  = elite_copies + new_population
            gen_best    = max(population, key=lambda x: x.fitness)
            if gen_best.fitness > best_individual.fitness:
                best_individual = gen_best.clone()
        stage_times["merge+update"]    = t.elapsed
        stage_totals["merge+update"]  += t.elapsed

        gen_total = sum(stage_times.values())

        # ── 日志 ─────────────────────────────────────────────────
        fitnesses = [ind.fitness for ind in population]
        gen_log   = {
            "gen":      gen + 1,
            "best":     best_individual.fitness,
            "gen_best": gen_best.fitness,
            "mean":     float(np.mean(fitnesses)),
            "std":      float(np.std(fitnesses)),
            "time":     gen_total,
            "stage_times": dict(stage_times),  # 各阶段秒数
        }
        generation_logs.append(gen_log)

        if verbose:
            # 主行
            print(f"Gen {gen+1:3d}/{max_generations} | "
                  f"best={best_individual.fitness:.4f} | "
                  f"gen_best={gen_best.fitness:.4f} | "
                  f"mean={gen_log['mean']:.4f} | "
                  f"cache={cache_size()} | "
                  f"total={_fmt(gen_total)}")
            # 子行：各阶段耗时及占比
            # parts = []
            # for stage, t_s in stage_times.items():
            #     pct = t_s / gen_total * 100 if gen_total > 0 else 0
            #     parts.append(f"{stage}={_fmt(t_s)}({pct:.0f}%)")
            # print(f"         ↳ {' | '.join(parts)}")

    # ── 进化结束汇总 ─────────────────────────────────────────────
    total_gp_time = time.perf_counter() - t_gp_start
    total_gp_time_with_init = t_init.elapsed + t_eval0.elapsed + total_gp_time

    if verbose:
        print(f"\n{'='*60}")
        print(f"进化完成")
        print(f"  初始化+第0代评估: {_fmt(t_init.elapsed + t_eval0.elapsed)}")
        print(f"  进化循环总耗时:   {_fmt(total_gp_time)}")
        print(f"  整体总耗时:       {_fmt(total_gp_time_with_init)}")
        print(f"\n── 各阶段累计耗时（进化循环内）──")
        header = f"  {'阶段':<25} {'累计':>10} {'占比':>7}"
        print(header)
        print(f"  {'-'*45}")
        for stage, total_t in sorted(stage_totals.items(),
                                      key=lambda x: -x[1]):
            pct = total_t / total_gp_time * 100 if total_gp_time > 0 else 0
            print(f"  {stage:<25} {_fmt(total_t):>10} {pct:>6.1f}%")
        print(f"  {'─'*45}")
        print(f"  {'合计 (进化循环)':<25} {_fmt(total_gp_time):>10} {'100.0%':>7}")
        print(f"{'='*60}\n")

        # 最终评估
        print(f"最优个体: {best_individual}")
        best_alignment = best_individual.evaluate(data)
        p, r, f1 = evaluate(best_alignment, data)
        print(f"\n最终评估结果（完整标准对齐）:")
        print(f"  Precision: {p:.4f}")
        print(f"  Recall:    {r:.4f}")
        print(f"  F1:        {f1:.4f}")

    return best_individual, generation_logs