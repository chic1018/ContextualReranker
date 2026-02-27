"""
超参数搜索脚本：网格搜索 + 可选 Optuna 贝叶斯优化。

用法：
  python tune_params.py                   # 默认网格搜索
  python tune_params.py --mode optuna     # Optuna 贝叶斯搜索（需安装 optuna）
  python tune_params.py --mode grid --top 20
"""

import argparse
import math
import os

from reranker import ContextualReranker
from utils import build_grid_lookup, load_data, ensure_output_dir


# ── 评估函数 ──────────────────────────────────────────────

def evaluate(df, all_actions, index_to_dist, reranker: ContextualReranker, top_k: int = 3):
    """返回 (avg_manhattan_dist, top_k_hit_rate)"""
    ranks = []
    for _, row in df.iterrows():
        t = row['timestamp'].to_pydatetime()
        action = row['action']
        ranks.append(reranker.rank(all_actions, t).index(action))
        reranker.update(action, t)

    dists = [index_to_dist[min(r, len(index_to_dist) - 1)] for r in ranks]
    hits  = [1 if r < top_k else 0 for r in ranks]
    return sum(dists) / len(dists), sum(hits) / len(hits)


def make_reranker(**params) -> ContextualReranker:
    return ContextualReranker(
        weights={
            'period':  params['wp'],
            'env':     params['we'],
            'recency': params['wr'],
        },
        period_max_records=params['period_max_records'],
        period_max_days=params['period_max_days'],
        period_decay_alpha=math.log(2) / params['period_half_life_days'],
        recency_lambda=math.log(2) / params['recency_half_life_min'],
        recency_max=params['recency_max'],
    )


# ── 网格搜索 ──────────────────────────────────────────────

def grid_search(df, all_actions, index_to_dist, top_k: int, top_n: int):
    results = []

    for wp in [i * 0.05 for i in range(21)]:
        for we in [0.0]:                    # 当前无环境数据，固定 0
            wr = round(1.0 - wp - we, 4)
            if wr < 0:
                continue
            for hl_days in [7, 14, 21, 30]:
                for max_days in [14, 21, 30]:
                    for max_rec in [50, 100]:
                        for rec_hl in [10, 15, 30]:
                            params = dict(
                                wp=wp, we=we, wr=wr,
                                period_half_life_days=hl_days,
                                period_max_days=max_days,
                                period_max_records=max_rec,
                                recency_half_life_min=rec_hl,
                                recency_max=10.0,
                            )
                            r = make_reranker(**params)
                            dist, hit = evaluate(df, all_actions, index_to_dist, r, top_k)
                            results.append({**params, 'dist': dist, 'hit': hit})

    results.sort(key=lambda x: x['dist'])
    return results[:top_n]


# ── Optuna 贝叶斯搜索 ─────────────────────────────────────

def optuna_search(df, all_actions, index_to_dist, top_k: int, n_trials: int):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        wp  = trial.suggest_float('wp', 0.3, 0.95)
        we  = 0.0
        wr  = round(1.0 - wp - we, 6)
        params = dict(
            wp=wp, we=we, wr=wr,
            period_half_life_days=trial.suggest_int('period_half_life_days', 5, 60),
            period_max_days=trial.suggest_int('period_max_days', 14, 60),
            period_max_records=trial.suggest_int('period_max_records', 30, 200),
            recency_half_life_min=trial.suggest_int('recency_half_life_min', 5, 60),
            recency_max=trial.suggest_float('recency_max', 5.0, 20.0),
        )
        r = make_reranker(**params)
        dist, _ = evaluate(df, all_actions, index_to_dist, r, top_k)
        return dist  # 最小化曼哈顿距离

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f'\nBest value (dist): {study.best_value:.4f}')
    print('Best params:')
    for k, v in study.best_params.items():
        print(f'  {k}: {v}')
    return study


# ── 主流程 ────────────────────────────────────────────────

def main(
    in_csv='dataset/casas/hh108_filtered.csv',
    out_dir='output',
    mode='grid',
    top_k=3,
    top_n=20,
    n_trials=200,
    grid_m=20,
    grid_n=4,
):
    ensure_output_dir(out_dir)
    df = load_data(in_csv)
    all_actions = sorted(df['action'].unique())
    index_to_dist = build_grid_lookup(grid_m, grid_n)
    print(f'数据: {len(df)} 事件, {len(all_actions)} 动作')

    if mode == 'grid':
        print('网格搜索中...')
        results = grid_search(df, all_actions, index_to_dist, top_k, top_n)
        print(f"\n{'Wp':>5} {'Wr':>5} {'HL_d':>5} {'MaxD':>5} {'MaxR':>5} {'RecHL':>6} | {'Dist':>7} {'Top3':>7}")
        print('-' * 60)
        for r in results:
            print(
                f"{r['wp']:>5.2f} {r['wr']:>5.2f} {r['period_half_life_days']:>5} "
                f"{r['period_max_days']:>5} {r['period_max_records']:>5} {r['recency_half_life_min']:>6} "
                f"| {r['dist']:>7.3f} {r['hit']:>6.1%}"
            )

    elif mode == 'optuna':
        print(f'Optuna 贝叶斯搜索 ({n_trials} trials)...')
        optuna_search(df, all_actions, index_to_dist, top_k, n_trials)

    else:
        raise ValueError(f'未知 mode: {mode}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ContextualReranker 超参数搜索')
    parser.add_argument('--in', dest='in_csv', default='dataset/casas/hh108_filtered.csv')
    parser.add_argument('--out-dir', dest='out_dir', default='output', help='输出目录')
    parser.add_argument('--mode', choices=['grid', 'optuna'], default='grid')
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--top-n', type=int, default=20, help='网格搜索显示前 N 名')
    parser.add_argument('--trials', type=int, default=200, help='Optuna 试验次数')
    parser.add_argument('--grid-m', type=int, default=20)
    parser.add_argument('--grid-n', type=int, default=4)
    args = parser.parse_args()
    main(
        in_csv=args.in_csv,
        out_dir=args.out_dir,
        mode=args.mode,
        top_k=args.top_k,
        top_n=args.top_n,
        n_trials=args.trials,
        grid_m=args.grid_m,
        grid_n=args.grid_n,
    )
