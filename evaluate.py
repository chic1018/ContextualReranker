"""
评估模块：在事件序列上运行重排算法，计算 Manhattan Distance 和 Top-K 命中率。
支持新算法 (ContextualReranker) 与纯 Recency 基线的对比。
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from reranker import ContextualReranker
from utils import build_grid_lookup, load_data, ensure_output_dir


# ── 纯 Recency 基线 ──────────────────────────────────────

def run_recency_baseline(df, all_actions):
    """纯 Recency 基线：按最近使用时间排序"""
    last_ts = {a: -1e18 for a in all_actions}
    ranks = []

    for _, row in df.iterrows():
        ts = row['timestamp'].timestamp()
        action = row['action']

        ordered = sorted(all_actions, key=lambda a: (-last_ts[a], a))
        rank = ordered.index(action)
        ranks.append(rank)

        last_ts[action] = ts

    return ranks


# ── ContextualReranker 评估 ──────────────────────────────

def run_contextual_reranker(df, all_actions, reranker: ContextualReranker):
    """运行 ContextualReranker，返回每个事件中目标设备的排名 (0-based)"""
    ranks = []

    for _, row in df.iterrows():
        current_time = row['timestamp'].to_pydatetime()
        action = row['action']

        ranked = reranker.rank(all_actions, current_time)
        rank = ranked.index(action)
        ranks.append(rank)

        reranker.update(action, current_time)

    return ranks


# ── 指标计算 ──────────────────────────────────────────────

def compute_metrics(ranks, index_to_dist, top_k=3):
    """
    给定 0-based rank 序列，计算：
    - manhattan distances（每个事件的曼哈顿距离）
    - top_k hit rate（目标在前 K 位的比例）
    """
    distances = [index_to_dist[min(r, len(index_to_dist) - 1)] for r in ranks]
    hits = [1 if r < top_k else 0 for r in ranks]
    return distances, hits


# ── 主流程 ────────────────────────────────────────────────

def main(
    in_csv='dataset/casas/hh108_filtered.csv',
    out_dir='output',
    out_csv='eval_results.csv',
    out_png='eval_comparison.png',
    window_size=100,
    grid_m=20,
    grid_n=4,
    top_k=3,
    datetime_cols=('date', 'time'),
):
    ensure_output_dir(out_dir)
    out_csv_path = os.path.join(out_dir, out_csv)
    out_png_path = os.path.join(out_dir, out_png)

    # 1) 读取数据
    df = load_data(in_csv, datetime_cols)
    all_actions = sorted(df['action'].unique())
    index_to_dist = build_grid_lookup(grid_m, grid_n)

    print(f"数据量: {len(df)} 事件, {len(all_actions)} 个设备/动作")

    # 2) 运行两个算法
    print("运行 Recency 基线...")
    recency_ranks = run_recency_baseline(df, all_actions)

    print("运行 ContextualReranker...")
    reranker = ContextualReranker()
    contextual_ranks = run_contextual_reranker(df, all_actions, reranker)

    # 3) 计算指标
    rec_dist, rec_hits = compute_metrics(recency_ranks, index_to_dist, top_k)
    ctx_dist, ctx_hits = compute_metrics(contextual_ranks, index_to_dist, top_k)

    # 4) 输出汇总
    df_out = df[['timestamp', 'action']].copy()
    df_out['recency_rank'] = recency_ranks
    df_out['recency_dist'] = rec_dist
    df_out['contextual_rank'] = contextual_ranks
    df_out['contextual_dist'] = ctx_dist
    df_out.to_csv(out_csv_path, index=False)
    print(f"保存逐事件结果至: {out_csv_path}")

    avg_rec_dist = sum(rec_dist) / len(rec_dist)
    avg_ctx_dist = sum(ctx_dist) / len(ctx_dist)
    avg_rec_hit = sum(rec_hits) / len(rec_hits)
    avg_ctx_hit = sum(ctx_hits) / len(ctx_hits)

    print(f"\n{'指标':<25} {'Recency':>10} {'Contextual':>12}")
    print("-" * 50)
    print(f"{'平均曼哈顿距离':<25} {avg_rec_dist:>10.3f} {avg_ctx_dist:>12.3f}")
    print(f"{'Top-{0} 命中率'.format(top_k):<25} {avg_rec_hit:>10.1%} {avg_ctx_hit:>12.1%}")

    # 5) 绘制对比图
    sns.set(style='whitegrid')
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    rec_rolling = pd.Series(rec_dist).rolling(window=window_size, min_periods=1).mean()
    ctx_rolling = pd.Series(ctx_dist).rolling(window=window_size, min_periods=1).mean()

    axes[0].plot(df['timestamp'], rec_rolling, label='Recency baseline', alpha=0.8)
    axes[0].plot(df['timestamp'], ctx_rolling, label='ContextualReranker', alpha=0.8)
    axes[0].set_ylabel('Manhattan distance (rolling avg)')
    axes[0].set_title(f'Manhattan Distance Comparison (window={window_size})')
    axes[0].legend()

    rec_hit_rolling = pd.Series(rec_hits).rolling(window=window_size, min_periods=1).mean()
    ctx_hit_rolling = pd.Series(ctx_hits).rolling(window=window_size, min_periods=1).mean()

    axes[1].plot(df['timestamp'], rec_hit_rolling, label='Recency baseline', alpha=0.8)
    axes[1].plot(df['timestamp'], ctx_hit_rolling, label='ContextualReranker', alpha=0.8)
    axes[1].set_ylabel(f'Top-{top_k} hit rate (rolling avg)')
    axes[1].set_xlabel('Timestamp')
    axes[1].set_title(f'Top-{top_k} Hit Rate Comparison (window={window_size})')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_png_path, dpi=200)
    plt.close()
    print(f"保存对比图至: {out_png_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="评估重排算法：ContextualReranker vs Recency baseline")
    parser.add_argument('--in', dest='in_csv', default='dataset/casas/hh108_filtered.csv', help='输入 CSV 路径')
    parser.add_argument('--out-dir', dest='out_dir', default='output', help='输出目录')
    parser.add_argument('--out-csv', dest='out_csv', default='eval_results.csv', help='输出结果 CSV 文件名')
    parser.add_argument('--out-png', dest='out_png', default='eval_comparison.png', help='输出对比图文件名')
    parser.add_argument('--window', dest='window_size', type=int, default=100, help='滑动窗口大小')
    parser.add_argument('--grid-m', type=int, default=20, help='网格行数')
    parser.add_argument('--grid-n', type=int, default=4, help='网格列数')
    parser.add_argument('--top-k', type=int, default=3, help='Top-K 命中率的 K 值')
    args = parser.parse_args()
    main(
        in_csv=args.in_csv,
        out_dir=args.out_dir,
        out_csv=args.out_csv,
        out_png=args.out_png,
        window_size=args.window_size,
        grid_m=args.grid_m,
        grid_n=args.grid_n,
        top_k=args.top_k,
    )
