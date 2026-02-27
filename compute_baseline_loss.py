import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def build_grid_lookup(m, n):
    """
    构建 index -> (x,y) 以及 index -> manhattan_distance 的查找表
    网格大小: m 行, n 列
    """
    coords = []

    for x in range(m):
        for y in range(n):
            coords.append((x, y))

    # 按曼哈顿层排序：先 x+y，再 x
    coords.sort(key=lambda c: (c[0] + c[1], c[0]))

    index_to_xy = coords
    index_to_dist = [x + y for (x, y) in coords]

    return index_to_xy, index_to_dist


def compute_recency_rank_losses(df, all_actions, timestamp_col='timestamp'):
    """
    遍历事件序列：在更新 last_time 之前计算当前 action 的 rank(0-based)。
    all_actions: list of all possible actions (strings)
    返回 losses list（与 df 行一一对应）
    """
    # 使用浮点秒数作为时间比较基准；未见过初始化为非常小的值 -> 排在最后
    last_ts = {a: -1e18 for a in all_actions}
    losses = []
    index_to_xy, index_to_dist = build_grid_lookup(20, 4)

    # 逐行遍历（假设 df 已按timestamp排序）
    for _, row in df.iterrows():
        ts = row[timestamp_col].timestamp()  # float seconds
        action = row['action']

        # 按 (最近时间 desc, action name asc) 排序：
        # 我们用 key = (-last_ts, action) 并按升序排序，达到 desired order
        ordered = sorted(all_actions, key=lambda a: (-last_ts[a], a))

        # 计算 rank（0-based）
        rank = ordered.index(action)
        losses.append(index_to_dist[rank])

        # 更新该 action 的最近执行时间（在计算之后）
        last_ts[action] = ts

    return losses


def main(in_csv='data.csv', out_csv='losses.csv', out_png='loss_curve.png',
         window_size=100, datetime_cols=('date', 'time')):
    # 1) 读取数据
    df = pd.read_csv(in_csv)
    # 合并 date + time 列为 timestamp
    df['timestamp'] = pd.to_datetime(df[datetime_cols[0]] + ' ' + df[datetime_cols[1]])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 2) 所有 action 列表（固定池），按字典序给出基线顺序（用于稳定性）
    all_actions = sorted(df['action'].unique())

    # 3) 计算 loss 列表（按规则：在更新之前计算）
    df['loss'] = compute_recency_rank_losses(df, all_actions, timestamp_col='timestamp')

    # 4) 保存 losses 到 CSV
    df_out = df[['timestamp', 'action', 'loss']].copy()
    df_out.to_csv(out_csv, index=False)
    print(f"Saved per-event losses to: {out_csv}")

    # 5) 计算滑动窗口平均（平滑曲线）
    loss_series = df['loss'].astype(float)
    rolling_mean = loss_series.rolling(window=window_size, min_periods=1).mean()

    # 6) 绘图（瞬时散点 + 滑动平均线）
    sns.set(style='whitegrid')  # 美观风格
    plt.figure(figsize=(14, 6))
    # 散点：瞬时 loss，轻透明
    plt.scatter(df['timestamp'], df['loss'], s=10, alpha=0.25, label='instant loss')
    # 平滑线：滑动平均
    plt.plot(df['timestamp'], rolling_mean, linewidth=2.0, label=f'rolling mean (window={window_size})')
    plt.xlabel('timestamp')
    plt.ylabel('loss (0-based rank dist)')
    plt.title('Recency-based baseline: instant loss (dots) and rolling average (line)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print(f"Saved loss curve to: {out_png}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute recency-based baseline losses for action-card recommendation")
    parser.add_argument('--in', dest='in_csv', default='data.csv',
                        help='input csv path (with date,time,action columns)')
    parser.add_argument('--out-csv', dest='out_csv', default='losses.csv', help='output csv for per-event losses')
    parser.add_argument('--out-png', dest='out_png', default='loss_curve.png', help='output PNG path for loss curve')
    parser.add_argument('--window', dest='window_size', type=int, default=100, help='rolling window size for smoothing')
    args = parser.parse_args()
    main(in_csv=args.in_csv, out_csv=args.out_csv, out_png=args.out_png, window_size=args.window_size)
