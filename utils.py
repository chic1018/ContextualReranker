"""
共享工具函数。
"""

import os

import pandas as pd


def build_grid_lookup(m: int, n: int):
    """
    构建网格排序的曼哈顿距离查找表。

    网格按曼哈顿距离层排序（先 x+y，再 x），返回
    index_to_dist[rank] = manhattan_distance(rank 位设备)。
    """
    coords = [(x, y) for x in range(m) for y in range(n)]
    coords.sort(key=lambda c: (c[0] + c[1], c[0]))
    return [x + y for (x, y) in coords]


def load_data(csv_path: str, datetime_cols: tuple = ('date', 'time')) -> pd.DataFrame:
    """
    读取 CSV，合并日期时间列为 timestamp，按时间升序排序。
    """
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df[datetime_cols[0]] + ' ' + df[datetime_cols[1]])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def ensure_output_dir(path: str) -> str:
    """确保输出目录存在，返回路径。"""
    os.makedirs(path, exist_ok=True)
    return path
