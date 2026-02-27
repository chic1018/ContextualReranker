"""
CASAS 数据集预处理脚本。

过滤逻辑：
1. 只保留包含 "begin" 的事件行
2. 提取动作名称（取 "=" 前半部分）
3. 去除连续重复动作

用法：
  python filter_casas.py
  python filter_casas.py --in dataset/casas/hh108.csv --out dataset/casas/hh108_filtered.csv
"""

import argparse
import os

import pandas as pd


def filter_casas(input_path: str, output_path: str) -> pd.DataFrame:
    """
    过滤 CASAS 原始数据，返回清洗后的 DataFrame 并保存到 output_path。
    """
    df = pd.read_csv(
        input_path,
        header=None,
        names=['date', 'time', 'location', 'status', 'action'],
    )

    # 只保留 begin 事件
    df = df[df['action'].astype(str).str.contains('begin', na=False)]

    # 提取动作名（去掉 "=begin" 后缀）
    df['action'] = df['action'].str.split('=').str[0]

    # 去除连续重复
    df = df[df['action'] != df['action'].shift()].reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"清洗完成：{len(df)} 条记录，已保存至 {output_path}")
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CASAS 数据集预处理')
    parser.add_argument(
        '--in', dest='input_path',
        default='dataset/casas/hh108.csv',
        help='原始 CASAS CSV 路径',
    )
    parser.add_argument(
        '--out', dest='output_path',
        default='dataset/casas/hh108_filtered.csv',
        help='输出过滤后的 CSV 路径',
    )
    args = parser.parse_args()
    filter_casas(args.input_path, args.output_path)
