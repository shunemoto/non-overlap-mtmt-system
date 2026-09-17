"""分離度を測るための小さな統計。

時空間尤度と外観類似度で同じ指標を使うので、どちらからも読めるように
切り出してある。どのスコアに対しても意味が変わらない計算だけを置く。
"""

from __future__ import annotations

import numpy as np

__all__ = ["auc", "quantiles", "format_quantiles"]

PERCENTILES = (10, 25, 50, 75, 90)


def auc(positive: list[float], negative: list[float]) -> float:
    """正解のスコアが誤ペアより高い確率。同値は0.5として数える。

    総当たりは O(n·m) になるので、順位和（Mann-Whitney U）から求める。
    どちらかが空なら nan を返す。
    """
    if not positive or not negative:
        return float("nan")

    values = np.concatenate([np.asarray(positive), np.asarray(negative)])
    order = values.argsort(kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1)

    # 同値には平均順位を割り当てる（同値を0.5として数えるのと等価）
    sorted_values = values[order]
    start = 0
    for i in range(1, len(sorted_values) + 1):
        if i == len(sorted_values) or sorted_values[i] != sorted_values[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i

    n_pos, n_neg = len(positive), len(negative)
    rank_sum = ranks[:n_pos].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def quantiles(values: list[float]) -> list[float]:
    """p10/p25/中央/p75/p90。空なら空リスト。"""
    if not values:
        return []
    a = np.asarray(values, dtype=np.float64)
    return [float(np.percentile(a, q)) for q in PERCENTILES]


def format_quantiles(values: list[float], digits: int = 3) -> str:
    """分位点を "0.100 / 0.200 / ..." の形にそろえる。"""
    qs = quantiles(values)
    if not qs:
        return "-"
    return " / ".join(f"{v:.{digits}f}" for v in qs)
