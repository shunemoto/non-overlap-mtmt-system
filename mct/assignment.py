"""対応付けを割当問題として解き、Global ID を付ける。

各トラックを「出口」と「入口」の2ノードに分け、対応仮説を出口→入口の枝と
して表す。制約は

- 各トラックの遷移先は高々1つ（出口から出る枝は高々1本）
- 各トラックの遷移元は高々1つ（入口に入る枝は高々1本）

の2つで、これは割当問題そのものになる。貪欲に高いスコアから取るのとは違い、
**片方から見て最良の相手が相手から見ても最良とは限らない**場合に、全体の
合計が最大になる組を選べる。1人が複数のトラックの遷移先に同時に選ばれる、
といった矛盾も構造的に起きない。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from mct.fusion import EPS
from mct.predictions import Link

__all__ = ["FORBIDDEN", "UnionFind", "build_cost_matrix", "solve", "assign_global_ids"]

# 選ばせたくないセルのコスト。無限大はハンガリー法に渡せないので大きな有限値。
FORBIDDEN = 1e6

TrackKey = tuple[str, int]


class UnionFind:
    """連結成分を求めるための素集合データ構造。

    採用された対応は隣り合うトラック間のものでしかないので、C→F→G のように
    3台にまたがる人物を1つの ID にまとめるには連結成分を取る必要がある。
    """

    def __init__(self) -> None:
        self._parent: dict[Any, Any] = {}

    def find(self, x: Any) -> Any:
        self._parent.setdefault(x, x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:          # 経路圧縮
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: Any, b: Any) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra

    def groups(self) -> dict[Any, list[Any]]:
        out: dict[Any, list[Any]] = {}
        for x in self._parent:
            out.setdefault(self.find(x), []).append(x)
        return out


def build_cost_matrix(
    tracks: list[TrackKey],
    scored: dict[tuple[TrackKey, TrackKey], float],
    no_match_score: float,
) -> np.ndarray:
    """`n × 2n` のコスト行列。

    行は各トラックの出口、列の前半 `n` は各トラックの入口、後半 `n` は
    **行ごとに専用の**「対応なし」ダミー列。ダミーを1本にすると、対応なしに
    できるトラックが1つに限られてしまう。

    `scored` のキーは (tgt, src) = (先に退出する側, 後に出現する側)。
    """
    n = len(tracks)
    index = {key: i for i, key in enumerate(tracks)}
    cost = np.full((n, 2 * n), FORBIDDEN, dtype=np.float64)

    # 行 i の専用ダミー列は n+i だけ
    cost[np.arange(n), np.arange(n) + n] = -math.log(max(EPS, no_match_score))

    for (tgt, src), score in scored.items():
        i, j = index.get(tgt), index.get(src)
        if i is None or j is None or i == j:
            continue
        cost[i, j] = -math.log(max(EPS, score))
    return cost


def solve(
    tracks: list[TrackKey],
    scored: dict[tuple[TrackKey, TrackKey], float],
    no_match_score: float,
) -> list[tuple[TrackKey, TrackKey]]:
    """採用された (tgt, src) の一覧。"""
    if not tracks:
        return []

    cost = build_cost_matrix(tracks, scored, no_match_score)
    rows, cols = linear_sum_assignment(cost)

    n = len(tracks)
    chosen = []
    for i, j in zip(rows, cols):
        if j >= n:
            continue                            # ダミー列 = 対応なし
        if cost[i, j] >= FORBIDDEN:
            continue                            # 候補に無い対。通常は選ばれない
        chosen.append((tracks[i], tracks[j]))
    return chosen


def assign_global_ids(
    links: list[Link], tracks: list[TrackKey], prefix: str = "G"
) -> dict[str, dict[str, str]]:
    """連結成分ごとに Global ID を振る。

    対応が付かなかったトラックにも単独の ID を与える。評価はトラック単位で
    行うので、ID が無いトラックは必ず誤りに数えられてしまい、「1台しか
    写らなかった人物」を正しく扱えない。
    """
    uf = UnionFind()
    for key in tracks:
        uf.find(key)
    for link in links:
        uf.union((link.from_cam, int(link.from_track)), (link.to_cam, int(link.to_track)))

    assignments: dict[str, dict[str, str]] = {}
    # 出現順を安定させるため、カメラ名と track_id で並べた代表元の順に振る
    for n, root in enumerate(
        sorted(uf.groups(), key=lambda r: min(uf.groups()[r])), start=1
    ):
        global_id = f"{prefix}{n:03d}"
        for camera, track_id in uf.groups()[root]:
            assignments.setdefault(camera, {})[str(track_id)] = global_id
    return assignments
