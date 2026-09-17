"""時空間尤度と外観類似度の融合、および比較手法の定義。

対応仮説1つのスコアを重み付き幾何平均で出す。

    S = S_st^α × S_app^(1-α)

積ではなく幾何平均にするのは、片方が極端に低いときに全体を引きずり下ろす
性質を残しつつ、重みで効き具合を変えられるようにするためである。

**自己ループでは α を別にする（`alpha_self`）。** 自己ループの時空間尤度は
経路ごとの定数 `c_dwell` で埋まっていて順位付けの機能を持たない（AUC 0.299）
のに対し、外観は同一カメラ内で照明も画角もそろうため AUC 0.973 と高い。
同じ重みで混ぜると、順位を決められない側が外観の判断を薄めてしまう。

比較手法の切り替えも α の差し替えだけで表す。式を手法ごとに分けると、
どこが効いているのかが式の違いに紛れる。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = [
    "EPS",
    "Method",
    "METHODS",
    "method_alpha",
    "app_value",
    "fuse",
]

# log(0) を避けるための下限。ハンガリー法に無限大のコストは渡せない。
EPS = 1e-12


@dataclass(frozen=True)
class Method:
    """論文の比較手法1つ分。"""

    number: int
    topology: bool
    st: bool
    color: bool
    sift: bool

    @property
    def label(self) -> str:
        parts = []
        parts.append("トポロジー" if self.topology else "全対")
        if self.st:
            parts.append("時空間")
        if self.color and self.sift:
            parts.append("色+SIFT")
        elif self.color:
            parts.append("色")
        elif self.sift:
            parts.append("SIFT")
        return " + ".join(parts)

    @property
    def uses_appearance(self) -> bool:
        return self.color or self.sift


METHODS: dict[int, Method] = {
    1: Method(1, topology=False, st=False, color=True, sift=True),
    2: Method(2, topology=True, st=False, color=True, sift=True),
    3: Method(3, topology=True, st=True, color=False, sift=False),
    4: Method(4, topology=True, st=True, color=True, sift=False),
    5: Method(5, topology=True, st=True, color=False, sift=True),
    6: Method(6, topology=True, st=True, color=True, sift=True),
}


def method_alpha(method: Method, *, is_self_loop: bool, params: dict[str, Any]) -> float:
    """その手法・その候補で使う α。

    手法が片方しか使わないなら 0 か 1 に振り切る。両方使う手法でのみ
    `alpha` / `alpha_self` が効く。
    """
    if not method.uses_appearance:
        return 1.0
    if not method.st:
        return 0.0
    key = "alpha_self" if is_self_loop else "alpha"
    return float(params[key])


def app_value(method: Method, score: Any) -> float | None:
    """その手法が外観として読む値。

    手法④は色だけ、⑤は SIFT だけを見る。`app_scores.json` に別々の列として
    入っているので、外観の計算をやり直す必要はない。
    """
    if score is None or not method.uses_appearance:
        return None
    if method.color and method.sift:
        return score.s_app
    if method.color:
        return score.s_color
    return score.s_sift


def fuse(
    s_st: float | None,
    s_app: float | None,
    *,
    is_self_loop: bool,
    method: Method,
    params: dict[str, Any],
) -> tuple[float, float] | None:
    """(融合スコア, 実際に使った α)。採点できなければ None。

    外観が欠損していれば α=1 に落として時空間だけで評価する。ただし手法が
    そもそも時空間を使わない（α=0）場合は、根拠が何も残らないので候補ごと
    落とす。無理に 0 を入れると「必ず未対応」という判断を暗黙に下すことに
    なり、候補が残っているのか落ちたのかを後から区別できなくなる。
    """
    alpha = method_alpha(method, is_self_loop=is_self_loop, params=params)

    if s_app is None:
        if alpha <= 0.0:
            return None
        alpha = 1.0
    if s_st is None:
        if alpha >= 1.0:
            return None
        alpha = 0.0

    log_st = math.log(max(EPS, s_st)) if alpha > 0.0 else 0.0
    log_app = math.log(max(EPS, s_app)) if alpha < 1.0 else 0.0
    return math.exp(alpha * log_st + (1.0 - alpha) * log_app), alpha
