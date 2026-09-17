"""候補ペアの外観類似度。

入力は `mct.appearance` が作ったトラック単位の記述子だけで、画像は読まない。
特徴の抽出と類似度の計算を分けてあるのは、パラメータを振ったときに再計算
する範囲を最小にするためである。

色の比較には **Bhattacharyya 距離**を使い、相関係数は使わない。相関は
ヒストグラムの平均を引いてから測るので、「どの色がどれだけ出たか」より
「ビン間の凹凸の形」を見てしまい、全ペアのスコアが狭い範囲に潰れる。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mct.appearance import N_JOINTS, AppearanceDescriptor
from mct.candidates import Candidate

__all__ = [
    "SCORES_FILENAME",
    "AppScore",
    "bhattacharyya",
    "color_similarity",
    "sift_similarity",
    "combine",
    "score_pair",
    "score_pairs",
    "write_scores",
    "load_scores",
]

SCORES_FILENAME = "app_scores.json"

MISSING_SIFT = "sift"
MISSING_COLOR = "color"
MISSING_BOTH = "both"

MIN_JOINTS = 2


@dataclass
class AppScore:
    """候補ペア1件ぶんの外観類似度。"""

    src: tuple[str, int]
    tgt: tuple[str, int]
    s_color: float | None = None
    s_color_upper: float | None = None
    s_color_lower: float | None = None
    s_sift: float | None = None
    n_joints: int = 0
    s_app: float | None = None
    missing: str | None = None

    def to_dict(self) -> dict[str, Any]:
        def r(v):
            return None if v is None else round(float(v), 6)

        return {
            "src": [self.src[0], self.src[1]],
            "tgt": [self.tgt[0], self.tgt[1]],
            "s_color": r(self.s_color),
            "s_color_upper": r(self.s_color_upper),
            "s_color_lower": r(self.s_color_lower),
            "s_sift": r(self.s_sift),
            "n_joints": self.n_joints,
            "s_app": r(self.s_app),
            "missing": self.missing,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AppScore:
        return cls(
            src=(d["src"][0], int(d["src"][1])),
            tgt=(d["tgt"][0], int(d["tgt"][1])),
            s_color=d.get("s_color"),
            s_color_upper=d.get("s_color_upper"),
            s_color_lower=d.get("s_color_lower"),
            s_sift=d.get("s_sift"),
            n_joints=int(d.get("n_joints", 0)),
            s_app=d.get("s_app"),
            missing=d.get("missing"),
        )


# --------------------------------------------------------------------------
# 色
# --------------------------------------------------------------------------


def bhattacharyya(a: list[float] | None, b: list[float] | None) -> float | None:
    """2つの L1 正規化済みヒストグラムの Bhattacharyya 距離（0〜1）。

    `cv2.compareHist` と同じ式を numpy で書く。cv2 に渡すには float32 に
    直して形をそろえる必要があり、ペアの数だけ変換を挟むと無駄が大きい。
    """
    if a is None or b is None:
        return None
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    mean_x, mean_y = x.mean(), y.mean()
    if mean_x <= 0 or mean_y <= 0:
        return None
    coefficient = float(np.sqrt(x * y).sum()) / np.sqrt(mean_x * mean_y * len(x) * len(x))
    return float(np.sqrt(max(0.0, 1.0 - min(1.0, coefficient))))


def color_similarity(
    a: AppearanceDescriptor, b: AppearanceDescriptor, lambda_missing: float
) -> tuple[float | None, float | None, float | None]:
    """(統合, 上半身, 下半身)。両方欠ければ (None, None, None)。

    上下どちらかしか無い場合は、残った方に `lambda_missing` を掛ける。
    根拠が半分しかない候補を、上下そろった候補と同じ尺度で並べないため。
    """
    d_upper = bhattacharyya(a.hist_upper, b.hist_upper)
    d_lower = bhattacharyya(a.hist_lower, b.hist_lower)

    s_upper = None if d_upper is None else 1.0 - d_upper
    s_lower = None if d_lower is None else 1.0 - d_lower

    if s_upper is not None and s_lower is not None:
        return float(np.sqrt(max(0.0, s_upper) * max(0.0, s_lower))), s_upper, s_lower
    if s_upper is not None:
        return s_upper * lambda_missing, s_upper, None
    if s_lower is not None:
        return s_lower * lambda_missing, None, s_lower
    return None, None, None


# --------------------------------------------------------------------------
# SIFT
# --------------------------------------------------------------------------


def sift_similarity(
    a: AppearanceDescriptor, b: AppearanceDescriptor, joints: list[int] | None = None
) -> tuple[float | None, int]:
    """(平均コサイン類似度, 使った関節点の数)。

    記述子は L2 正規化済みなので内積がそのままコサインになる。両側で有効な
    関節点が `MIN_JOINTS` に満たなければ欠損とする。1点だけの一致は
    たまたま似た模様を拾っただけのことが多い。
    """
    if a.sift is None or b.sift is None:
        return None, 0

    indices = range(N_JOINTS) if joints is None else joints
    values = []
    for j in indices:
        if j >= len(a.sift_valid) or j >= len(b.sift_valid):
            continue
        if not (a.sift_valid[j] and b.sift_valid[j]):
            continue
        values.append(float(np.dot(a.sift[j], b.sift[j])))

    if len(values) < MIN_JOINTS:
        return None, len(values)
    return float(np.mean(values)), len(values)


# --------------------------------------------------------------------------
# 統合
# --------------------------------------------------------------------------


def combine(
    s_color: float | None, s_sift: float | None, lambda_missing: float
) -> tuple[float | None, str | None]:
    """(統合スコア, 欠損の種類)。

    両方そろっていれば幾何平均。片方だけなら `lambda_missing` を掛けて、
    根拠が1本しかない候補が同じ土俵に並ばないようにする。
    """
    if s_color is not None and s_sift is not None:
        return float(np.sqrt(max(0.0, s_color) * max(0.0, s_sift))), None
    if s_color is not None:
        return s_color * lambda_missing, MISSING_SIFT
    if s_sift is not None:
        return s_sift * lambda_missing, MISSING_COLOR
    return None, MISSING_BOTH


def score_pair(
    src: AppearanceDescriptor | None,
    tgt: AppearanceDescriptor | None,
    params: dict[str, Any],
) -> AppScore | None:
    """記述子が片方でも無ければ None。"""
    if src is None or tgt is None:
        return None

    lambda_missing = float(params["lambda_missing"])
    s_color, s_upper, s_lower = color_similarity(src, tgt, lambda_missing)
    s_sift, n_joints = sift_similarity(src, tgt)
    s_app, missing = combine(s_color, s_sift, lambda_missing)

    return AppScore(
        src=src.key, tgt=tgt.key,
        s_color=s_color, s_color_upper=s_upper, s_color_lower=s_lower,
        s_sift=s_sift, n_joints=n_joints, s_app=s_app, missing=missing,
    )


def score_pairs(
    candidates: list[Candidate],
    descriptors: dict[tuple[str, int], AppearanceDescriptor],
    params: dict[str, Any],
) -> list[AppScore | None]:
    """候補と同じ順序・同じ長さのリストを返す。

    記述子が無い候補には None を置いて長さをそろえる。呼び出し側が
    `zip(candidates, scores)` で対応付けられるようにするため。
    """
    return [
        score_pair(descriptors.get(c.src), descriptors.get(c.tgt), params)
        for c in candidates
    ]


# --------------------------------------------------------------------------
# 入出力
# --------------------------------------------------------------------------


def write_scores(
    date: str,
    out_root: str | Path,
    scores: list[AppScore | None],
    params: dict[str, Any],
) -> Path:
    from mct.cache import cache_key

    dest = Path(out_root) / date / SCORES_FILENAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    present = [s for s in scores if s is not None]
    payload = {
        "meta": {
            "date": date,
            "n_candidates": len(present),
            "n_skipped": len(scores) - len(present),
            "config_hash": cache_key({"appearance": dict(sorted(params.items()))}),
        },
        "scores": [s.to_dict() for s in present],
    }
    with dest.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    return dest


def load_scores(path: str | Path) -> list[AppScore]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return [AppScore.from_dict(d) for d in payload.get("scores", [])]
