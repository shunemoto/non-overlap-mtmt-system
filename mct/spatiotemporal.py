"""時空間尤度の算出。

「そのカメラ間を歩けばこれくらいの時間がかかるはず」という見込みと、実際の
出現時刻とのずれから尤度を出す。外観を見ずに、時刻と経路だけでどこまで
絞れるかを測るのが目的。

到着時刻の予測が外れる要因は2つある。歩く速さの個人差（これは正規分布で
モデル化できる）と、途中で立ち止まったり寄り道したりする滞留（こちらは
分布に乗らない）。探索時間の上限を超えた候補と自己ループは後者なので、
正規分布ではなく固定値 `c_dwell` を当てる。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mct.candidates import Candidate
from mct.config import TransitionRoute
from mct.summarize import TrackSummary, harmonic_mean_speed

__all__ = [
    "KIND_IN_RANGE",
    "KIND_OUT_OF_RANGE",
    "KIND_SELF_LOOP",
    "STScore",
    "SigmaTable",
    "predicted_arrival",
    "spatiotemporal_likelihood",
    "estimate_sigma",
    "score_candidates",
]

KIND_IN_RANGE = "in_range"
KIND_OUT_OF_RANGE = "out_of_range"
KIND_SELF_LOOP = "self_loop"

# MAD を標準偏差に直す係数（正規分布を仮定したときの換算）
MAD_TO_SIGMA = 1.4826


@dataclass
class STScore:
    src: tuple[str, int]
    tgt: tuple[str, int]
    kind: str
    likelihood: float
    v_harmonic: float | None = None
    predicted_arrival: float | None = None
    actual_arrival: int | None = None
    error_frames: float | None = None
    speed_imputed: bool = False
    """どちらかの速度が欠けていて中央値で代用したか。"""

    def to_dict(self) -> dict[str, Any]:
        return {
            "src": [self.src[0], self.src[1]],
            "tgt": [self.tgt[0], self.tgt[1]],
            "predicted_arrival": self.predicted_arrival,
            "actual_arrival": self.actual_arrival,
            "error_frames": self.error_frames,
            "v_harmonic": self.v_harmonic,
            "kind": self.kind,
            "likelihood": self.likelihood,
            "speed_imputed": self.speed_imputed,
        }


@dataclass
class SigmaTable:
    """ゾーン対ごとの σ。無いゾーン対には共通の値を使う。"""

    default: float
    per_zone: dict[tuple[str, str, str, str], float] = field(default_factory=dict)

    def get(self, route: TransitionRoute | None) -> float:
        if route is None:
            return self.default
        key = (route.from_cam, route.from_zone, route.to_cam, route.to_zone)
        return self.per_zone.get(key, self.default)


def predicted_arrival(
    tgt: TrackSummary,
    src: TrackSummary,
    route: TransitionRoute,
    fps: float,
    speed: float | None = None,
) -> tuple[float | None, float | None]:
    """(予測到着フレーム, 使った調和平均速度)。速度が無ければ (None, None)。

    退出した側と出現した側の速度の調和平均を使う。到着時刻は距離÷速度で
    決まるので、平均すべきは速度ではなく「1m あたりの所要時間」であり、
    その逆数が調和平均にあたる。
    """
    v = speed if speed is not None else harmonic_mean_speed(tgt.speed_mps, src.speed_mps)
    if not v or v <= 0:
        return None, None
    return tgt.end_frame + (route.distance_m / v) * fps, v


def _resolve_speeds(
    tgt: TrackSummary,
    src: TrackSummary,
    medians: dict[str, float],
    overall_median: float | None,
) -> tuple[float | None, bool]:
    """調和平均の速度。欠けていればカメラ中央値、無ければ全体の中央値で埋める。"""
    imputed = False
    values = []
    for s in (tgt, src):
        v = s.speed_mps
        if v is None:
            v = medians.get(s.camera, overall_median)
            imputed = True
        values.append(v)
    return harmonic_mean_speed(*values), imputed


def spatiotemporal_likelihood(
    candidate: Candidate,
    summaries: dict[tuple[str, int], TrackSummary],
    sigma: SigmaTable,
    params: dict[str, Any],
    medians: dict[str, float] | None = None,
    overall_median: float | None = None,
) -> STScore:
    """1候補の時空間尤度。

    探索時間範囲の中なら誤差に応じた正規分布、外や自己ループなら固定値。
    滞留の長さは歩行速度から予測できないので、分布を当てても意味がない。
    """
    fps = float(params.get("fps", 30))
    route = candidate.route
    c_dwell = route.c_dwell if route is not None else float(params.get("no_match_score", 0.1))

    if candidate.is_self_loop:
        return STScore(candidate.src, candidate.tgt, KIND_SELF_LOOP, c_dwell)
    if not candidate.in_search_range:
        return STScore(candidate.src, candidate.tgt, KIND_OUT_OF_RANGE, c_dwell)

    # 種類は候補の性質なので、速度が取れなくても in_range のまま。
    # 尤度だけ、到着時刻を予測できないときの既定値に落とす。
    tgt, src = summaries.get(candidate.tgt), summaries.get(candidate.src)
    if tgt is None or src is None or route is None:
        return STScore(candidate.src, candidate.tgt, KIND_IN_RANGE, c_dwell)

    speed, imputed = _resolve_speeds(tgt, src, medians or {}, overall_median)
    arrival, v = predicted_arrival(tgt, src, route, fps, speed)
    if arrival is None:
        return STScore(
            candidate.src, candidate.tgt, KIND_IN_RANGE, c_dwell, speed_imputed=imputed
        )

    error = arrival - src.start_frame
    s = sigma.get(route)
    likelihood = float(np.exp(-(error ** 2) / (2.0 * s * s))) if s > 0 else 0.0

    return STScore(
        src=candidate.src, tgt=candidate.tgt, kind=KIND_IN_RANGE,
        likelihood=likelihood, v_harmonic=v,
        predicted_arrival=arrival, actual_arrival=src.start_frame,
        error_frames=error, speed_imputed=imputed,
    )


def speed_medians(
    summaries: list[TrackSummary],
) -> tuple[dict[str, float], float | None]:
    """カメラごとの速度の中央値と、全体の中央値。"""
    by_camera: dict[str, list[float]] = {}
    for s in summaries:
        if s.speed_mps is not None:
            by_camera.setdefault(s.camera, []).append(s.speed_mps)

    everything = [v for values in by_camera.values() for v in values]
    return (
        {cam: statistics.median(values) for cam, values in by_camera.items()},
        statistics.median(everything) if everything else None,
    )


def score_candidates(
    candidates: list[Candidate],
    summaries: list[TrackSummary],
    sigma: SigmaTable,
    params: dict[str, Any],
) -> list[STScore]:
    by_key = {s.key: s for s in summaries}
    medians, overall = speed_medians(summaries)
    return [
        spatiotemporal_likelihood(c, by_key, sigma, params, medians, overall)
        for c in candidates
    ]


# --------------------------------------------------------------------------
# σ の推定
# --------------------------------------------------------------------------


def _mad(values: list[float]) -> float:
    """中央絶対偏差。外れ値に引きずられない散らばりの尺度。"""
    if not values:
        return 0.0
    median = statistics.median(values)
    return statistics.median([abs(v - median) for v in values])


@dataclass
class SigmaEstimate:
    from_cam: str
    from_zone: str
    to_cam: str
    to_zone: str
    errors: list[float] = field(default_factory=list)
    adopted: bool = False

    @property
    def n_samples(self) -> int:
        return len(self.errors)

    @property
    def median_e(self) -> float | None:
        return statistics.median(self.errors) if self.errors else None

    @property
    def mad_e(self) -> float | None:
        return _mad(self.errors) if self.errors else None

    @property
    def sigma(self) -> float | None:
        mad = self.mad_e
        return MAD_TO_SIGMA * mad if mad else None

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.from_cam, self.from_zone, self.to_cam, self.to_zone)


def collect_errors(
    candidates: list[Candidate],
    summaries: list[TrackSummary],
    correct_pairs: set[tuple[tuple[str, int], tuple[str, int]]],
    params: dict[str, Any],
) -> dict[tuple[str, str, str, str], SigmaEstimate]:
    """正解と分かっている候補のうち、探索時間範囲内のものの誤差を集める。

    範囲外と自己ループは固定値を当てる側なので、σ の推定には使わない。
    """
    fps = float(params.get("fps", 30))
    by_key = {s.key: s for s in summaries}
    medians, overall = speed_medians(summaries)

    out: dict[tuple[str, str, str, str], SigmaEstimate] = {}
    for c in candidates:
        if c.route is None or c.is_self_loop or not c.in_search_range:
            continue
        if (c.tgt, c.src) not in correct_pairs:
            continue
        tgt, src = by_key.get(c.tgt), by_key.get(c.src)
        if tgt is None or src is None:
            continue
        speed, _ = _resolve_speeds(tgt, src, medians, overall)
        arrival, _ = predicted_arrival(tgt, src, c.route, fps, speed)
        if arrival is None:
            continue

        key = (c.route.from_cam, c.route.from_zone, c.route.to_cam, c.route.to_zone)
        out.setdefault(key, SigmaEstimate(*key)).errors.append(arrival - src.start_frame)
    return out


def estimate_sigma(
    per_date: list[dict[tuple[str, str, str, str], SigmaEstimate]],
    params: dict[str, Any],
) -> tuple[SigmaTable, dict[tuple[str, str, str, str], SigmaEstimate], float]:
    """ゾーン対ごとの σ と、共通の σ を推定する。

    標準偏差ではなく MAD から出す。滞留したサンプルが数件混ざるだけで
    標準偏差は大きく膨らむが、MAD はほとんど動かない。
    """
    min_samples = int(params.get("sigma_min_samples", 5))
    default = float(params.get("sigma_default_frames", 60))

    merged: dict[tuple[str, str, str, str], SigmaEstimate] = {}
    for samples in per_date:
        for key, estimate in samples.items():
            merged.setdefault(key, SigmaEstimate(*key)).errors.extend(estimate.errors)

    pooled = [v for e in merged.values() for v in e.errors]
    pooled_sigma = MAD_TO_SIGMA * _mad(pooled) if pooled else default
    if pooled_sigma <= 0:
        pooled_sigma = default

    table = SigmaTable(default=pooled_sigma)
    for key, estimate in merged.items():
        estimate.adopted = estimate.n_samples >= min_samples and bool(estimate.sigma)
        if estimate.adopted:
            table.per_zone[key] = estimate.sigma          # type: ignore[assignment]
    return table, merged, pooled_sigma
