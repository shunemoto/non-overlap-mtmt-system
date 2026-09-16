"""候補列挙に使うトラックの要約。

1トラックを「いつ・どこから入って、どこへ・どの向きに・どれくらいの速さで
出ていったか」に畳む。候補列挙も後段の再同定も、軌跡そのものではなく
この要約だけを見れば済むようにする。

ゾーン判定と進行方向はフェーズ2の実装をそのまま使う。同じ判定が2箇所に
生まれると、診断で見えているものと候補列挙が食い違う。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mct.config import CameraConfig
from mct.diagnostics import match_zones, track_direction
from mct.geometry import to_world
from mct.schema import FramePoint

__all__ = [
    "TrackSummary",
    "segment_speeds",
    "track_speed",
    "harmonic_mean_speed",
    "summarize_track",
    "summarize_camera",
]


@dataclass
class TrackSummary:
    date: str
    camera: str
    track_id: int
    start_frame: int
    end_frame: int
    n_points: int
    entry_point: list[float]
    exit_point: list[float]
    entry_dir_deg: float | None = None
    exit_dir_deg: float | None = None
    entry_zones: list[str] = field(default_factory=list)
    exit_zones: list[str] = field(default_factory=list)

    speed_mps: float | None = None
    """トラック全体から求めた歩行速度。

    出現直後と退出直前で分けることも試したが、トラック長の中央値が
    128フレームしかなく、1.5秒×3区間を半分だけで確保できるトラックが
    3%しか無かった。区間を短くすると1区間あたりの誤差が増える
    （0.5秒で25%、1.5秒で8.3%）ので、分離より精度を取って1つにまとめた。
    """

    speed_fallback: bool = False
    """区間がすべて停止とみなされ、同じカメラ・同じ日の中央値で代用したか。"""

    speed_clipped: bool = False
    """妥当範囲を外れてクリップされたか。ホモグラフィの外挿誤差の目印。"""

    @property
    def key(self) -> tuple[str, int]:
        return (self.camera, self.track_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "camera": self.camera,
            "track_id": self.track_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "n_points": self.n_points,
            "entry_point": list(self.entry_point),
            "exit_point": list(self.exit_point),
            "entry_dir_deg": self.entry_dir_deg,
            "exit_dir_deg": self.exit_dir_deg,
            "entry_zones": list(self.entry_zones),
            "exit_zones": list(self.exit_zones),
            "speed_mps": self.speed_mps,
            "speed_fallback": self.speed_fallback,
            "speed_clipped": self.speed_clipped,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrackSummary:
        return cls(
            date=str(d["date"]),
            camera=str(d["camera"]),
            track_id=int(d["track_id"]),
            start_frame=int(d["start_frame"]),
            end_frame=int(d["end_frame"]),
            n_points=int(d["n_points"]),
            entry_point=[float(v) for v in d["entry_point"]],
            exit_point=[float(v) for v in d["exit_point"]],
            entry_dir_deg=d.get("entry_dir_deg"),
            exit_dir_deg=d.get("exit_dir_deg"),
            entry_zones=list(d.get("entry_zones") or []),
            exit_zones=list(d.get("exit_zones") or []),
            speed_mps=d.get("speed_mps"),
            speed_fallback=bool(d.get("speed_fallback", False)),
            speed_clipped=bool(d.get("speed_clipped", False)),
        )


def _natural_segments(n_frames: int, params: dict[str, Any]) -> int:
    """目標の刻み（既定1.5秒）で素直に割ったときの区間数。

    下限を当てる前の値。これが下限を割るなら、区間が目標より細切れになり
    速度の推定が検出のぶれに引きずられる。
    """
    fps = float(params.get("fps", 30))
    target = float(params.get("speed_target_sec", 1.5))
    return int(n_frames // max(1.0, target * fps))


def _n_segments(n_frames: int, params: dict[str, Any]) -> int:
    """実際に使う区間分割数。目標の刻みで割りつつ、下限を割らないようにする。"""
    minimum = int(params.get("speed_min_segments", 3))
    return max(minimum, _natural_segments(n_frames, params))


def segment_speeds(
    points: list[FramePoint], H: np.ndarray | None, params: dict[str, Any]
) -> list[float]:
    """軌跡を等間隔に分けた各区間の速度 [m/s]。

    全体の始終点だけで測ると回り込みで過小評価になり、フレームごとに測ると
    検出のぶれがそのまま速度になるので、目標の刻みで区切って測る。
    """
    if H is None or len(points) < 2:
        return []

    fps = float(params.get("fps", 30))
    n_seg = _n_segments(len(points), params)
    if n_seg < 1:
        return []

    world = to_world([p.foot_point for p in points], H)
    frames = [p.frame_idx for p in points]
    bounds = np.linspace(0, len(points) - 1, n_seg + 1).round().astype(int)

    speeds = []
    for a, b in zip(bounds, bounds[1:]):
        if b <= a:
            continue
        seconds = (frames[b] - frames[a]) / fps
        if seconds <= 0:
            continue
        speeds.append(float(np.linalg.norm(world[b] - world[a])) / seconds)
    return speeds


def track_speed(
    points: list[FramePoint], H: np.ndarray | None, params: dict[str, Any]
) -> tuple[float | None, bool]:
    """トラック全体の歩行速度 [m/s] と、クリップしたかどうか。

    シナリオに停止・滞留を含むため、止まっている区間を混ぜた中央値は
    「その人の歩行速度」にならない。`speed_stop_threshold` 未満の区間を
    落としてから中央値を取る。全区間が落ちたら None を返し、呼び出し側が
    同じカメラ・同じ日の中央値で代用する。

    最後に妥当範囲でクリップする。ホモグラフィの外挿が効く画面端では
    3.5 m/s のような歩行では有り得ない値が出るため。
    """
    stop_threshold = float(params.get("speed_stop_threshold", 0.3))
    lo = float(params.get("speed_min_mps", 0.6))
    hi = float(params.get("speed_max_mps", 2.0))

    moving = [s for s in segment_speeds(points, H, params) if s >= stop_threshold]
    if not moving:
        return None, False

    value = statistics.median(moving)
    clipped = min(max(value, lo), hi)
    return clipped, clipped != value


def harmonic_mean_speed(a: float | None, b: float | None) -> float | None:
    """2トラックの速度の調和平均。

    到着時刻は距離÷速度で決まるので、平均すべきは速度ではなく「1mあたりの
    所要時間」。その逆数が調和平均にあたる。片方でも欠ければ None。
    """
    if not a or not b or a <= 0 or b <= 0:
        return None
    return 2.0 * a * b / (a + b)


def summarize_track(
    date: str,
    camera: str,
    track_id: str | int,
    points: list[FramePoint],
    cam_cfg: CameraConfig,
    H: np.ndarray | None,
    params: dict[str, Any],
    dir_frames: int,
) -> TrackSummary:
    entry = match_zones(points, cam_cfg, dir_frames, entering=True)
    exit_ = match_zones(points, cam_cfg, dir_frames, entering=False)
    speed, clipped = track_speed(points, H, params)

    return TrackSummary(
        date=date,
        camera=camera,
        track_id=int(track_id),
        start_frame=points[0].frame_idx,
        end_frame=points[-1].frame_idx,
        n_points=len(points),
        entry_point=list(points[0].foot_point),
        exit_point=list(points[-1].foot_point),
        entry_dir_deg=entry.direction,
        exit_dir_deg=exit_.direction,
        entry_zones=list(entry.zones),
        exit_zones=list(exit_.zones),
        speed_mps=speed,
        speed_clipped=clipped,
    )


def summarize_camera(
    date: str,
    camera: str,
    tracks: dict[str, list[FramePoint]],
    cam_cfg: CameraConfig,
    H: np.ndarray | None,
    params: dict[str, Any],
    dir_frames: int,
) -> list[TrackSummary]:
    """1カメラ分を要約する。

    ずっと止まっていて速度を出せなかったトラックには、同じカメラ・同じ日の
    中央値を当てる。そのカメラを通る人の典型的な速さが、そのトラックにとっても
    いちばんましな推定値になる。
    """
    summaries = [
        summarize_track(date, camera, tid, points, cam_cfg, H, params, dir_frames)
        for tid, points in sorted(tracks.items(), key=lambda kv: int(kv[0]))
        if points
    ]

    known = [s.speed_mps for s in summaries if s.speed_mps is not None]
    if known:
        median = statistics.median(known)
        for s in summaries:
            if s.speed_mps is None:
                s.speed_mps = median
                s.speed_fallback = True
    return summaries


def track_direction_deg(
    points: list[FramePoint], dir_frames: int, *, entering: bool
) -> float | None:
    """`mct.diagnostics.track_direction` の再公開（呼び出し側の import を1つに）。"""
    return track_direction(points, dir_frames, entering=entering)
