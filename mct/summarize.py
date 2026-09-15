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

__all__ = ["TrackSummary", "speed_mps", "summarize_track", "summarize_camera"]


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
    entry_speed_mps: float | None = None
    exit_speed_mps: float | None = None
    speed_fallback: bool = False
    """前半・後半で区間数が足りず、トラック全体の速度で代用したか。"""

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
            "entry_speed_mps": self.entry_speed_mps,
            "exit_speed_mps": self.exit_speed_mps,
            "speed_fallback": self.speed_fallback,
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
            entry_speed_mps=d.get("entry_speed_mps"),
            exit_speed_mps=d.get("exit_speed_mps"),
            speed_fallback=bool(d.get("speed_fallback", False)),
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


def speed_mps(
    points: list[FramePoint], H: np.ndarray | None, params: dict[str, Any]
) -> float | None:
    """BEV 実空間での速度 [m/s]。求められなければ None。

    軌跡を等間隔に分割して区間ごとの速度を出し、その**中央値**を取る。
    全体の始終点だけで測ると立ち止まりや回り込みで過小評価になり、
    フレームごとに測ると検出のぶれがそのまま速度になるため。
    """
    if H is None or len(points) < 2:
        return None

    fps = float(params.get("fps", 30))
    n_seg = _n_segments(len(points), params)
    if n_seg < 1:
        return None

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

    return statistics.median(speeds) if speeds else None


def _half_speed(
    points: list[FramePoint], H: np.ndarray | None, params: dict[str, Any]
) -> tuple[float | None, float | None, bool]:
    """前半・後半それぞれの速度。区間数が足りなければ全体の値で代用する。"""
    minimum = int(params.get("speed_min_segments", 3))
    half = len(points) // 2

    # 半分にすると目標の刻みで下限の区間数を取れない場合、前半・後半それぞれの
    # 値は細切れの区間から出ることになり信用できない。全体の値で代用する。
    if half < 2 or _natural_segments(half, params) < minimum:
        whole = speed_mps(points, H, params)
        return whole, whole, True

    return speed_mps(points[:half], H, params), speed_mps(points[half:], H, params), False


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
    entry_speed, exit_speed, fallback = _half_speed(points, H, params)

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
        entry_speed_mps=entry_speed,
        exit_speed_mps=exit_speed,
        speed_fallback=fallback,
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
    return [
        summarize_track(date, camera, tid, points, cam_cfg, H, params, dir_frames)
        for tid, points in sorted(tracks.items(), key=lambda kv: int(kv[0]))
        if points
    ]


def track_direction_deg(
    points: list[FramePoint], dir_frames: int, *, entering: bool
) -> float | None:
    """`mct.diagnostics.track_direction` の再公開（呼び出し側の import を1つに）。"""
    return track_direction(points, dir_frames, entering=entering)
