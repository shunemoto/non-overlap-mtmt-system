"""カメラ単体の追跡結果を表すデータ構造。

段階2以降（カメラ間再同定）がこの形式を入力として読むため、ここが
システム全体のデータ契約になる。標準ライブラリのみに依存する。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["FramePoint", "TrackMeta", "TrackingResult"]


@dataclass
class FramePoint:
    """1トラックの1フレーム分の観測。"""

    frame_idx: int
    """1-based のフレーム番号（MOT 形式に合わせる）。"""

    bbox: list[float]
    """[x, y, w, h]。左上原点。"""

    conf: float

    foot_point: list[float]
    """[x, y]。bbox 下辺中央。"""

    saved: bool
    """crop / mask 画像を保存したフレームかどうか。"""

    keypoints: list[list[float]] | None = None
    """COCO 17点 [[x, y, conf], ...]。saved=True のフレームのみ格納し、
    saved=False のときは None。"""

    def __post_init__(self) -> None:
        # saved=False のフレームにキーポイントが混ざると、段階2側が
        # 「キーポイントがある = crop 画像もある」と仮定できなくなる。
        if not self.saved and self.keypoints is not None:
            raise ValueError("keypoints must be None when saved=False")

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "bbox": list(self.bbox),
            "conf": self.conf,
            "foot_point": list(self.foot_point),
            "saved": self.saved,
            "keypoints": (
                None if self.keypoints is None else [list(kp) for kp in self.keypoints]
            ),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FramePoint:
        kps = d.get("keypoints")
        return cls(
            frame_idx=int(d["frame_idx"]),
            bbox=[float(v) for v in d["bbox"]],
            conf=float(d["conf"]),
            foot_point=[float(v) for v in d["foot_point"]],
            saved=bool(d["saved"]),
            keypoints=(None if kps is None else [[float(v) for v in kp] for kp in kps]),
        )


@dataclass
class TrackMeta:
    """追跡を実行した際の条件。再現性のために結果と一緒に保存する。"""

    date: str
    """撮影日。"0720" のような4桁。"""

    camera: str
    """カメラID。"camA" など。"""

    width: int
    height: int
    fps: float
    n_frames: int
    model_pose: str
    model_seg: str
    tracker: str
    n_saved_per_track: int
    created_at: str
    """ISO8601 形式のタイムスタンプ。"""

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "camera": self.camera,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "n_frames": self.n_frames,
            "model_pose": self.model_pose,
            "model_seg": self.model_seg,
            "tracker": self.tracker,
            "n_saved_per_track": self.n_saved_per_track,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrackMeta:
        return cls(
            date=str(d["date"]),
            camera=str(d["camera"]),
            width=int(d["width"]),
            height=int(d["height"]),
            fps=float(d["fps"]),
            n_frames=int(d["n_frames"]),
            model_pose=str(d["model_pose"]),
            model_seg=str(d["model_seg"]),
            tracker=str(d["tracker"]),
            n_saved_per_track=int(d["n_saved_per_track"]),
            created_at=str(d["created_at"]),
        )


@dataclass
class TrackingResult:
    """カメラ1台・1日分の追跡結果全体。"""

    meta: TrackMeta

    tracks: dict[str, list[FramePoint]] = field(default_factory=dict)
    """key = local_track_id（文字列）。値はフレーム順の観測列。"""

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": self.meta.to_dict(),
            "tracks": {
                tid: [p.to_dict() for p in points] for tid, points in self.tracks.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrackingResult:
        return cls(
            meta=TrackMeta.from_dict(d["meta"]),
            tracks={
                str(tid): [FramePoint.from_dict(p) for p in points]
                for tid, points in d.get("tracks", {}).items()
            },
        )
