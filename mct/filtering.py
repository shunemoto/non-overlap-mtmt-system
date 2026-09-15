"""観測区間の適用と静止誤検出の除去。

フェーズ1が出す `tracks.json` には、観測区間の外側の点も、背景の静止物を
人物と誤検出したトラックも混ざっている。このまま再同定に渡すと存在しない
人物を他カメラと突き合わせることになるので、ここで落とす。

crop / mask の画像ファイルは消さない。設定を変えて回し直せば選び直せる
ようにしておくためで、`saved=False` にした点の画像が残っていても害はない。
"""

from __future__ import annotations

import shutil
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from mct.config import CameraConfig
from mct.geometry import point_in_regions, to_world
from mct.io_utils import crop_path
from mct.schema import FramePoint, TrackingResult
from mct.select import pose_score, select_frames

__all__ = [
    "apply_observation_region",
    "remove_static_detections",
    "copy_removed_crops",
]


def _rebuild_point(point: FramePoint, *, saved: bool) -> FramePoint:
    """`saved` を変えた FramePoint を作り直す。

    `FramePoint.__post_init__` が「saved=False なら keypoints も crop_box も
    None」を検査するので、属性を書き換えるのではなく作り直す。
    """
    if saved:
        return point
    return replace(point, saved=False, keypoints=None, crop_box=None)


def apply_observation_region(
    result: TrackingResult, cam_cfg: CameraConfig, params: dict[str, Any]
) -> tuple[TrackingResult, dict[str, Any]]:
    """観測区間の外側の点を落とし、短くなったトラックを捨てる。

    戻り値の2つ目は診断用の内訳。
    """
    excludes = cam_cfg.exclude_regions
    min_len = int(params["min_track_len"])
    n_target = int(params["n_frames_per_track"])

    kept: dict[str, list[FramePoint]] = {}
    dropped_empty: list[str] = []      # 観測区間で全点が消えた
    dropped_short: list[str] = []      # 残ったが min_track_len 未満
    n_points_before = 0
    n_points_removed = 0

    for tid, points in result.tracks.items():
        n_points_before += len(points)
        inside = [p for p in points if not point_in_regions(p.foot_point, excludes)]
        n_points_removed += len(points) - len(inside)

        if not inside:
            dropped_empty.append(tid)
            continue
        if len(inside) < min_len:
            dropped_short.append(tid)
            continue

        kept[tid] = _limit_saved(inside, n_target, int(params["n_bins"]))

    filtered = TrackingResult(meta=result.meta, tracks=kept)
    stats = {
        "n_tracks_before": len(result.tracks),
        "n_tracks_after": len(kept),
        "dropped_empty": dropped_empty,
        "dropped_short": dropped_short,
        "n_points_before": n_points_before,
        "n_points_removed": n_points_removed,
        "n_under_target_saved": sum(
            1 for ps in kept.values() if sum(1 for p in ps if p.saved) < n_target
        ),
    }
    return filtered, stats


def _limit_saved(points: list[FramePoint], n_target: int, n_bins: int) -> list[FramePoint]:
    """観測区間内に残った saved 点を n_target 枚に絞る。

    観測区間の外側を落とすと、フェーズ1で選んだ90枚のうち区間内に残るものが
    偏ることがある。区間内の点だけで選び直して時間的な散らばりを取り戻す。
    """
    saved_idx = [i for i, p in enumerate(points) if p.saved]
    if len(saved_idx) <= n_target:
        return list(points)

    scores = [
        pose_score(np.asarray(points[i].keypoints, dtype=np.float32)) for i in saved_idx
    ]
    chosen = {saved_idx[j] for j in select_frames(scores, n_target=n_target, n_bins=n_bins)}
    return [
        _rebuild_point(p, saved=(i in chosen)) if p.saved else p
        for i, p in enumerate(points)
    ]


def _track_motion(
    points: list[FramePoint], H: np.ndarray | None
) -> tuple[float, float]:
    """トラックの総移動距離と位置の標準偏差を返す。

    ホモグラフィがあれば BEV 実空間（m）で、無ければ画像座標（px）で測る。
    単位が変わるので、閾値も呼び出し側で使い分ける。
    """
    foot = np.asarray([p.foot_point for p in points], dtype=np.float64)
    coords = to_world(foot, H) if H is not None else foot

    if len(coords) < 2:
        return 0.0, 0.0
    total = float(np.linalg.norm(np.diff(coords, axis=0), axis=1).sum())
    # x と y の標準偏差を合成して「どれだけ散らばっているか」の1つの尺度にする
    spread = float(np.hypot(*coords.std(axis=0)))
    return total, spread


def remove_static_detections(
    result: TrackingResult,
    H: np.ndarray | None,
    params: dict[str, Any],
) -> tuple[TrackingResult, list[dict[str, Any]]]:
    """背景の静止物を人物と誤検出し続けたトラックを除く。

    pose モデルは靴置き場や暗い構造物を低い信頼度で人物と取り続けることがある。
    実在の人物と違って動かないので、ほとんど移動していないトラックを落とす。

    条件は2つ。長く居座るのに動かないもの（1）と、散らばりが小さく信頼度も
    低いもの（2）。短時間立ち止まっただけの実在人物を巻き込まないよう、
    (1) には長さの下限を、(2) には信頼度の上限を付けてある。
    """
    use_world = H is not None
    max_distance = float(
        params["static_max_distance_m"] if use_world else params["static_max_distance_px"]
    )
    max_std = float(
        params["static_max_std_m"] if use_world else params["static_max_std_px"]
    )
    min_len = int(params["static_min_len"])
    max_conf = float(params["static_max_conf"])
    unit = "m" if use_world else "px"

    kept: dict[str, list[FramePoint]] = {}
    removed: list[dict[str, Any]] = []

    for tid, points in result.tracks.items():
        distance, spread = _track_motion(points, H)
        conf_median = statistics.median(p.conf for p in points)

        reasons = []
        if distance < max_distance and len(points) > min_len:
            reasons.append("移動距離が小さく長時間居座る")
        if spread < max_std and conf_median < max_conf:
            reasons.append("散らばりが小さく信頼度も低い")

        if not reasons:
            kept[tid] = points
            continue

        removed.append(
            {
                "track_id": tid,
                "frame_range": (points[0].frame_idx, points[-1].frame_idx),
                "n_points": len(points),
                "distance": distance,
                "spread": spread,
                "conf_median": conf_median,
                "unit": unit,
                "reasons": reasons,
            }
        )

    return TrackingResult(meta=result.meta, tracks=kept), removed


def copy_removed_crops(
    removed: list[dict[str, Any]],
    before_removal: TrackingResult,
    out_dir: str | Path,
) -> int:
    """除外したトラックの crop を1枚ずつ控える。

    誤って実在の人物を落としていないか目視で確かめるための材料なので、
    元の crop は消さずにコピーする。除外済みのトラックを引くので、
    `before_removal` には `remove_static_detections` に渡した方の結果を渡す。
    """
    dest_dir = Path(out_dir) / "removed_static"
    dest_dir.mkdir(parents=True, exist_ok=True)

    n_copied = 0
    for info in removed:
        tid = info["track_id"]
        points = before_removal.tracks.get(tid) or []
        saved = next((p for p in points if p.saved), None)
        if saved is None:
            continue
        src = crop_path(out_dir, tid, saved.frame_idx)
        if src.is_file():
            shutil.copy2(src, dest_dir / src.name)
            n_copied += 1
    return n_copied
