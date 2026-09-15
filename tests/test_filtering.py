"""観測区間の適用と静止誤検出の除去のテスト。"""

from __future__ import annotations

import numpy as np
import pytest

from mct.config import CameraConfig, Region
from mct.filtering import apply_observation_region, remove_static_detections
from mct.geometry import compute_homography
from mct.schema import FramePoint, TrackingResult, TrackMeta

PARAMS = {
    "min_track_len": 31,
    "n_frames_per_track": 30,
    "n_bins": 9,
    "static_max_distance_m": 1.0,
    "static_min_len": 60,
    "static_max_std_m": 0.3,
    "static_max_conf": 0.7,
    "static_max_distance_px": 80,
    "static_max_std_px": 25,
    "dir_frames": 30,
}

# 1画素 = 1cm になる素直な対応（画像 (0,0)-(1000,1000) が実空間 10m 四方）
H_UNIT = compute_homography(
    [[0, 0], [1000, 0], [1000, 1000], [0, 1000]],
    [[0, 0], [1000, 0], [1000, 1000], [0, 1000]],
    "cm",
)


def _meta() -> TrackMeta:
    return TrackMeta(
        date="0720", camera="camA", width=1920, height=1080, fps=30.0,
        n_frames=1000, model_pose="p", model_seg="s", tracker="t",
        n_saved_per_track=90, created_at="2026-07-20T00:00:00+09:00",
    )


def _kps(conf: float = 0.9) -> list[list[float]]:
    kps = [[0.0, 0.0, 0.0] for _ in range(17)]
    for i in (5, 6, 11, 12):
        kps[i] = [10.0, 10.0, conf]
    return kps


def _point(frame: int, x: float, y: float, *, saved: bool = False, conf: float = 0.9,
           score_conf: float = 0.9) -> FramePoint:
    return FramePoint(
        frame_idx=frame,
        bbox=[x - 10, y - 40, 20.0, 40.0],
        conf=conf,
        foot_point=[x, y],
        saved=saved,
        keypoints=_kps(score_conf) if saved else None,
        crop_box=[x - 12, y - 44, 24.0, 48.0] if saved else None,
    )


def _track(n: int, x0: float, y0: float, step: float = 0.0, *, n_saved: int = 0,
           conf: float = 0.9) -> list[FramePoint]:
    """step だけ毎フレーム x が進むトラック。step=0 なら静止。"""
    saved_at = set(range(0, n, max(1, n // n_saved))) if n_saved else set()
    saved_at = set(list(saved_at)[:n_saved])
    return [
        _point(i + 1, x0 + i * step, y0, saved=i in saved_at, conf=conf,
               score_conf=0.5 + 0.005 * i)
        for i in range(n)
    ]


def _result(tracks: dict[str, list[FramePoint]]) -> TrackingResult:
    return TrackingResult(meta=_meta(), tracks=tracks)


def _camera(regions: list[Region]) -> CameraConfig:
    return CameraConfig(name="camA", image_size=(1920, 1080), exclude_regions=regions)


# --- 観測区間の適用 ------------------------------------------------------


def test_track_entirely_inside_exclude_is_dropped() -> None:
    cam = _camera([Region(type="rect", points=[[0, 0], [500, 500]])])
    result = _result({"1": _track(50, 100, 100)})

    filtered, stats = apply_observation_region(result, cam, PARAMS)

    assert filtered.tracks == {}
    assert stats["dropped_empty"] == ["1"]
    assert stats["dropped_short"] == []


def test_partially_excluded_track_survives() -> None:
    """除外域にかかる点だけが落ち、トラック自体は残る。"""
    cam = _camera([Region(type="rect", points=[[0, 0], [100, 1080]])])
    # x=50 から 1 ずつ進むので、最初の 51 点が除外域、残り 49 点が残る
    result = _result({"1": _track(100, 50, 500, step=1.0)})

    filtered, stats = apply_observation_region(result, cam, PARAMS)

    assert "1" in filtered.tracks
    assert len(filtered.tracks["1"]) == 49
    assert all(p.foot_point[0] > 100 for p in filtered.tracks["1"])
    assert stats["n_points_removed"] == 51


def test_track_below_min_len_is_dropped() -> None:
    cam = _camera([])
    result = _result({"short": _track(30, 500, 500, step=1.0)})   # 31 未満

    filtered, stats = apply_observation_region(result, cam, PARAMS)

    assert filtered.tracks == {}
    assert stats["dropped_short"] == ["short"]


def test_track_at_min_len_is_kept() -> None:
    cam = _camera([])
    filtered, _ = apply_observation_region(_result({"ok": _track(31, 500, 500, step=1.0)}), cam, PARAMS)

    assert len(filtered.tracks["ok"]) == 31


def test_saved_points_are_limited_to_target() -> None:
    cam = _camera([])
    result = _result({"1": _track(200, 500, 500, step=1.0, n_saved=90)})

    filtered, _ = apply_observation_region(result, cam, PARAMS)
    saved = [p for p in filtered.tracks["1"] if p.saved]

    assert len(saved) == 30
    assert all(p.keypoints is not None and p.crop_box is not None for p in saved)
    # 外れた点は keypoints も crop_box も落ちていること
    assert all(
        p.keypoints is None and p.crop_box is None
        for p in filtered.tracks["1"] if not p.saved
    )


def test_saved_points_below_target_are_untouched() -> None:
    cam = _camera([])
    result = _result({"1": _track(100, 500, 500, step=1.0, n_saved=12)})

    filtered, stats = apply_observation_region(result, cam, PARAMS)

    assert sum(1 for p in filtered.tracks["1"] if p.saved) == 12
    assert stats["n_under_target_saved"] == 1


def test_no_exclude_regions_keeps_everything() -> None:
    cam = _camera([])
    result = _result({"1": _track(50, 500, 500, step=1.0)})

    filtered, stats = apply_observation_region(result, cam, PARAMS)

    assert len(filtered.tracks["1"]) == 50
    assert stats["n_points_removed"] == 0


# --- 静止誤検出の除去 ----------------------------------------------------


def test_static_track_is_removed_with_homography() -> None:
    """長く居座って動かないトラックは落とす。"""
    result = _result({"static": _track(100, 500, 500, step=0.0)})

    filtered, removed = remove_static_detections(result, H_UNIT, PARAMS)

    assert filtered.tracks == {}
    assert len(removed) == 1
    assert removed[0]["track_id"] == "static"
    assert removed[0]["unit"] == "m"
    assert removed[0]["distance"] == pytest.approx(0.0)


def test_moving_track_is_kept_with_homography() -> None:
    # 5px/frame × 99 = 495px = 4.95m 動く
    result = _result({"moving": _track(100, 100, 500, step=5.0)})

    filtered, removed = remove_static_detections(result, H_UNIT, PARAMS)

    assert "moving" in filtered.tracks
    assert removed == []


def test_short_static_track_is_kept() -> None:
    """立ち止まっただけの短いトラックは巻き込まない。"""
    result = _result({"pause": _track(50, 500, 500, step=0.0, conf=0.95)})

    filtered, removed = remove_static_detections(result, H_UNIT, PARAMS)

    assert "pause" in filtered.tracks   # 長さ 50 <= static_min_len 60
    assert removed == []


def test_low_confidence_static_track_is_removed_by_second_rule() -> None:
    """短くても、散らばりが小さく信頼度が低ければ落とす。"""
    result = _result({"ghost": _track(40, 500, 500, step=0.0, conf=0.58)})

    filtered, removed = remove_static_detections(result, H_UNIT, PARAMS)

    assert filtered.tracks == {}
    assert "散らばりが小さく信頼度も低い" in removed[0]["reasons"]


def test_static_track_is_removed_without_homography() -> None:
    """ホモグラフィが無ければ画像座標と px の閾値で判定する。"""
    result = _result({"static": _track(100, 500, 500, step=0.0)})

    filtered, removed = remove_static_detections(result, None, PARAMS)

    assert filtered.tracks == {}
    assert removed[0]["unit"] == "px"


def test_moving_track_is_kept_without_homography() -> None:
    # 2px/frame × 99 = 198px > static_max_distance_px 80
    result = _result({"moving": _track(100, 100, 500, step=2.0)})

    filtered, removed = remove_static_detections(result, None, PARAMS)

    assert "moving" in filtered.tracks
    assert removed == []


def test_removed_entry_carries_diagnostic_fields() -> None:
    result = _result({"static": _track(100, 500, 500, step=0.0)})

    _, removed = remove_static_detections(result, H_UNIT, PARAMS)
    info = removed[0]

    assert info["frame_range"] == (1, 100)
    assert info["n_points"] == 100
    assert info["conf_median"] == pytest.approx(0.9)
    assert info["reasons"]
