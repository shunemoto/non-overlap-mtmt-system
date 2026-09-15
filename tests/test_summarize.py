"""トラック要約と速度算出のテスト。"""

from __future__ import annotations

import numpy as np
import pytest

from mct.config import CameraConfig, Region, TransitionZone
from mct.geometry import compute_homography
from mct.schema import FramePoint
from mct.summarize import TrackSummary, speed_mps, summarize_track

PARAMS = {"fps": 30, "speed_target_sec": 1.5, "speed_min_segments": 3}

# 画像1px = 実空間1cm。10px 動けば 0.1 m。
H_CM = compute_homography(
    [[0, 0], [1000, 0], [1000, 1000], [0, 1000]],
    [[0, 0], [1000, 0], [1000, 1000], [0, 1000]],
    "cm",
)


def _points(n: int, step_px: float, y: float = 500.0, start: int = 1) -> list[FramePoint]:
    return [
        FramePoint(
            frame_idx=start + i, bbox=[i * step_px, y, 20.0, 40.0], conf=0.9,
            foot_point=[i * step_px, y], saved=False,
        )
        for i in range(n)
    ]


# --- 速度 -----------------------------------------------------------------


def test_speed_is_metres_per_second() -> None:
    """30 fps で毎フレーム 100px(=1m) 進めば 30 m/s。単位の確認。"""
    assert speed_mps(_points(300, 100.0), H_CM, PARAMS) == pytest.approx(30.0, rel=1e-3)


def test_walking_speed() -> None:
    """毎フレーム 5px(=5cm) なら 1.5 m/s。"""
    assert speed_mps(_points(300, 5.0), H_CM, PARAMS) == pytest.approx(1.5, rel=1e-3)


def test_standing_still_gives_zero() -> None:
    assert speed_mps(_points(300, 0.0), H_CM, PARAMS) == pytest.approx(0.0)


def test_speed_uses_median_of_segments() -> None:
    """一部だけ速く動いても、中央値なので引きずられない。"""
    slow = _points(200, 3.0)
    fast = [
        FramePoint(
            frame_idx=201 + i, bbox=[600.0 + i * 50, 500.0, 20.0, 40.0], conf=0.9,
            foot_point=[600.0 + i * 50, 500.0], saved=False,
        )
        for i in range(20)
    ]
    median = speed_mps(slow + fast, H_CM, PARAMS)

    assert median == pytest.approx(0.9, rel=0.2)     # 遅い側に寄る


def test_speed_without_homography_is_none() -> None:
    assert speed_mps(_points(100, 5.0), None, PARAMS) is None


def test_speed_with_too_few_points_is_none() -> None:
    assert speed_mps(_points(1, 5.0), H_CM, PARAMS) is None


def test_segment_count_follows_target_interval() -> None:
    """長いトラックほど区間数が増え、短くても下限を割らない。"""
    # 45 フレーム = 1.5 秒 × 1 なので下限の3区間が使われる
    assert speed_mps(_points(45, 5.0), H_CM, PARAMS) == pytest.approx(1.5, rel=1e-3)
    # 450 フレーム = 10 区間
    assert speed_mps(_points(450, 5.0), H_CM, PARAMS) == pytest.approx(1.5, rel=1e-3)


# --- 要約 -----------------------------------------------------------------


def _camera() -> CameraConfig:
    return CameraConfig(
        name="camA", image_size=(1920, 1080),
        transition_zones=[
            TransitionZone(
                id="z_in", region=Region(type="rect", points=[[0, 0], [200, 1080]]),
                entry_dir=(0.0, 90.0),
            ),
            TransitionZone(
                id="z_out", region=Region(type="rect", points=[[1400, 0], [1919, 1080]]),
                exit_dir=(0.0, 90.0),
            ),
        ],
    )


def test_summary_fields() -> None:
    points = _points(300, 5.0, start=10)
    s = summarize_track("0720", "camA", "7", points, _camera(), H_CM, PARAMS, 30)

    assert s.date == "0720" and s.camera == "camA" and s.track_id == 7
    assert s.start_frame == 10 and s.end_frame == 309
    assert s.n_points == 300
    assert s.entry_point == [0.0, 500.0]
    assert s.exit_point == [pytest.approx(1495.0), 500.0]
    assert s.entry_dir_deg == pytest.approx(0.0)      # 右向き
    assert s.exit_dir_deg == pytest.approx(0.0)


def test_summary_detects_zones() -> None:
    points = _points(300, 5.0)
    s = summarize_track("0720", "camA", "1", points, _camera(), H_CM, PARAMS, 30)

    assert s.entry_zones == ["z_in"]     # 開始が x=0
    assert s.exit_zones == ["z_out"]     # 終了が x=1495


def test_half_speeds_are_computed_separately() -> None:
    """前半と後半で速さが違えば、別々の値になる。"""
    slow = _points(300, 2.0)
    fast = [
        FramePoint(
            frame_idx=301 + i, bbox=[600.0 + i * 20, 500.0, 20.0, 40.0], conf=0.9,
            foot_point=[600.0 + i * 20, 500.0], saved=False,
        )
        for i in range(300)
    ]
    s = summarize_track("0720", "camA", "1", slow + fast, _camera(), H_CM, PARAMS, 30)

    assert s.speed_fallback is False
    assert s.entry_speed_mps == pytest.approx(0.6, rel=0.1)
    assert s.exit_speed_mps == pytest.approx(6.0, rel=0.1)


def test_short_track_falls_back_to_whole_speed() -> None:
    """半分にすると区間数が足りない場合、全体の速度を両方に使う。"""
    s = summarize_track("0720", "camA", "1", _points(40, 5.0), _camera(), H_CM, PARAMS, 30)

    assert s.speed_fallback is True
    assert s.entry_speed_mps == s.exit_speed_mps
    assert s.entry_speed_mps == pytest.approx(1.5, rel=1e-2)


def test_summary_without_homography_has_no_speed() -> None:
    s = summarize_track("0720", "camA", "1", _points(300, 5.0), _camera(), None, PARAMS, 30)

    assert s.entry_speed_mps is None and s.exit_speed_mps is None


def test_summary_roundtrip() -> None:
    s = summarize_track("0720", "camA", "1", _points(300, 5.0), _camera(), H_CM, PARAMS, 30)

    assert TrackSummary.from_dict(s.to_dict()) == s
