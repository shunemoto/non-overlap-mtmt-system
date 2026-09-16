"""トラック要約と速度算出のテスト。"""

from __future__ import annotations

import numpy as np
import pytest

from mct.config import CameraConfig, Region, TransitionZone
from mct.geometry import compute_homography
from mct.schema import FramePoint
from mct.summarize import (
    TrackSummary, harmonic_mean_speed, segment_speeds, summarize_camera,
    summarize_track, track_speed,
)

PARAMS = {
    "fps": 30, "speed_target_sec": 1.5, "speed_min_segments": 3,
    "speed_stop_threshold": 0.3, "speed_min_mps": 0.6, "speed_max_mps": 2.0,
}
# クリップ無効（単位の検算用）
WIDE = {**PARAMS, "speed_min_mps": 0.0, "speed_max_mps": 1e9}

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
    """30 fps で毎フレーム 100px(=1m) 進めば 30 m/s。単位の確認（クリップ無効）。"""
    assert track_speed(_points(300, 100.0), H_CM, WIDE)[0] == pytest.approx(30.0, rel=1e-3)


def test_walking_speed() -> None:
    """毎フレーム 5px(=5cm) なら 1.5 m/s。"""
    assert track_speed(_points(300, 5.0), H_CM, PARAMS)[0] == pytest.approx(1.5, rel=1e-3)


def test_standing_still_yields_none() -> None:
    """全区間が停止とみなされたら None（呼び出し側がカメラ中央値で代用する）。"""
    assert track_speed(_points(300, 0.0), H_CM, PARAMS) == (None, False)


def test_stop_segments_are_excluded() -> None:
    """止まっている区間を混ぜても、歩いている速さが返る。

    前半は 5px/frame(=1.5 m/s)、後半は静止。停止を含めた中央値なら
    0.75 m/s 付近になるが、除外するので 1.5 m/s のままになる。
    """
    moving = _points(300, 5.0)
    still = [
        FramePoint(
            frame_idx=301 + i, bbox=[1495.0, 500.0, 20.0, 40.0], conf=0.9,
            foot_point=[1495.0, 500.0], saved=False,
        )
        for i in range(300)
    ]
    assert track_speed(moving + still, H_CM, PARAMS)[0] == pytest.approx(1.5, rel=0.05)


def test_speed_is_clipped_to_range() -> None:
    """ホモグラフィの外挿で出る有り得ない速さを妥当範囲に収める。"""
    fast, clipped = track_speed(_points(300, 100.0), H_CM, PARAMS)

    assert fast == pytest.approx(2.0)      # speed_max_mps
    assert clipped is True


def test_slow_speed_is_clipped_up() -> None:
    # 1.5px/frame = 0.45 m/s。停止閾値 0.3 は超えるが下限 0.6 は下回る
    slow, clipped = track_speed(_points(300, 1.5), H_CM, PARAMS)

    assert slow == pytest.approx(0.6)      # speed_min_mps
    assert clipped is True


def test_speed_within_range_is_not_clipped() -> None:
    assert track_speed(_points(300, 5.0), H_CM, PARAMS)[1] is False


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
    assert track_speed(slow + fast, H_CM, WIDE)[0] == pytest.approx(0.9, rel=0.2)


def test_speed_without_homography_is_none() -> None:
    assert track_speed(_points(100, 5.0), None, PARAMS) == (None, False)


def test_speed_with_too_few_points_is_none() -> None:
    assert track_speed(_points(1, 5.0), H_CM, PARAMS) == (None, False)


def test_segment_count_follows_target_interval() -> None:
    """長いトラックほど区間数が増え、短くても下限を割らない。"""
    assert len(segment_speeds(_points(45, 5.0), H_CM, PARAMS)) == 3      # 下限
    assert len(segment_speeds(_points(450, 5.0), H_CM, PARAMS)) == 10    # 450/45


def test_harmonic_mean_speed() -> None:
    """到着時刻は距離÷速度なので、平均すべきは速度ではなく所要時間。"""
    assert harmonic_mean_speed(1.0, 1.0) == pytest.approx(1.0)
    assert harmonic_mean_speed(1.0, 2.0) == pytest.approx(4.0 / 3.0)
    # 調和平均は算術平均より小さい（遅い側に寄る）
    assert harmonic_mean_speed(0.5, 2.0) < 1.25


def test_harmonic_mean_needs_both_speeds() -> None:
    assert harmonic_mean_speed(None, 1.0) is None
    assert harmonic_mean_speed(1.0, None) is None
    assert harmonic_mean_speed(0.0, 1.0) is None


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


def test_summary_has_single_speed() -> None:
    """速度はトラック全体で1つ。前半・後半の分離はしない。"""
    s = summarize_track("0720", "camA", "1", _points(300, 5.0), _camera(), H_CM, PARAMS, 30)

    assert s.speed_mps == pytest.approx(1.5, rel=1e-2)
    assert s.speed_fallback is False and s.speed_clipped is False


def test_summary_without_homography_has_no_speed() -> None:
    s = summarize_track("0720", "camA", "1", _points(300, 5.0), _camera(), None, PARAMS, 30)

    assert s.speed_mps is None


def test_camera_median_fills_missing_speed() -> None:
    """ずっと止まっていたトラックには、同じカメラ・同じ日の中央値を当てる。"""
    tracks = {
        "1": _points(300, 5.0),      # 1.5 m/s
        "2": _points(300, 5.0),      # 1.5 m/s
        "3": _points(300, 0.0),      # 静止 -> 算出できない
    }
    summaries = summarize_camera("0720", "camA", tracks, _camera(), H_CM, PARAMS, 30)
    by_id = {s.track_id: s for s in summaries}

    assert by_id[3].speed_mps == pytest.approx(1.5, rel=1e-2)
    assert by_id[3].speed_fallback is True
    assert by_id[1].speed_fallback is False


def test_camera_median_is_skipped_when_nothing_is_known() -> None:
    """全トラックが静止なら代用元が無いので None のまま。"""
    summaries = summarize_camera(
        "0720", "camA", {"1": _points(300, 0.0)}, _camera(), H_CM, PARAMS, 30
    )

    assert summaries[0].speed_mps is None
    assert summaries[0].speed_fallback is False


def test_summary_roundtrip() -> None:
    s = summarize_track("0720", "camA", "1", _points(300, 5.0), _camera(), H_CM, PARAMS, 30)

    assert TrackSummary.from_dict(s.to_dict()) == s
