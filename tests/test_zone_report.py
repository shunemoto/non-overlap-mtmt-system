"""ゾーン該当の判定と CSV 出力のテスト。"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from mct.config import CameraConfig, Region, TransitionZone
from mct.diagnostics import angle_gap, match_zones
from mct.schema import FramePoint, TrackingResult, TrackMeta
from mct.zone_report import COLUMNS, build_rows, write_zone_report

PARAMS = {"dir_frames": 30}


def _zone(zone_id: str, rect, entry=None, exit_=None) -> TransitionZone:
    return TransitionZone(
        id=zone_id,
        region=Region(type="rect", points=rect),
        entry_dir=entry,
        exit_dir=exit_,
    )


def _camera(zones: list[TransitionZone]) -> CameraConfig:
    return CameraConfig(name="camA", image_size=(1920, 1080), transition_zones=zones)


def _point(frame: int, x: float, y: float) -> FramePoint:
    return FramePoint(
        frame_idx=frame, bbox=[x - 10, y - 40, 20.0, 40.0], conf=0.9,
        foot_point=[x, y], saved=False,
    )


def _track(n: int, x0: float, y0: float, dx: float, dy: float) -> list[FramePoint]:
    return [_point(i + 1, x0 + i * dx, y0 + i * dy) for i in range(n)]


def _result(tracks) -> TrackingResult:
    meta = TrackMeta(
        date="0720", camera="camA", width=1920, height=1080, fps=30.0, n_frames=1000,
        model_pose="p", model_seg="s", tracker="t", n_saved_per_track=90,
        created_at="2026-07-20T00:00:00+09:00",
    )
    return TrackingResult(meta=meta, tracks=tracks)


# --- angle_gap -----------------------------------------------------------


def test_angle_gap_is_zero_inside_range() -> None:
    assert angle_gap(150, 110, 240) == 0.0
    assert angle_gap(350, 290, 60) == 0.0       # 0 をまたぐ区間


def test_angle_gap_measures_to_nearest_edge() -> None:
    assert angle_gap(100, 110, 240) == pytest.approx(10.0)
    assert angle_gap(250, 110, 240) == pytest.approx(10.0)


def test_angle_gap_wraps_around_zero() -> None:
    assert angle_gap(70, 290, 60) == pytest.approx(10.0)
    assert angle_gap(280, 290, 60) == pytest.approx(10.0)


# --- match_zones ---------------------------------------------------------


def test_matches_single_zone() -> None:
    cam = _camera([_zone("z1", [[0, 0], [200, 200]], entry=(0.0, 90.0))])
    match = match_zones(_track(60, 50, 50, 2.0, 0.0), cam, 30, entering=True)

    assert match.zones == ["z1"]
    assert match.reason is None
    assert match.direction == pytest.approx(0.0)


def test_reports_all_overlapping_zones() -> None:
    """重なった遷移領域はすべて列挙する（重複の検出が目的）。"""
    cam = _camera([
        _zone("z1", [[0, 0], [200, 200]], entry=(0.0, 90.0)),
        _zone("z2", [[100, 100], [300, 300]], entry=(0.0, 90.0)),
    ])
    match = match_zones(_track(60, 150, 150, 2.0, 0.0), cam, 30, entering=True)

    assert sorted(match.zones) == ["z1", "z2"]
    assert match.is_duplicate


def test_outside_region() -> None:
    cam = _camera([_zone("z1", [[0, 0], [200, 200]], entry=(0.0, 90.0))])
    match = match_zones(_track(60, 900, 900, 2.0, 0.0), cam, 30, entering=True)

    assert match.zones == []
    assert match.reason == "outside_region"
    assert match.nearest_zone is None


def test_angle_mismatch_records_nearest_zone_and_gap() -> None:
    cam = _camera([_zone("z1", [[0, 0], [200, 200]], entry=(100.0, 160.0))])
    # 右向き（0度）に進むので [100,160] から 100 度ずれる
    match = match_zones(_track(60, 50, 50, 2.0, 0.0), cam, 30, entering=True)

    assert match.zones == []
    assert match.reason == "angle_mismatch"
    assert match.nearest_zone == "z1"
    assert match.nearest_range == (100.0, 160.0)
    assert match.nearest_gap == pytest.approx(100.0)


def test_angle_mismatch_picks_the_closest_zone() -> None:
    cam = _camera([
        _zone("far", [[0, 0], [300, 300]], entry=(170.0, 190.0)),   # 170度ずれ
        _zone("near", [[0, 0], [300, 300]], entry=(20.0, 40.0)),    # 20度ずれ
    ])
    match = match_zones(_track(60, 50, 50, 2.0, 0.0), cam, 30, entering=True)

    assert match.reason == "angle_mismatch"
    assert match.nearest_zone == "near"
    assert match.nearest_gap == pytest.approx(20.0)


def test_zone_without_angle_matches_on_position_alone() -> None:
    cam = _camera([_zone("z1", [[0, 0], [200, 200]])])
    match = match_zones(_track(60, 50, 50, 2.0, 0.0), cam, 30, entering=True)

    assert match.zones == ["z1"]


def test_exit_uses_the_last_point_and_reversed_direction() -> None:
    cam = _camera([_zone("z1", [[0, 0], [200, 200]], exit_=(170.0, 190.0))])
    # 左向きに進んで (10, 50) で終わる
    match = match_zones(_track(60, 130, 50, -2.0, 0.0), cam, 30, entering=False)

    assert match.point == (12.0, 50.0)
    assert match.direction == pytest.approx(180.0)
    assert match.zones == ["z1"]


# --- CSV -----------------------------------------------------------------


def test_csv_columns_and_content(tmp_path: Path) -> None:
    cam = _camera([
        _zone("z1", [[0, 0], [200, 200]], entry=(0.0, 90.0)),
        _zone("z2", [[0, 0], [200, 200]], entry=(0.0, 90.0)),
    ])
    result = _result({"7": _track(60, 50, 50, 2.0, 0.0)})

    dest = write_zone_report("0720", "camA", result, cam, PARAMS, tmp_path)
    rows = list(csv.DictReader(dest.open(encoding="utf-8-sig")))

    assert dest.name == "zone_report.csv"
    assert list(rows[0]) == COLUMNS
    assert rows[0]["track_id"] == "7"
    assert rows[0]["start_frame"] == "1"
    assert rows[0]["end_frame"] == "60"
    assert rows[0]["n_points"] == "60"
    assert rows[0]["entry_zones"] == "z1;z2"     # セミコロン区切りで全件
    assert rows[0]["entry_reason"] == ""


def test_csv_reason_columns() -> None:
    cam = _camera([_zone("z1", [[0, 0], [200, 200]], entry=(100.0, 160.0))])
    rows = build_rows(_result({"1": _track(60, 50, 50, 2.0, 0.0)}), cam, PARAMS)

    assert rows[0]["entry_zones"] == ""
    assert "angle_mismatch" in rows[0]["entry_reason"]
    assert "z1" in rows[0]["entry_reason"]
    assert "100" in rows[0]["entry_reason"]     # 定義範囲との差


def test_csv_outside_region_reason() -> None:
    cam = _camera([_zone("z1", [[0, 0], [200, 200]], entry=(0.0, 90.0))])
    rows = build_rows(_result({"1": _track(60, 900, 900, 2.0, 0.0)}), cam, PARAMS)

    assert rows[0]["entry_reason"] == "outside_region"


def test_csv_is_sorted_by_track_id_numerically() -> None:
    cam = _camera([])
    tracks = {str(i): _track(40, 500, 500, 1.0, 0.0) for i in (10, 2, 33, 1)}
    rows = build_rows(_result(tracks), cam, PARAMS)

    assert [r["track_id"] for r in rows] == ["1", "2", "10", "33"]
