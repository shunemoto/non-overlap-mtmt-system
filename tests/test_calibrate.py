"""距離校正のテスト。"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from mct.calibrate import ZoneCalibration, calibrate, collect_samples, write_csv, write_yaml
from mct.candidates import Candidate
from mct.config import TransitionRoute
from mct.summarize import TrackSummary

FPS = 30.0


def _route(from_cam="camA", from_zone="a_out", to_cam="camB", to_zone="b_in",
           distance=10.0) -> TransitionRoute:
    return TransitionRoute(
        from_cam=from_cam, from_zone=from_zone, to_cam=to_cam, to_zone=to_zone,
        distance_m=distance, search_range_sec=(1.0, 60.0), c_dwell=0.1,
    )


def _summary(camera: str, track_id: int, speed: float | None) -> TrackSummary:
    return TrackSummary(
        date="0720", camera=camera, track_id=track_id,
        start_frame=1, end_frame=100, n_points=100,
        entry_point=[0.0, 0.0], exit_point=[0.0, 0.0], speed_mps=speed,
    )


def _candidate(delta: int, route: TransitionRoute, *, self_loop: bool = False) -> Candidate:
    tgt = (route.from_cam, 1)
    src = (route.from_cam, 2) if self_loop else (route.to_cam, 2)
    return Candidate(
        src=src, tgt=tgt, src_zone=route.to_zone, tgt_zone=route.from_zone,
        route=route, delta_frames=delta, in_search_range=True,
    )


def test_implied_distance_from_speed_and_time() -> None:
    """1 m/s で 300 フレーム(10秒)なら 10 m。"""
    route = _route()
    summaries = {("camA", 1): _summary("camA", 1, 1.0), ("camB", 2): _summary("camB", 2, 1.0)}
    c = _candidate(300, route)

    samples = collect_samples([c], summaries, {(c.tgt, c.src)}, FPS)
    entry = samples[route.from_cam, route.from_zone, route.to_cam, route.to_zone]

    assert entry.n_samples == 1
    assert entry.median == pytest.approx(10.0)


def test_uses_harmonic_mean_of_both_speeds() -> None:
    """速度が違えば調和平均が使われる（遅い側に寄る）。"""
    route = _route()
    summaries = {("camA", 1): _summary("camA", 1, 1.0), ("camB", 2): _summary("camB", 2, 2.0)}
    c = _candidate(300, route)

    entry = collect_samples([c], summaries, {(c.tgt, c.src)}, FPS)[
        route.from_cam, route.from_zone, route.to_cam, route.to_zone
    ]
    # 調和平均 4/3 m/s × 10 秒
    assert entry.median == pytest.approx(40.0 / 3.0)


def test_self_loops_are_excluded() -> None:
    """居室滞在は距離と速度から決まらないので校正に使わない。"""
    route = _route(to_cam="camA", to_zone="a_out")
    summaries = {("camA", 1): _summary("camA", 1, 1.0), ("camA", 2): _summary("camA", 2, 1.0)}
    c = _candidate(300, route, self_loop=True)

    assert collect_samples([c], summaries, {(c.tgt, c.src)}, FPS) == {}


def test_incorrect_pairs_are_excluded() -> None:
    route = _route()
    summaries = {("camA", 1): _summary("camA", 1, 1.0), ("camB", 2): _summary("camB", 2, 1.0)}

    assert collect_samples([_candidate(300, route)], summaries, set(), FPS) == {}


def test_missing_speed_is_skipped() -> None:
    route = _route()
    summaries = {("camA", 1): _summary("camA", 1, None), ("camB", 2): _summary("camB", 2, 1.0)}
    c = _candidate(300, route)

    assert collect_samples([c], summaries, {(c.tgt, c.src)}, FPS) == {}


def test_adoption_requires_enough_samples() -> None:
    route = _route()
    key = (route.from_cam, route.from_zone, route.to_cam, route.to_zone)
    few = {key: ZoneCalibration(*key, current_distance_m=10.0, fps=FPS,
                                samples=[(1.0, 300)] * 3)}
    many = {key: ZoneCalibration(*key, current_distance_m=10.0, fps=FPS,
                                 samples=[(1.0, 300)] * 6)}

    assert calibrate([few], [route], 5)[key].adopted is False
    assert calibrate([many], [route], 5)[key].adopted is True


def test_unadopted_keeps_manual_distance() -> None:
    route = _route(distance=10.0)
    key = (route.from_cam, route.from_zone, route.to_cam, route.to_zone)
    samples = {key: ZoneCalibration(*key, current_distance_m=10.0, fps=FPS,
                                    samples=[(1.0, 900)] * 3)}      # 逆算 30 m

    entry = calibrate([samples], [route], 5)[key]

    assert entry.median == pytest.approx(30.0)
    assert entry.adopted_distance_m == pytest.approx(10.0)          # 手動値のまま


def test_routes_without_samples_appear_in_the_table() -> None:
    route = _route()
    key = (route.from_cam, route.from_zone, route.to_cam, route.to_zone)

    entry = calibrate([], [route], 5)[key]

    assert entry.n_samples == 0 and entry.adopted is False


def test_errors_shrink_after_calibration() -> None:
    """逆算した距離を使えば、その標本での誤差は 0 に近づく。"""
    key = ("camA", "a_out", "camB", "b_in")
    entry = ZoneCalibration(*key, current_distance_m=20.0, fps=FPS,
                            samples=[(1.0, 300), (1.0, 300), (1.0, 300)])

    before = entry.errors(entry.current_distance_m)
    after = entry.errors(10.0)      # 逆算値

    assert all(abs(a) < abs(b) for a, b in zip(after, before))


def test_csv_columns(tmp_path: Path) -> None:
    key = ("camA", "a_out", "camB", "b_in")
    calibrations = {key: ZoneCalibration(*key, current_distance_m=10.0, fps=FPS,
                                         samples=[(1.0, 300)] * 6, adopted=True)}

    dest = write_csv(calibrations, tmp_path)
    rows = list(csv.DictReader(dest.open(encoding="utf-8-sig")))

    assert dest.name == "distance_calibration.csv"
    assert rows[0]["n_samples"] == "6"
    assert rows[0]["current_distance_m"] == "10.00"
    assert rows[0]["implied_distance_m_median"] == "10.00"
    assert rows[0]["adopted"] == "True"


def test_yaml_keeps_unadopted_values(tmp_path: Path) -> None:
    routes = [_route(distance=10.0), _route(from_zone="a_alt", distance=25.0)]
    keys = [(r.from_cam, r.from_zone, r.to_cam, r.to_zone) for r in routes]
    calibrations = {
        keys[0]: ZoneCalibration(*keys[0], current_distance_m=10.0, fps=FPS,
                                 samples=[(1.0, 900)] * 6, adopted=True),
        keys[1]: ZoneCalibration(*keys[1], current_distance_m=25.0, fps=FPS, adopted=False),
    }

    text = write_yaml(calibrations, routes, tmp_path / "t.yaml").read_text(encoding="utf-8")

    assert "distance_m: 30.0" in text        # 採用した方は逆算値
    assert "distance_m: 25.0" in text        # 採用しなかった方は手動値
