"""時空間尤度と σ 推定のテスト。"""

from __future__ import annotations

import numpy as np
import pytest

from mct.candidates import Candidate
from mct.config import TransitionRoute
from mct.spatiotemporal import (
    KIND_IN_RANGE,
    KIND_OUT_OF_RANGE,
    KIND_SELF_LOOP,
    SigmaEstimate,
    SigmaTable,
    collect_errors,
    estimate_sigma,
    predicted_arrival,
    score_candidates,
    spatiotemporal_likelihood,
    speed_medians,
)
from mct.st_report import auc
from mct.summarize import TrackSummary

PARAMS = {"fps": 30, "sigma_min_samples": 5, "sigma_default_frames": 60}
SIGMA = SigmaTable(default=60.0)


def _route(distance=30.0, c_dwell=0.2, from_cam="camA", from_zone="a_out",
           to_cam="camB", to_zone="b_in") -> TransitionRoute:
    return TransitionRoute(
        from_cam=from_cam, from_zone=from_zone, to_cam=to_cam, to_zone=to_zone,
        distance_m=distance, search_range_sec=(1.0, 60.0), c_dwell=c_dwell,
    )


def _summary(camera: str, track_id: int, speed: float | None,
             start: int = 1, end: int = 100) -> TrackSummary:
    return TrackSummary(
        date="0720", camera=camera, track_id=track_id,
        start_frame=start, end_frame=end, n_points=end - start + 1,
        entry_point=[0.0, 0.0], exit_point=[0.0, 0.0], speed_mps=speed,
    )


def _candidate(route, src, tgt, *, in_range=True) -> Candidate:
    return Candidate(
        src=src.key, tgt=tgt.key,
        src_zone=route.to_zone if route else "", tgt_zone=route.from_zone if route else "",
        route=route, delta_frames=src.start_frame - tgt.end_frame,
        in_search_range=in_range,
    )


# --- 予測到着時刻 ---------------------------------------------------------


def test_predicted_arrival_uses_distance_over_speed() -> None:
    """30 m を 1 m/s で歩けば 30 秒 = 900 フレーム後。"""
    tgt = _summary("camA", 1, 1.0, end=100)
    src = _summary("camB", 2, 1.0, start=1000)

    arrival, v = predicted_arrival(tgt, src, _route(distance=30.0), 30.0)

    assert arrival == pytest.approx(100 + 900)
    assert v == pytest.approx(1.0)


def test_predicted_arrival_uses_harmonic_mean() -> None:
    """速度が違えば調和平均。1 と 2 なら 4/3 m/s。"""
    tgt = _summary("camA", 1, 1.0, end=100)
    src = _summary("camB", 2, 2.0, start=1000)

    arrival, v = predicted_arrival(tgt, src, _route(distance=40.0), 30.0)

    assert v == pytest.approx(4.0 / 3.0)
    assert arrival == pytest.approx(100 + 40.0 / (4.0 / 3.0) * 30)


def test_predicted_arrival_without_speed() -> None:
    tgt = _summary("camA", 1, None, end=100)
    src = _summary("camB", 2, None, start=1000)

    assert predicted_arrival(tgt, src, _route(), 30.0) == (None, None)


# --- 尤度の分岐 -----------------------------------------------------------


def test_in_range_uses_gaussian() -> None:
    route = _route(distance=30.0)
    tgt = _summary("camA", 1, 1.0, end=100)
    src = _summary("camB", 2, 1.0, start=1000)      # 予測ちょうど
    summaries = {tgt.key: tgt, src.key: src}

    score = spatiotemporal_likelihood(
        _candidate(route, src, tgt), summaries, SIGMA, PARAMS
    )

    assert score.kind == KIND_IN_RANGE
    assert score.error_frames == pytest.approx(0.0)
    assert score.likelihood == pytest.approx(1.0)


def test_error_of_one_sigma_gives_exp_minus_half() -> None:
    route = _route(distance=30.0)
    tgt = _summary("camA", 1, 1.0, end=100)
    src = _summary("camB", 2, 1.0, start=1000 - 60)   # σ=60 ぶん早く出現
    summaries = {tgt.key: tgt, src.key: src}

    score = spatiotemporal_likelihood(
        _candidate(route, src, tgt), summaries, SIGMA, PARAMS
    )

    assert score.error_frames == pytest.approx(60.0)
    assert score.likelihood == pytest.approx(np.exp(-0.5))


def test_out_of_range_uses_c_dwell() -> None:
    route = _route(c_dwell=0.25)
    tgt = _summary("camA", 1, 1.0, end=100)
    src = _summary("camB", 2, 1.0, start=9000)
    summaries = {tgt.key: tgt, src.key: src}

    score = spatiotemporal_likelihood(
        _candidate(route, src, tgt, in_range=False), summaries, SIGMA, PARAMS
    )

    assert score.kind == KIND_OUT_OF_RANGE
    assert score.likelihood == pytest.approx(0.25)
    assert score.error_frames is None


def test_self_loop_uses_c_dwell() -> None:
    route = _route(c_dwell=0.3, to_cam="camA", to_zone="a_out")
    tgt = _summary("camA", 1, 1.0, end=100)
    src = _summary("camA", 2, 1.0, start=5000)
    summaries = {tgt.key: tgt, src.key: src}

    score = spatiotemporal_likelihood(
        _candidate(route, src, tgt), summaries, SIGMA, PARAMS
    )

    assert score.kind == KIND_SELF_LOOP
    assert score.likelihood == pytest.approx(0.3)


def test_per_zone_sigma_is_used() -> None:
    route = _route(distance=30.0)
    key = (route.from_cam, route.from_zone, route.to_cam, route.to_zone)
    table = SigmaTable(default=60.0, per_zone={key: 30.0})
    tgt = _summary("camA", 1, 1.0, end=100)
    src = _summary("camB", 2, 1.0, start=1000 - 30)
    summaries = {tgt.key: tgt, src.key: src}

    score = spatiotemporal_likelihood(_candidate(route, src, tgt), summaries, table, PARAMS)

    # 個別 σ=30 なので誤差30は 1σ 相当
    assert score.likelihood == pytest.approx(np.exp(-0.5))


# --- 速度のフォールバック -------------------------------------------------


def test_missing_speed_falls_back_to_camera_median() -> None:
    route = _route(distance=30.0)
    tgt = _summary("camA", 1, None, end=100)
    src = _summary("camB", 2, 1.0, start=1000)
    summaries = {tgt.key: tgt, src.key: src}

    score = spatiotemporal_likelihood(
        _candidate(route, src, tgt), summaries, SIGMA, PARAMS,
        medians={"camA": 1.0}, overall_median=5.0,
    )

    assert score.speed_imputed is True
    assert score.v_harmonic == pytest.approx(1.0)      # 全体中央値 5.0 ではない


def test_missing_speed_falls_back_to_overall_median() -> None:
    route = _route(distance=30.0)
    tgt = _summary("camA", 1, None, end=100)
    src = _summary("camB", 2, 1.0, start=1000)
    summaries = {tgt.key: tgt, src.key: src}

    score = spatiotemporal_likelihood(
        _candidate(route, src, tgt), summaries, SIGMA, PARAMS,
        medians={}, overall_median=1.0,
    )

    assert score.speed_imputed is True
    assert score.v_harmonic == pytest.approx(1.0)


def test_kind_stays_in_range_without_speed() -> None:
    """速度が取れなくても種類は候補の性質のまま。尤度だけ既定値に落とす。"""
    route = _route(c_dwell=0.2)
    tgt = _summary("camA", 1, None, end=100)
    src = _summary("camB", 2, None, start=1000)
    summaries = {tgt.key: tgt, src.key: src}

    score = spatiotemporal_likelihood(
        _candidate(route, src, tgt), summaries, SIGMA, PARAMS,
        medians={}, overall_median=None,
    )

    assert score.kind == KIND_IN_RANGE
    assert score.likelihood == pytest.approx(0.2)


def test_speed_medians() -> None:
    summaries = [
        _summary("camA", 1, 1.0), _summary("camA", 2, 3.0), _summary("camB", 3, 2.0),
        _summary("camB", 4, None),
    ]
    medians, overall = speed_medians(summaries)

    assert medians == {"camA": 2.0, "camB": 2.0}
    assert overall == pytest.approx(2.0)


# --- σ の推定 -------------------------------------------------------------


def _estimate(errors: list[float], key=("camA", "a_out", "camB", "b_in")) -> SigmaEstimate:
    return SigmaEstimate(*key, errors=list(errors))


def test_sigma_is_mad_based() -> None:
    """σ = 1.4826 × MAD。"""
    e = _estimate([-20.0, -10.0, 0.0, 10.0, 20.0])

    assert e.median_e == pytest.approx(0.0)
    assert e.mad_e == pytest.approx(10.0)
    assert e.sigma == pytest.approx(14.826)


def test_mad_resists_outliers() -> None:
    """滞留したサンプルが1件混ざっても σ はほとんど動かない。"""
    base = [-20.0, -10.0, 0.0, 10.0, 20.0]
    with_outlier = base + [5000.0]

    clean = _estimate(base).sigma
    dirty = _estimate(with_outlier).sigma

    assert dirty == pytest.approx(clean, rel=0.6)
    # 標準偏差なら桁が変わってしまうことを対比で示す
    assert np.std(with_outlier) > 20 * np.std(base)


def test_per_zone_adopted_when_enough_samples() -> None:
    key = ("camA", "a_out", "camB", "b_in")
    samples = {key: _estimate([-20.0, -10.0, 0.0, 10.0, 20.0], key)}

    table, estimates, pooled = estimate_sigma([samples], PARAMS)

    assert estimates[key].adopted is True
    assert table.per_zone[key] == pytest.approx(14.826)


def test_few_samples_fall_back_to_pooled() -> None:
    key = ("camA", "a_out", "camB", "b_in")
    samples = {key: _estimate([-10.0, 0.0, 10.0], key)}      # 3 件 < 5

    table, estimates, pooled = estimate_sigma([samples], PARAMS)

    assert estimates[key].adopted is False
    assert table.per_zone == {}
    assert table.get(_route()) == pytest.approx(pooled)


def test_pooled_sigma_uses_all_zone_pairs() -> None:
    a = ("camA", "a_out", "camB", "b_in")
    b = ("camB", "b_out", "camC", "c_in")
    samples = {
        a: _estimate([-10.0, 0.0, 10.0], a),
        b: _estimate([-10.0, 0.0, 10.0], b),
    }

    _, _, pooled = estimate_sigma([samples], PARAMS)

    assert pooled == pytest.approx(1.4826 * 10.0)


def test_sigma_defaults_without_samples() -> None:
    table, estimates, pooled = estimate_sigma([], PARAMS)

    assert estimates == {}
    assert pooled == pytest.approx(60.0)
    assert table.get(None) == pytest.approx(60.0)


def test_collect_errors_skips_out_of_range_and_self_loop() -> None:
    route = _route(distance=30.0)
    tgt = _summary("camA", 1, 1.0, end=100)
    src = _summary("camB", 2, 1.0, start=1000)
    summaries = [tgt, src]
    correct = {(tgt.key, src.key)}

    in_range = _candidate(route, src, tgt)
    out_range = _candidate(route, src, tgt, in_range=False)

    assert collect_errors([in_range], summaries, correct, PARAMS)
    assert collect_errors([out_range], summaries, correct, PARAMS) == {}


def test_collect_errors_skips_wrong_pairs() -> None:
    route = _route(distance=30.0)
    tgt = _summary("camA", 1, 1.0, end=100)
    src = _summary("camB", 2, 1.0, start=1000)

    assert collect_errors([_candidate(route, src, tgt)], [tgt, src], set(), PARAMS) == {}


# --- AUC ------------------------------------------------------------------


def test_auc_perfect_separation() -> None:
    assert auc([0.9, 0.8, 0.7], [0.3, 0.2, 0.1]) == pytest.approx(1.0)


def test_auc_reversed_separation() -> None:
    assert auc([0.1, 0.2], [0.8, 0.9]) == pytest.approx(0.0)


def test_auc_all_ties_is_half() -> None:
    assert auc([0.5] * 4, [0.5] * 6) == pytest.approx(0.5)


def test_auc_partial_overlap() -> None:
    # 正解 2件, 誤 2件。0.6 は 0.5 に勝ち 0.7 に負ける
    assert auc([0.6, 0.8], [0.5, 0.7]) == pytest.approx(0.75)


def test_score_candidates_covers_every_candidate() -> None:
    route = _route()
    tgt = _summary("camA", 1, 1.0, end=100)
    src = _summary("camB", 2, 1.0, start=1000)

    scores = score_candidates([_candidate(route, src, tgt)], [tgt, src], SIGMA, PARAMS)

    assert len(scores) == 1 and scores[0].kind == KIND_IN_RANGE
