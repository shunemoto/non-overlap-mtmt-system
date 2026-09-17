"""候補ペアの外観類似度のテスト。

記述子だけを入力にするので、画像もモデルも使わない。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from mct.appearance import N_JOINTS, AppearanceDescriptor
from mct.appearance_sim import (
    MISSING_BOTH,
    MISSING_COLOR,
    MISSING_SIFT,
    AppScore,
    bhattacharyya,
    color_similarity,
    combine,
    score_pair,
    score_pairs,
    sift_similarity,
)
from mct.candidates import Candidate

PARAMS = {"lambda_missing": 0.9}


def _hist(peak: int, size: int = 145) -> list[float]:
    h = np.full(size, 0.001)
    h[peak] = 1.0
    return list(h / h.sum())


def _unit(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=128)
    return list(v / np.linalg.norm(v))


def _descriptor(
    camera: str = "camA", track_id: int = 1, *, peak: int = 3, seed: int = 0,
    n_valid: int = N_JOINTS, color: bool = True,
) -> AppearanceDescriptor:
    return AppearanceDescriptor(
        camera=camera, track_id=track_id,
        hist_upper=_hist(peak) if color else None,
        hist_lower=_hist(peak + 1) if color else None,
        sift=[_unit(seed + j) for j in range(N_JOINTS)],
        sift_valid=[j < n_valid for j in range(N_JOINTS)],
        n_frames_used=30,
    )


# --- Bhattacharyya --------------------------------------------------------


def test_bhattacharyya_matches_opencv() -> None:
    rng = np.random.default_rng(0)
    a = rng.random(145); a /= a.sum()
    b = rng.random(145); b /= b.sum()

    mine = bhattacharyya(list(a), list(b))
    theirs = cv2.compareHist(
        a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA
    )

    assert mine == pytest.approx(theirs, abs=1e-6)


def test_bhattacharyya_of_identical_histograms_is_zero() -> None:
    assert bhattacharyya(_hist(3), _hist(3)) == pytest.approx(0.0, abs=1e-9)


def test_bhattacharyya_needs_both_sides() -> None:
    assert bhattacharyya(None, _hist(3)) is None
    assert bhattacharyya(_hist(3), None) is None


# --- 色 -------------------------------------------------------------------


def test_identical_colour_gives_one() -> None:
    a = _descriptor()

    s_color, s_upper, s_lower = color_similarity(a, a, 0.9)

    assert s_color == pytest.approx(1.0)
    assert s_upper == pytest.approx(1.0) and s_lower == pytest.approx(1.0)


def test_missing_half_applies_lambda() -> None:
    """下半身が取れなければ、上半身のスコアに lambda_missing が掛かる。"""
    a = _descriptor()
    b = _descriptor(track_id=2)
    b.hist_lower = None

    s_color, s_upper, s_lower = color_similarity(a, b, 0.9)

    assert s_lower is None
    assert s_color == pytest.approx(s_upper * 0.9)


def test_colour_missing_on_both_sides() -> None:
    a = _descriptor(color=False)

    assert color_similarity(a, a, 0.9) == (None, None, None)


def test_different_colours_score_lower_than_identical() -> None:
    a = _descriptor(peak=3)
    b = _descriptor(track_id=2, peak=100)

    assert color_similarity(a, b, 0.9)[0] < color_similarity(a, a, 0.9)[0]


# --- SIFT -----------------------------------------------------------------


def test_identical_sift_gives_one() -> None:
    a = _descriptor()

    value, n_joints = sift_similarity(a, a)

    assert value == pytest.approx(1.0)
    assert n_joints == N_JOINTS


def test_only_jointly_valid_joints_are_used() -> None:
    a = _descriptor(n_valid=6)
    b = _descriptor(track_id=2, seed=0, n_valid=3)

    value, n_joints = sift_similarity(a, b)

    assert n_joints == 3
    assert value == pytest.approx(1.0)       # 同じ seed なので記述子は一致


def test_fewer_than_two_joints_is_missing() -> None:
    a = _descriptor(n_valid=1)
    b = _descriptor(track_id=2, n_valid=1)

    value, n_joints = sift_similarity(a, b)

    assert value is None and n_joints == 1


def test_exactly_two_joints_is_enough() -> None:
    a = _descriptor(n_valid=2)
    b = _descriptor(track_id=2, seed=0, n_valid=2)

    value, n_joints = sift_similarity(a, b)

    assert value is not None and n_joints == 2


def test_sift_can_be_restricted_to_one_joint() -> None:
    """5f の関節点ごとの寄与で使う経路。"""
    a = _descriptor()

    _, n_joints = sift_similarity(a, a, joints=[2])

    assert n_joints == 1


def test_sift_missing_when_descriptor_absent() -> None:
    a = _descriptor()
    b = _descriptor(track_id=2)
    b.sift = None

    assert sift_similarity(a, b) == (None, 0)


# --- 統合 -----------------------------------------------------------------


def test_combine_uses_geometric_mean() -> None:
    value, missing = combine(0.64, 0.49, 0.9)

    assert value == pytest.approx(0.56)      # sqrt(0.64 * 0.49)
    assert missing is None


def test_combine_penalises_missing_sift() -> None:
    value, missing = combine(0.8, None, 0.9)

    assert value == pytest.approx(0.72)
    assert missing == MISSING_SIFT


def test_combine_penalises_missing_colour() -> None:
    value, missing = combine(None, 0.8, 0.9)

    assert value == pytest.approx(0.72)
    assert missing == MISSING_COLOR


def test_combine_with_nothing_left() -> None:
    assert combine(None, None, 0.9) == (None, MISSING_BOTH)


def test_identical_descriptors_score_one() -> None:
    a = _descriptor()

    score = score_pair(a, a, PARAMS)

    assert score.s_app == pytest.approx(1.0)
    assert score.s_color == pytest.approx(1.0)
    assert score.s_sift == pytest.approx(1.0)
    assert score.missing is None


def test_score_pair_without_descriptor() -> None:
    assert score_pair(_descriptor(), None, PARAMS) is None


def test_single_source_candidate_is_penalised_below_a_complete_one() -> None:
    """根拠が片方だけの候補が、両方そろった候補と同じ土俵に並ばない。"""
    a = _descriptor()
    b = _descriptor(track_id=2, seed=0)         # 記述子は完全一致
    b.hist_upper = b.hist_lower = None

    complete = score_pair(a, a, PARAMS)
    partial = score_pair(a, b, PARAMS)

    assert partial.missing == MISSING_COLOR
    assert partial.s_app == pytest.approx(0.9)
    assert partial.s_app < complete.s_app


# --- 一覧 -----------------------------------------------------------------


def _candidate(src: tuple[str, int], tgt: tuple[str, int]) -> Candidate:
    return Candidate(
        src=src, tgt=tgt, src_zone="", tgt_zone="", route=None,
        delta_frames=100, in_search_range=True,
    )


def test_score_pairs_keeps_the_candidate_order_and_length() -> None:
    descriptors = {("camA", 1): _descriptor(), ("camB", 2): _descriptor("camB", 2)}
    candidates = [
        _candidate(("camA", 1), ("camB", 2)),
        _candidate(("camA", 1), ("camC", 9)),        # 記述子が無い
    ]

    scores = score_pairs(candidates, descriptors, PARAMS)

    assert len(scores) == len(candidates)
    assert scores[0] is not None
    assert scores[1] is None


def test_score_roundtrip() -> None:
    original = AppScore(
        src=("camA", 1), tgt=("camB", 2), s_color=0.5, s_color_upper=0.6,
        s_color_lower=0.4, s_sift=0.7, n_joints=5, s_app=0.59, missing=None,
    )

    restored = AppScore.from_dict(original.to_dict())

    assert restored.src == ("camA", 1) and restored.tgt == ("camB", 2)
    assert restored.n_joints == 5
    assert restored.s_app == pytest.approx(0.59)
