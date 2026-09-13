"""保存フレーム選抜のテスト。"""

from __future__ import annotations

import numpy as np
import pytest

from mct.select import L_HIP, L_SHOULDER, R_HIP, R_SHOULDER, pose_score, select_frames


def _kps(ls: float, rs: float, lh: float, rh: float) -> np.ndarray:
    """指定した4点だけ信頼度を持つ COCO 17点配列を作る。"""
    kps = np.zeros((17, 3), dtype=np.float32)
    kps[L_SHOULDER, 2] = ls
    kps[R_SHOULDER, 2] = rs
    kps[L_HIP, 2] = lh
    kps[R_HIP, 2] = rh
    return kps


def test_pose_score_sums_six_points() -> None:
    # 実測4点 + 両肩の中点 (0.9+0.8)/2 + 両肩両腰の中点 (0.9+0.8+0.7+0.6)/4
    score = pose_score(_kps(0.9, 0.8, 0.7, 0.6))
    expected = 0.9 + 0.8 + 0.7 + 0.6 + (0.9 + 0.8) / 2 + (0.9 + 0.8 + 0.7 + 0.6) / 4
    assert score == pytest.approx(expected)


def test_pose_score_ignores_other_keypoints() -> None:
    """肩・腰以外の信頼度はスコアに影響しない。"""
    base = _kps(0.5, 0.5, 0.5, 0.5)
    noisy = base.copy()
    noisy[0, 2] = 1.0   # 鼻
    noisy[9, 2] = 1.0   # 左手首
    assert pose_score(noisy) == pytest.approx(pose_score(base))


def test_pose_score_is_zero_for_invisible_torso() -> None:
    assert pose_score(np.zeros((17, 3), dtype=np.float32)) == pytest.approx(0.0)


def test_short_track_selects_every_frame() -> None:
    for n in (1, 45, 89, 90):
        assert select_frames([0.5] * n) == list(range(n))


def test_long_track_selects_exactly_n_target() -> None:
    rng = np.random.default_rng(0)
    scores = rng.random(1000).tolist()
    chosen = select_frames(scores)

    assert len(chosen) == 90
    assert chosen == sorted(chosen)
    assert len(set(chosen)) == 90


def test_selection_is_spread_over_all_bins() -> None:
    """先頭に高スコアを固めても、後半の区画から取りこぼさない。"""
    n = 900
    scores = [10.0] * 100 + [0.1] * (n - 100)
    chosen = select_frames(scores, n_target=90, n_bins=9)

    bins = np.array_split(np.arange(n), 9)
    per_bin = [sum(1 for i in chosen if i in set(b.tolist())) for b in bins]
    assert per_bin == [10] * 9


def test_top_scores_within_a_bin_win() -> None:
    """各区画の中ではスコア上位が選ばれる。"""
    n = 900
    scores = [0.0] * n
    # 第1区画(0..99)のうち 0..9 だけ高スコアにする
    for i in range(10):
        scores[i] = 1.0
    chosen = set(select_frames(scores, n_target=90, n_bins=9))

    assert set(range(10)).issubset(chosen)


def test_fills_from_global_top_when_bins_are_short() -> None:
    """区画が10件に満たないときも合計90件になる。"""
    n = 91  # 9等分すると 11,10,10,10,10,10,10,10,10 で最大でも 91 件
    scores = [float(i) for i in range(n)]
    chosen = select_frames(scores, n_target=90, n_bins=9)

    assert len(chosen) == 90
    # 最低スコアの 0 番だけが落ちる
    assert 0 not in chosen


def test_is_deterministic_with_ties() -> None:
    """全部同点でも結果が一意に決まり、先頭側が優先される。"""
    scores = [1.0] * 500
    first = select_frames(scores)
    assert first == select_frames(scores)
    assert len(first) == 90


def test_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        select_frames([1.0] * 200, n_target=0)
    with pytest.raises(ValueError):
        select_frames([1.0] * 200, n_bins=0)
