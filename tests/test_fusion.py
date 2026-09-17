"""スコアの融合のテスト。"""

from __future__ import annotations

import pytest

from mct.fusion import METHODS, app_value, fuse, method_alpha

PARAMS = {"alpha": 0.5, "alpha_self": 0.1, "no_match_score": 0.10}

M6 = METHODS[6]          # トポロジー + 時空間 + 色 + SIFT
M3 = METHODS[3]          # 時空間のみ
M2 = METHODS[2]          # 外観のみ（トポロジーあり）
M1 = METHODS[1]          # 外観のみ（トポロジーなし）
M4 = METHODS[4]          # 時空間 + 色
M5 = METHODS[5]          # 時空間 + SIFT


def _fuse(s_st, s_app, *, method=M6, self_loop=False, params=PARAMS):
    return fuse(s_st, s_app, is_self_loop=self_loop, method=method, params=params)


# --- 重み付き幾何平均 -----------------------------------------------------


def test_fusion_is_a_weighted_geometric_mean() -> None:
    score, alpha = _fuse(0.64, 0.49)

    assert alpha == pytest.approx(0.5)
    assert score == pytest.approx(0.56)          # sqrt(0.64 * 0.49)


def test_alpha_shifts_the_weight() -> None:
    """α を上げると時空間側の値に寄る。"""
    low, _ = _fuse(0.9, 0.1, params=dict(PARAMS, alpha=0.1))
    high, _ = _fuse(0.9, 0.1, params=dict(PARAMS, alpha=0.9))

    assert low < high
    assert high == pytest.approx(0.9**0.9 * 0.1**0.1)


def test_identical_inputs_are_unchanged_by_alpha() -> None:
    for alpha in (0.0, 0.3, 1.0):
        score, _ = _fuse(0.7, 0.7, params=dict(PARAMS, alpha=alpha))
        assert score == pytest.approx(0.7)


# --- 自己ループ -----------------------------------------------------------


def test_self_loop_uses_alpha_self() -> None:
    normal, a_normal = _fuse(0.9, 0.1)
    loop, a_loop = _fuse(0.9, 0.1, self_loop=True)

    assert a_normal == pytest.approx(0.5)
    assert a_loop == pytest.approx(0.1)
    assert loop < normal                         # 外観に寄るので低くなる


def test_self_loop_alpha_is_ignored_when_the_method_has_one_source() -> None:
    """時空間しか使わない手法では、自己ループでも α=1 のまま。"""
    _, alpha = _fuse(0.9, 0.1, method=M3, self_loop=True)

    assert alpha == pytest.approx(1.0)


# --- 手法による α の振り切り ---------------------------------------------


def test_method_without_appearance_uses_spatiotemporal_only() -> None:
    score, alpha = _fuse(0.8, 0.2, method=M3)

    assert alpha == pytest.approx(1.0)
    assert score == pytest.approx(0.8)


def test_methods_without_spatiotemporal_use_appearance_only() -> None:
    for method in (M1, M2):
        score, alpha = _fuse(0.8, 0.2, method=method)

        assert alpha == pytest.approx(0.0)
        assert score == pytest.approx(0.2)


def test_method_alpha_table() -> None:
    assert method_alpha(M1, is_self_loop=False, params=PARAMS) == 0.0
    assert method_alpha(M2, is_self_loop=True, params=PARAMS) == 0.0
    assert method_alpha(M3, is_self_loop=False, params=PARAMS) == 1.0
    assert method_alpha(M6, is_self_loop=False, params=PARAMS) == 0.5
    assert method_alpha(M6, is_self_loop=True, params=PARAMS) == 0.1


# --- 手法ごとに読む外観の列 ----------------------------------------------


class _Score:
    def __init__(self, s_app=0.6, s_color=0.7, s_sift=0.5):
        self.s_app, self.s_color, self.s_sift = s_app, s_color, s_sift


def test_method_reads_the_right_appearance_column() -> None:
    score = _Score()

    assert app_value(M6, score) == 0.6           # 色 + SIFT
    assert app_value(M4, score) == 0.7           # 色だけ
    assert app_value(M5, score) == 0.5           # SIFT だけ
    assert app_value(M3, score) is None          # 外観を使わない


def test_app_value_without_a_score() -> None:
    assert app_value(M6, None) is None


# --- 欠損 -----------------------------------------------------------------


def test_missing_appearance_falls_back_to_spatiotemporal() -> None:
    score, alpha = _fuse(0.8, None)

    assert alpha == pytest.approx(1.0)
    assert score == pytest.approx(0.8)


def test_missing_appearance_drops_the_candidate_when_alpha_is_zero() -> None:
    """外観しか使わない手法で外観が無ければ、根拠が残らないので落とす。"""
    assert _fuse(0.8, None, method=M2) is None


def test_missing_spatiotemporal_falls_back_to_appearance() -> None:
    score, alpha = _fuse(None, 0.3)

    assert alpha == pytest.approx(0.0)
    assert score == pytest.approx(0.3)


def test_missing_spatiotemporal_drops_the_candidate_for_a_st_only_method() -> None:
    assert _fuse(None, 0.3, method=M3) is None


def test_zero_scores_do_not_produce_infinities() -> None:
    """log(0) を避けるため下限で切る。割当に無限大は渡せない。"""
    score, _ = _fuse(0.0, 0.5)

    assert score >= 0.0
    assert score < 1e-5


# --- 種類ごとに式を変える（dwell_penalty_only） ---------------------------

PENALTY = dict(PARAMS, dwell_penalty_only=True)


def test_penalty_only_keeps_the_appearance_exponent_at_one() -> None:
    """範囲外では S = c^α × S_app。外観の指数を下げない。"""
    score, alpha = fuse(
        0.04, 0.6, kind="out_of_range", is_self_loop=False, method=M6, params=PENALTY
    )

    assert alpha == pytest.approx(0.5)
    assert score == pytest.approx(0.04**0.5 * 0.6)


def test_penalty_only_applies_to_self_loops_too() -> None:
    score, _ = fuse(
        0.135, 0.8, kind="self_loop", is_self_loop=True, method=M6, params=PENALTY
    )

    assert score == pytest.approx(0.135**0.1 * 0.8)      # alpha_self=0.1


def test_in_range_keeps_the_current_formula() -> None:
    score, _ = fuse(
        0.64, 0.49, kind="in_range", is_self_loop=False, method=M6, params=PENALTY
    )

    assert score == pytest.approx(0.56)                  # sqrt(0.64 * 0.49)


def test_penalty_only_preserves_the_appearance_ranking() -> None:
    """定数 c のもとでは、外観の順位がそのまま融合スコアの順位になる。"""
    pairs = [(0.04, 0.3), (0.04, 0.6), (0.04, 0.5)]
    scores = [
        fuse(st, app, kind="out_of_range", is_self_loop=False,
             method=M6, params=PENALTY)[0]
        for st, app in pairs
    ]

    assert sorted(range(3), key=lambda i: scores[i]) == [0, 2, 1]
    # 現行の式でも順位は同じだが、外観の差が (1-α) 乗で縮む
    now = [
        fuse(st, app, kind="out_of_range", is_self_loop=False,
             method=M6, params=PARAMS)[0]
        for st, app in pairs
    ]
    assert (scores[1] - scores[0]) > (now[1] - now[0])


def test_penalty_only_is_inert_without_a_kind() -> None:
    """種類が分からない候補では現行の式に戻す。"""
    a, _ = fuse(0.64, 0.49, is_self_loop=False, method=M6, params=PENALTY)
    b, _ = fuse(0.64, 0.49, is_self_loop=False, method=M6, params=PARAMS)

    assert a == pytest.approx(b)


def test_penalty_only_is_inert_at_the_alpha_extremes() -> None:
    """α=0（外観のみ）と α=1（時空間のみ）では式の違いが出ない。"""
    for params, method in (
        (dict(PENALTY, alpha=0.0, alpha_self=0.0), M6),
        (PENALTY, M3),
    ):
        new, _ = fuse(0.04, 0.6, kind="out_of_range", is_self_loop=False,
                      method=method, params=params)
        now, _ = fuse(0.04, 0.6, kind="out_of_range", is_self_loop=False,
                      method=method, params=dict(params, dwell_penalty_only=False))
        assert new == pytest.approx(now)
