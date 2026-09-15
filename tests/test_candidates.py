"""候補列挙のテスト。"""

from __future__ import annotations

import pytest

from mct.candidates import enumerate_candidates
from mct.config import TransitionRoute
from mct.summarize import TrackSummary

PARAMS = {"fps": 30, "self_loop_dir_tol": 60}


def _summary(
    camera: str, track_id: int, start: int, end: int,
    entry_zones=(), exit_zones=(), entry_dir=None, exit_dir=None,
) -> TrackSummary:
    return TrackSummary(
        date="0720", camera=camera, track_id=track_id,
        start_frame=start, end_frame=end, n_points=end - start + 1,
        entry_point=[0.0, 0.0], exit_point=[0.0, 0.0],
        entry_dir_deg=entry_dir, exit_dir_deg=exit_dir,
        entry_zones=list(entry_zones), exit_zones=list(exit_zones),
    )


def _route(
    from_cam: str, from_zone: str, to_cam: str, to_zone: str,
    lo: float = 1.0, hi: float | None = 20.0,
) -> TransitionRoute:
    return TransitionRoute(
        from_cam=from_cam, from_zone=from_zone, to_cam=to_cam, to_zone=to_zone,
        distance_m=10.0, search_range_sec=(lo, hi), c_dwell=0.1,
    )


A_TO_B = _route("camA", "a_out", "camB", "b_in")


def test_basic_candidate_is_found() -> None:
    """tgt が先に退出し、src が後から出現する対が候補になる。"""
    tgt = _summary("camA", 1, 100, 200, exit_zones=["a_out"])
    src = _summary("camB", 2, 400, 500, entry_zones=["b_in"])

    cands = enumerate_candidates([tgt, src], [A_TO_B], PARAMS)

    assert len(cands) == 1
    assert cands[0].tgt == ("camA", 1) and cands[0].src == ("camB", 2)
    assert cands[0].tgt_zone == "a_out" and cands[0].src_zone == "b_in"
    assert cands[0].delta_frames == 200
    assert cands[0].in_search_range


def test_direction_of_route_matters() -> None:
    """transitions の from は tgt 側、to は src 側。逆向きでは成立しない。"""
    # camB から出て camA に現れる対は、A->B の経路では拾えない
    tgt = _summary("camB", 1, 100, 200, exit_zones=["b_in"])
    src = _summary("camA", 2, 400, 500, entry_zones=["a_out"])

    assert enumerate_candidates([tgt, src], [A_TO_B], PARAMS) == []


def test_time_order_is_required() -> None:
    """src が先に出現していれば候補にしない。"""
    tgt = _summary("camA", 1, 400, 500, exit_zones=["a_out"])
    src = _summary("camB", 2, 100, 200, entry_zones=["b_in"])

    assert enumerate_candidates([tgt, src], [A_TO_B], PARAMS) == []


def test_overlapping_intervals_are_rejected() -> None:
    """同時に写っていれば同一人物ではあり得ない。"""
    tgt = _summary("camA", 1, 100, 450, exit_zones=["a_out"])
    src = _summary("camB", 2, 400, 500, entry_zones=["b_in"])

    assert enumerate_candidates([tgt, src], [A_TO_B], PARAMS) == []


def test_below_minimum_time_is_rejected() -> None:
    """下限（1秒 = 30フレーム）に満たない対は候補にしない。"""
    tgt = _summary("camA", 1, 100, 200, exit_zones=["a_out"])
    src = _summary("camB", 2, 210, 300, entry_zones=["b_in"])   # delta = 10

    assert enumerate_candidates([tgt, src], [A_TO_B], PARAMS) == []


def test_above_maximum_time_is_kept_and_flagged() -> None:
    """上限を超えた対も捨てず、印だけ付けて残す。"""
    tgt = _summary("camA", 1, 100, 200, exit_zones=["a_out"])
    src = _summary("camB", 2, 2000, 2100, entry_zones=["b_in"])  # 上限 20 秒 = 600

    cands = enumerate_candidates([tgt, src], [A_TO_B], PARAMS)

    assert len(cands) == 1
    assert cands[0].in_search_range is False


def test_open_ended_range_is_always_in_range() -> None:
    route = _route("camA", "a_out", "camB", "b_in", lo=1.0, hi=None)
    tgt = _summary("camA", 1, 100, 200, exit_zones=["a_out"])
    src = _summary("camB", 2, 9000, 9100, entry_zones=["b_in"])

    assert enumerate_candidates([tgt, src], [route], PARAMS)[0].in_search_range


def test_zone_mismatch_is_rejected() -> None:
    tgt = _summary("camA", 1, 100, 200, exit_zones=["other"])
    src = _summary("camB", 2, 400, 500, entry_zones=["b_in"])

    assert enumerate_candidates([tgt, src], [A_TO_B], PARAMS) == []


def test_track_without_zone_is_rejected() -> None:
    tgt = _summary("camA", 1, 100, 200, exit_zones=[])
    src = _summary("camB", 2, 400, 500, entry_zones=["b_in"])

    assert enumerate_candidates([tgt, src], [A_TO_B], PARAMS) == []


def test_multiple_zones_produce_multiple_candidates() -> None:
    """トラックが複数ゾーンに該当していれば、経路ごとに候補が立つ。"""
    routes = [A_TO_B, _route("camA", "a_alt", "camB", "b_in")]
    tgt = _summary("camA", 1, 100, 200, exit_zones=["a_out", "a_alt"])
    src = _summary("camB", 2, 400, 500, entry_zones=["b_in"])

    cands = enumerate_candidates([tgt, src], routes, PARAMS)

    assert {c.tgt_zone for c in cands} == {"a_out", "a_alt"}


# --- 自己ループ -----------------------------------------------------------

SELF = _route("camA", "a_out", "camA", "a_out", lo=1.0, hi=None)


def test_self_loop_requires_reversed_direction() -> None:
    """出た向きと戻ってきた向きが逆なら候補になる。"""
    tgt = _summary("camA", 1, 100, 200, exit_zones=["a_out"], exit_dir=0.0)
    src = _summary("camA", 2, 1000, 1100, entry_zones=["a_out"], entry_dir=180.0)

    cands = enumerate_candidates([tgt, src], [SELF], PARAMS)

    assert len(cands) == 1 and cands[0].is_self_loop


def test_self_loop_rejects_same_direction() -> None:
    """同じ向きのままなら、別人が同じ経路を通っただけとみなす。"""
    tgt = _summary("camA", 1, 100, 200, exit_zones=["a_out"], exit_dir=0.0)
    src = _summary("camA", 2, 1000, 1100, entry_zones=["a_out"], entry_dir=10.0)

    assert enumerate_candidates([tgt, src], [SELF], PARAMS) == []


def test_self_loop_within_tolerance() -> None:
    """180度から許容差（60度）以内のずれは認める。"""
    tgt = _summary("camA", 1, 100, 200, exit_zones=["a_out"], exit_dir=0.0)
    src = _summary("camA", 2, 1000, 1100, entry_zones=["a_out"], entry_dir=130.0)

    assert len(enumerate_candidates([tgt, src], [SELF], PARAMS)) == 1


def test_self_loop_outside_tolerance() -> None:
    tgt = _summary("camA", 1, 100, 200, exit_zones=["a_out"], exit_dir=0.0)
    src = _summary("camA", 2, 1000, 1100, entry_zones=["a_out"], entry_dir=110.0)

    assert enumerate_candidates([tgt, src], [SELF], PARAMS) == []


def test_self_loop_without_direction_is_rejected() -> None:
    """向きが取れないトラックは反転を確かめられないので候補にしない。"""
    tgt = _summary("camA", 1, 100, 200, exit_zones=["a_out"], exit_dir=None)
    src = _summary("camA", 2, 1000, 1100, entry_zones=["a_out"], entry_dir=180.0)

    assert enumerate_candidates([tgt, src], [SELF], PARAMS) == []


def test_track_is_not_paired_with_itself() -> None:
    one = _summary("camA", 1, 100, 200, entry_zones=["a_out"], exit_zones=["a_out"],
                   entry_dir=180.0, exit_dir=0.0)

    assert enumerate_candidates([one], [SELF], PARAMS) == []


# --- 絞り込みなしモード ---------------------------------------------------


def test_no_topology_ignores_zones_and_routes() -> None:
    """ゾーンも経路も見ず、時間順序と区間非重複だけで列挙する。"""
    a = _summary("camA", 1, 100, 200)
    b = _summary("camB", 2, 400, 500)
    c = _summary("camC", 3, 700, 800)

    cands = enumerate_candidates([a, b, c], [], PARAMS, no_topology=True)

    # 3トラックが時間的に重ならないので 3C2 = 3 対
    assert len(cands) == 3
    assert all(c.route is None for c in cands)


def test_no_topology_still_rejects_overlap() -> None:
    a = _summary("camA", 1, 100, 450)
    b = _summary("camB", 2, 400, 500)

    assert enumerate_candidates([a, b], [], PARAMS, no_topology=True) == []


def test_no_topology_yields_more_than_with_topology() -> None:
    summaries = [
        _summary("camA", 1, 100, 200, exit_zones=["a_out"]),
        _summary("camB", 2, 400, 500, entry_zones=["b_in"]),
        _summary("camC", 3, 700, 800),
    ]
    with_topo = enumerate_candidates(summaries, [A_TO_B], PARAMS)
    without = enumerate_candidates(summaries, [A_TO_B], PARAMS, no_topology=True)

    assert len(without) > len(with_topo)


def test_as_link_is_ordered_by_time() -> None:
    """candidate_recall が読む形は from=tgt（先）, to=src（後）。"""
    tgt = _summary("camA", 1, 100, 200, exit_zones=["a_out"])
    src = _summary("camB", 2, 400, 500, entry_zones=["b_in"])

    link = enumerate_candidates([tgt, src], [A_TO_B], PARAMS)[0].as_link()

    assert link["from"] == {"cam": "camA", "track": 1}
    assert link["to"] == {"cam": "camB", "track": 2}
