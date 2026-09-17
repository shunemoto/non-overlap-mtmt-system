"""割当問題と Global ID 付与のテスト。"""

from __future__ import annotations

import collections

import pytest

from mct.assignment import FORBIDDEN, UnionFind, assign_global_ids, build_cost_matrix, solve
from mct.predictions import Link

A1, A2, A3 = ("camA", 1), ("camA", 2), ("camA", 3)
B1, B2 = ("camB", 1), ("camB", 2)
C1 = ("camC", 1)

NO_MATCH = 0.10


def _link(tgt, src) -> Link:
    return Link(tgt[0], str(tgt[1]), src[0], str(src[1]))


# --- コスト行列 -----------------------------------------------------------


def test_matrix_shape_is_n_by_2n() -> None:
    assert build_cost_matrix([A1, A2, B1], {}, NO_MATCH).shape == (3, 6)


def test_dummy_column_is_private_to_its_row() -> None:
    """ダミーを1本にすると、対応なしにできるトラックが1つに限られてしまう。"""
    cost = build_cost_matrix([A1, A2], {}, NO_MATCH)

    assert cost[0, 2] < FORBIDDEN and cost[0, 3] >= FORBIDDEN
    assert cost[1, 3] < FORBIDDEN and cost[1, 2] >= FORBIDDEN


def test_pairs_not_in_the_candidates_are_forbidden() -> None:
    cost = build_cost_matrix([A1, B1], {(A1, B1): 0.9}, NO_MATCH)

    assert cost[0, 1] < FORBIDDEN            # 候補にある
    assert cost[1, 0] >= FORBIDDEN           # 逆向きは候補にない


def test_self_pairs_are_forbidden() -> None:
    cost = build_cost_matrix([A1, B1], {(A1, A1): 0.9}, NO_MATCH)

    assert cost[0, 0] >= FORBIDDEN


def test_higher_scores_cost_less() -> None:
    cost = build_cost_matrix([A1, B1, B2], {(A1, B1): 0.9, (A1, B2): 0.3}, NO_MATCH)

    assert cost[0, 1] < cost[0, 2]


# --- 求解 -----------------------------------------------------------------


def test_a_strong_pair_is_chosen() -> None:
    assert solve([A1, B1], {(A1, B1): 0.9}, NO_MATCH) == [(A1, B1)]


def test_a_weak_pair_loses_to_no_match() -> None:
    assert solve([A1, B1], {(A1, B1): 0.05}, NO_MATCH) == []


def test_each_track_has_at_most_one_successor_and_one_predecessor() -> None:
    scored = {
        (A1, B1): 0.9, (A1, B2): 0.8,        # A1 から2本
        (A2, B1): 0.85,                       # B1 へ2本
    }

    chosen = solve([A1, A2, B1, B2], scored, NO_MATCH)

    assert chosen                                       # 何かは選ばれている
    assert max(collections.Counter(t for t, _ in chosen).values()) == 1
    assert max(collections.Counter(s for _, s in chosen).values()) == 1


def test_several_tracks_can_be_unmatched_at_once() -> None:
    """ダミー列が行ごとに専用なので、複数が同時に未対応になれる。"""
    chosen = solve([A1, A2, A3], {}, NO_MATCH)

    assert chosen == []


def test_the_global_optimum_beats_the_greedy_choice() -> None:
    """貪欲だと最初に 0.9 を取って合計が下がる配置。

    貪欲: A1->B1 (0.9) を取ると A2 は B2 (0.2) しか残らず、積は 0.18。
    全体最適: A1->B2 (0.6) と A2->B1 (0.7) で積 0.42。
    """
    scored = {(A1, B1): 0.9, (A1, B2): 0.6, (A2, B1): 0.7, (A2, B2): 0.2}

    chosen = set(solve([A1, A2, B1, B2], scored, 0.01))

    assert chosen == {(A1, B2), (A2, B1)}


def test_solve_without_tracks() -> None:
    assert solve([], {(A1, B1): 0.9}, NO_MATCH) == []


def test_scores_for_unknown_tracks_are_ignored() -> None:
    assert solve([A1, B1], {(A1, C1): 0.99}, NO_MATCH) == []


# --- Union-Find -----------------------------------------------------------


def test_union_find_merges_transitively() -> None:
    uf = UnionFind()
    uf.union("a", "b")
    uf.union("b", "c")

    assert uf.find("a") == uf.find("c")
    assert len(uf.groups()) == 1


def test_union_find_keeps_separate_groups() -> None:
    uf = UnionFind()
    uf.union("a", "b")
    uf.find("z")

    assert len(uf.groups()) == 2


# --- Global ID ------------------------------------------------------------


def test_three_cameras_share_one_global_id() -> None:
    """C->F->G のような3台にまたがる人物も一貫した ID になる。"""
    links = [_link(A1, B1), _link(B1, C1)]

    assignments = assign_global_ids(links, [A1, B1, C1])

    ids = {assignments["camA"]["1"], assignments["camB"]["1"], assignments["camC"]["1"]}
    assert len(ids) == 1


def test_unmatched_tracks_get_their_own_id() -> None:
    """1台にしか写らなかった人物にも ID を与える。評価はトラック単位のため。"""
    assignments = assign_global_ids([], [A1, B1])

    assert assignments["camA"]["1"] != assignments["camB"]["1"]
    assert len({assignments["camA"]["1"], assignments["camB"]["1"]}) == 2


def test_separate_chains_get_separate_ids() -> None:
    assignments = assign_global_ids([_link(A1, B1)], [A1, B1, A2, B2])

    assert assignments["camA"]["1"] == assignments["camB"]["1"]
    assert assignments["camA"]["2"] != assignments["camA"]["1"]


def test_global_ids_are_zero_padded_and_prefixed() -> None:
    assignments = assign_global_ids([], [A1])

    assert assignments["camA"]["1"] == "G001"


def test_assignment_is_stable_across_runs() -> None:
    tracks = [B2, A1, C1, A2]
    first = assign_global_ids([_link(A1, B2)], tracks)
    second = assign_global_ids([_link(A1, B2)], list(reversed(tracks)))

    assert first == second
