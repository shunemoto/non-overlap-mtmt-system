"""トラックの対応付けと ID 指標のテスト。"""

from __future__ import annotations

import pytest

from mct.evaluation import candidate_recall, evaluate, match_tracks
from mct.gt import GroundTruth, GTBox
from mct.schema import FramePoint, TrackingResult, TrackMeta

PARAMS = {"iou_threshold": 0.5, "match_min_ratio": 0.3}


def _meta() -> TrackMeta:
    return TrackMeta(
        date="0720", camera="camA", width=1920, height=1080, fps=30.0, n_frames=1000,
        model_pose="p", model_seg="s", tracker="t", n_saved_per_track=90,
        created_at="2026-07-20T00:00:00+09:00",
    )


def _pred_track(frames: range, x: float, y: float = 100.0) -> list[FramePoint]:
    return [
        FramePoint(
            frame_idx=f, bbox=[x, y, 50.0, 100.0], conf=0.9,
            foot_point=[x + 25, y + 100], saved=False,
        )
        for f in frames
    ]


def _gt_boxes(frames: range, x: float, y: float = 100.0, not_ignored: bool = True):
    return [
        GTBox(frame_idx=f, bbox=(x, y, 50.0, 100.0), not_ignored=not_ignored, visibility=1.0)
        for f in frames
    ]


def _gt(boxes, global_ids, ignored=()) -> GroundTruth:
    return GroundTruth(
        date="0720", boxes={"camA": boxes},
        global_ids={("camA", t): g for t, g in global_ids.items()},
        ignored={("camA", t) for t in ignored},
    )


def _pred(tracks) -> TrackingResult:
    return TrackingResult(meta=_meta(), tracks=tracks)


# --- 対応付け -------------------------------------------------------------


def test_identical_tracks_match() -> None:
    pred = _pred({"1": _pred_track(range(1, 101), 100.0)})
    gt = _gt({5: _gt_boxes(range(1, 101), 100.0)}, {5: "G1"})

    m = match_tracks(pred, gt, "camA", PARAMS)

    assert m.pairs == {"1": 5}
    assert m.mean_iou["1"] == pytest.approx(1.0)
    assert m.overlap_ratio["1"] == pytest.approx(1.0)
    assert m.unmatched_pred == [] and m.unmatched_gt == []


def test_overlap_below_min_ratio_is_not_matched() -> None:
    """重なりが GT 長の 30% 未満なら対応なし。"""
    pred = _pred({"1": _pred_track(range(1, 21), 100.0)})       # 20 フレーム重なる
    gt = _gt({5: _gt_boxes(range(1, 101), 100.0)}, {5: "G1"})   # GT は 100 フレーム

    m = match_tracks(pred, gt, "camA", PARAMS)

    assert m.pairs == {}
    assert m.unmatched_pred == ["1"]
    assert m.unmatched_gt == [5]


def test_overlap_at_min_ratio_is_matched() -> None:
    pred = _pred({"1": _pred_track(range(1, 31), 100.0)})       # ちょうど 30%
    gt = _gt({5: _gt_boxes(range(1, 101), 100.0)}, {5: "G1"})

    assert match_tracks(pred, gt, "camA", PARAMS).pairs == {"1": 5}


def test_low_iou_does_not_count_as_overlap() -> None:
    """位置がずれていれば共存していても重なりに数えない。"""
    pred = _pred({"1": _pred_track(range(1, 101), 500.0)})
    gt = _gt({5: _gt_boxes(range(1, 101), 100.0)}, {5: "G1"})

    assert match_tracks(pred, gt, "camA", PARAMS).pairs == {}


def test_frames_without_gt_are_not_evaluated() -> None:
    """GT が1つも無いフレームは共存フレームに数えない。

    予測は 1-100、GT は 1-30 にしかない。GT のあるフレームだけを見るので
    重なりは 30/30 = 100% になり、対応が成立する。
    """
    pred = _pred({"1": _pred_track(range(1, 101), 100.0)})
    gt = _gt({5: _gt_boxes(range(1, 31), 100.0)}, {5: "G1"})

    m = match_tracks(pred, gt, "camA", PARAMS)

    assert m.pairs == {"1": 5}
    assert m.overlap_ratio["1"] == pytest.approx(1.0)


def test_assignment_is_one_to_one() -> None:
    pred = _pred({
        "1": _pred_track(range(1, 101), 100.0),
        "2": _pred_track(range(1, 101), 800.0),
    })
    gt = _gt(
        {5: _gt_boxes(range(1, 101), 100.0), 6: _gt_boxes(range(1, 101), 800.0)},
        {5: "G1", 6: "G2"},
    )

    assert match_tracks(pred, gt, "camA", PARAMS).pairs == {"1": 5, "2": 6}


def test_prediction_without_any_gt_is_ignored() -> None:
    """どの GT にも対応付かない予測は FP に数えない。"""
    pred = _pred({
        "1": _pred_track(range(1, 101), 100.0),
        "99": _pred_track(range(1, 101), 1500.0),      # GT に無い
    })
    gt = _gt({5: _gt_boxes(range(1, 101), 100.0)}, {5: "G1"})

    m = match_tracks(pred, gt, "camA", PARAMS)

    assert "99" in m.ignored_pred
    assert m.pairs == {"1": 5}


def test_prediction_matched_to_ignored_gt_is_ignored() -> None:
    pred = _pred({"1": _pred_track(range(1, 101), 100.0)})
    gt = _gt({5: _gt_boxes(range(1, 101), 100.0, not_ignored=False)}, {}, ignored=[5])

    m = match_tracks(pred, gt, "camA", PARAMS)

    assert m.pairs == {"1": 5}
    assert "1" in m.ignored_pred
    assert m.evaluable_pairs(gt) == []


def test_no_gt_tracks_at_all() -> None:
    """GT が空でも落ちず、予測はすべて免責される（camH のような無人カメラ）。"""
    pred = _pred({"1": _pred_track(range(1, 101), 100.0)})
    m = match_tracks(pred, _gt({}, {}), "camA", PARAMS)

    assert m.pairs == {}
    assert m.ignored_pred == {"1"}


# --- 指標 -----------------------------------------------------------------


def test_perfect_assignment_gives_idf1_one() -> None:
    gt = _gt(
        {5: _gt_boxes(range(1, 101), 100.0), 6: _gt_boxes(range(1, 101), 800.0)},
        {5: "G1", 6: "G2"},
    )
    assignments = {"camA": {"5": "G1", "6": "G2"}}

    r = evaluate(assignments, gt, use_gt_tracks=True)

    assert (r.idtp, r.idfp, r.idfn) == (2, 0, 0)
    assert r.idf1 == pytest.approx(1.0)
    assert r.global_id_accuracy == pytest.approx(1.0)


def test_renamed_global_ids_still_perfect() -> None:
    """予測 GID の名前が違っても、1対1に対応していれば正解。"""
    gt = _gt(
        {5: _gt_boxes(range(1, 101), 100.0), 6: _gt_boxes(range(1, 101), 800.0)},
        {5: "G1", 6: "G2"},
    )
    r = evaluate({"camA": {"5": "X", "6": "Y"}}, gt, use_gt_tracks=True)

    assert r.idf1 == pytest.approx(1.0)


def test_swapped_ids_produce_errors() -> None:
    """2人を取り違えると、割り当てで救えるのは片方だけ。"""
    gt = _gt(
        {
            5: _gt_boxes(range(1, 101), 100.0), 6: _gt_boxes(range(1, 101), 300.0),
            7: _gt_boxes(range(1, 101), 800.0),
        },
        {5: "G1", 6: "G1", 7: "G2"},
    )
    # GT では 5,6 が同一人物だが、予測は 6 を別人にしてしまった
    r = evaluate({"camA": {"5": "X", "6": "Y", "7": "Y"}}, gt, use_gt_tracks=True)

    assert r.idtp < 3
    assert r.idfp == r.n_pred_evaluable - r.idtp
    assert r.idfn == r.n_gt_evaluable - r.idtp
    assert r.idf1 < 1.0
    assert r.wrong_merges


def test_merging_two_people_lowers_global_id_accuracy() -> None:
    """2人を1つの予測 GID にまとめたら、どちらも正解にしない。"""
    gt = _gt(
        {5: _gt_boxes(range(1, 101), 100.0), 6: _gt_boxes(range(1, 101), 800.0)},
        {5: "G1", 6: "G2"},
    )
    r = evaluate({"camA": {"5": "X", "6": "X"}}, gt, use_gt_tracks=True)

    assert r.global_id_accuracy == pytest.approx(0.0)
    assert r.idtp == 1


def test_split_person_lowers_global_id_accuracy() -> None:
    """1人を2つの GID に分けてしまうと GID 単位では不正解。"""
    gt = _gt(
        {5: _gt_boxes(range(1, 101), 100.0), 6: _gt_boxes(range(1, 101), 800.0)},
        {5: "G1", 6: "G1"},
    )
    r = evaluate({"camA": {"5": "X", "6": "Y"}}, gt, use_gt_tracks=True)

    assert r.n_gt_global_ids == 1
    assert r.global_id_accuracy == pytest.approx(0.0)


def test_unassigned_track_counts_as_error() -> None:
    gt = _gt(
        {5: _gt_boxes(range(1, 101), 100.0), 6: _gt_boxes(range(1, 101), 800.0)},
        {5: "G1", 6: "G2"},
    )
    r = evaluate({"camA": {"5": "X"}}, gt, use_gt_tracks=True)      # 6 が未割当

    assert r.idtp == 1
    assert r.idfp == 1 and r.idfn == 1


def test_ignored_gt_is_excluded_from_denominators() -> None:
    gt = _gt(
        {
            5: _gt_boxes(range(1, 101), 100.0),
            9: _gt_boxes(range(1, 101), 800.0, not_ignored=False),
        },
        {5: "G1"}, ignored=[9],
    )
    r = evaluate({"camA": {"5": "G1"}}, gt, use_gt_tracks=True)

    assert r.n_gt_evaluable == 1
    assert (r.idtp, r.idfp, r.idfn) == (1, 0, 0)


def test_end_to_end_with_matching() -> None:
    """対応付けを通したうえで IDF1 = 1.0 になること。"""
    pred = _pred({
        "11": _pred_track(range(1, 101), 100.0),
        "22": _pred_track(range(1, 101), 800.0),
    })
    gt = _gt(
        {5: _gt_boxes(range(1, 101), 100.0), 6: _gt_boxes(range(1, 101), 800.0)},
        {5: "G1", 6: "G2"},
    )
    matchings = {"camA": match_tracks(pred, gt, "camA", PARAMS)}
    r = evaluate({"camA": {"11": "X", "22": "Y"}}, gt, matchings)

    assert r.idf1 == pytest.approx(1.0)
    assert len(r.mean_ious) == 2


# --- 候補の再現率 ---------------------------------------------------------


def _two_camera_gt() -> GroundTruth:
    return GroundTruth(
        date="0720",
        boxes={
            "camA": {1: _gt_boxes(range(1, 51), 100.0)},
            "camB": {2: _gt_boxes(range(100, 151), 100.0)},
        },
        global_ids={("camA", 1): "G1", ("camB", 2): "G1"},
    )


def test_candidate_recall_counts_covered_transitions() -> None:
    gt = _two_camera_gt()
    candidates = [{"from": {"cam": "camA", "track": "1"}, "to": {"cam": "camB", "track": "2"}}]

    r = candidate_recall(candidates, gt)

    assert (r.n_gt, r.n_covered) == (1, 1)
    assert r.recall == pytest.approx(1.0)
    assert r.missed == []


def test_candidate_recall_ignores_direction() -> None:
    gt = _two_camera_gt()
    reversed_candidate = [
        {"from": {"cam": "camB", "track": "2"}, "to": {"cam": "camA", "track": "1"}}
    ]

    assert candidate_recall(reversed_candidate, gt).n_covered == 1


def test_candidate_recall_reports_missed() -> None:
    r = candidate_recall([], _two_camera_gt())

    assert (r.n_gt, r.n_covered) == (1, 0)
    assert r.recall == pytest.approx(0.0)
    assert r.missed[0].from_cam == "camA" and r.missed[0].to_cam == "camB"
