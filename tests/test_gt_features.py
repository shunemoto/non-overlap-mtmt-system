"""GT からの特徴抽出のうち、モデルを使わない部分のテスト。

推論そのものは torch を読むのでテストしない。観測区間の適用・選抜・座標変換
だけを対象にする。
"""

from __future__ import annotations

import numpy as np
import pytest

from mct.config import CameraConfig, Region
from mct.gt import GroundTruth, GTBox
from mct.gt_features import (
    apply_observation_region,
    build_tracks,
    foot_point,
    keypoints_to_image,
    padded_crop_box,
    select_by_height,
)

PARAMS = {"min_track_len": 31, "n_frames_per_track": 30, "n_bins": 9}


def _box(frame: int, x: float, y: float, w: float = 50.0, h: float = 120.0,
         not_ignored: bool = True) -> GTBox:
    return GTBox(frame_idx=frame, bbox=(x, y, w, h), not_ignored=not_ignored, visibility=1.0)


def _camera(regions: list[Region] | None = None) -> CameraConfig:
    return CameraConfig(
        name="camA", image_size=(1920, 1080), exclude_regions=regions or []
    )


# --- foot_point / 観測区間 ------------------------------------------------


def test_foot_point_is_bottom_centre() -> None:
    assert foot_point(_box(1, 100.0, 200.0, 50.0, 120.0)) == (125.0, 320.0)


def test_observation_region_drops_outside_points() -> None:
    cam = _camera([Region(type="rect", points=[[0, 0], [200, 1080]])])
    # 足元 x = 位置+25。0..150 は除外域、160.. は残る
    boxes = [_box(i + 1, float(i * 10), 500.0) for i in range(40)]

    inside = apply_observation_region(boxes, cam, min_track_len=1)

    assert inside is not None
    assert all(foot_point(b)[0] > 200 for b in inside)
    assert len(inside) == 22


def test_track_below_min_len_is_dropped() -> None:
    cam = _camera([Region(type="rect", points=[[0, 0], [1919, 1080]])])
    boxes = [_box(i + 1, 100.0, 500.0) for i in range(40)]

    assert apply_observation_region(boxes, cam, min_track_len=31) is None


def test_track_at_min_len_is_kept() -> None:
    cam = _camera()
    boxes = [_box(i + 1, 500.0, 500.0) for i in range(31)]

    assert len(apply_observation_region(boxes, cam, min_track_len=31)) == 31


# --- 選抜 -----------------------------------------------------------------


def test_selection_prefers_taller_boxes() -> None:
    """同じ区画の中では bbox の高さが大きいフレームが選ばれる。"""
    boxes = [_box(i + 1, 500.0, 500.0, h=100.0) for i in range(90)]
    boxes[3] = _box(4, 500.0, 500.0, h=900.0)          # 飛び抜けて大きい

    chosen = select_by_height(boxes, n_target=9, n_bins=9)

    assert 3 in chosen


def test_selection_caps_at_target() -> None:
    boxes = [_box(i + 1, 500.0, 500.0, h=100.0 + i) for i in range(200)]

    assert len(select_by_height(boxes, n_target=30, n_bins=9)) == 30


def test_short_track_selects_every_frame() -> None:
    boxes = [_box(i + 1, 500.0, 500.0) for i in range(12)]

    assert select_by_height(boxes, n_target=30, n_bins=9) == list(range(12))


def test_selection_is_spread_over_time() -> None:
    """先頭に大きい箱を固めても、後半の区画から取りこぼさない。"""
    boxes = [_box(i + 1, 500.0, 500.0, h=100.0) for i in range(180)]
    for i in range(20):
        boxes[i] = _box(i + 1, 500.0, 500.0, h=900.0)

    chosen = select_by_height(boxes, n_target=27, n_bins=9)

    assert max(chosen) >= 160        # 最後の区画からも選ばれている


# --- crop と座標変換 ------------------------------------------------------


def test_padded_crop_box_expands_and_clips() -> None:
    # 100x200 の箱を 10% 広げると 120x240、左上は (90, 180)
    assert padded_crop_box((100.0, 200.0, 100.0, 200.0), 0.10, 1920, 1080) == (90, 180, 120, 240)


def test_padded_crop_box_clips_at_image_edge() -> None:
    x, y, w, h = padded_crop_box((0.0, 0.0, 100.0, 200.0), 0.10, 1920, 1080)

    assert (x, y) == (0, 0)
    assert w == 110 and h == 220     # 左と上は切れる


def test_keypoints_are_moved_to_image_coordinates() -> None:
    """crop 座標系の骨格点に crop の原点を足して元画像にそろえる。"""
    kps = np.array([[10.0, 20.0, 0.9], [5.0, 5.0, 0.4]])

    out = keypoints_to_image(kps, (100, 200))

    assert out[0] == [110.0, 220.0, pytest.approx(0.9)]
    assert out[1] == [105.0, 205.0, pytest.approx(0.4)]


def test_undetected_keypoints_stay_at_origin() -> None:
    """信頼度0の点は未検出なので動かさない。"""
    kps = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 0.8]])

    out = keypoints_to_image(kps, (100, 200))

    assert out[0] == [0.0, 0.0, 0.0]
    assert out[1] == [110.0, 210.0, pytest.approx(0.8)]


def test_keypoints_roundtrip_stays_inside_crop() -> None:
    crop_box = padded_crop_box((100.0, 200.0, 50.0, 120.0), 0.10, 1920, 1080)
    kps = np.array([[10.0, 20.0, 0.9]])

    x, y, _ = keypoints_to_image(kps, (crop_box[0], crop_box[1]))[0]

    assert crop_box[0] <= x <= crop_box[0] + crop_box[2]
    assert crop_box[1] <= y <= crop_box[1] + crop_box[3]


# --- build_tracks ---------------------------------------------------------


def _gt(tracks: dict[int, list[GTBox]], global_ids=None) -> GroundTruth:
    return GroundTruth(
        date="0720", boxes={"camA": tracks},
        global_ids={("camA", t): f"P{t:03d}" for t in (global_ids or tracks)},
    )


def test_build_tracks_keeps_long_tracks() -> None:
    gt = _gt({1: [_box(i + 1, 500.0, 500.0) for i in range(100)]})

    kept, selected, stats = build_tracks(gt, "camA", _camera(), PARAMS)

    assert list(kept) == [1]
    assert len(selected[1]) == 30
    assert stats.n_tracks_before == 1 and stats.n_tracks_after == 1


def test_build_tracks_drops_short_tracks() -> None:
    gt = _gt({1: [_box(i + 1, 500.0, 500.0) for i in range(10)]})

    kept, _, stats = build_tracks(gt, "camA", _camera(), PARAMS)

    assert kept == {}
    assert stats.n_tracks_before == 1 and stats.n_tracks_after == 0


def test_build_tracks_counts_removed_points() -> None:
    cam = _camera([Region(type="rect", points=[[0, 0], [200, 1080]])])
    gt = _gt({1: [_box(i + 1, float(i * 10), 500.0) for i in range(60)]})

    _, _, stats = build_tracks(gt, "camA", cam, PARAMS)

    assert stats.n_points_before == 60
    assert stats.n_points_after < 60


def test_build_tracks_skips_ignored() -> None:
    gt = _gt({1: [_box(i + 1, 500.0, 500.0) for i in range(100)]})
    gt.ignored.add(("camA", 1))

    kept, _, stats = build_tracks(gt, "camA", _camera(), PARAMS)

    assert kept == {} and stats.n_tracks_before == 0


def test_build_tracks_on_empty_camera() -> None:
    """GT が空のカメラ（0720/camH のような無人）でも落ちない。"""
    gt = GroundTruth(date="0720", boxes={"camH": {}})

    kept, selected, stats = build_tracks(gt, "camH", _camera(), PARAMS)

    assert kept == {} and selected == {}
    assert stats.n_tracks_before == 0 and stats.n_selected == 0


# --- gt_features: による設定の上書き ------------------------------------


def _params_file(tmp_path, body: str):
    path = tmp_path / "params.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_gt_features_section_overrides_tracking(tmp_path) -> None:
    """GT 側だけマスクの閾値を緩められる。予測側の値は動かさない。"""
    from mct.cli import _gt_feature_params
    from mct.config import tracking_params

    path = _params_file(
        tmp_path,
        "tracking:\n  mask_iou_threshold: 0.30\ngt_features:\n  mask_iou_threshold: 0.15\n",
    )

    assert _gt_feature_params(path)["mask_iou_threshold"] == 0.15
    assert tracking_params(path)["mask_iou_threshold"] == 0.30


def test_gt_features_falls_back_to_tracking(tmp_path) -> None:
    from mct.cli import _gt_feature_params

    path = _params_file(tmp_path, "tracking:\n  mask_iou_threshold: 0.30\n")

    assert _gt_feature_params(path)["mask_iou_threshold"] == 0.30


def test_gt_features_ignores_unknown_keys(tmp_path) -> None:
    """関係ないキーを書いても抽出側の設定には入らない。"""
    from mct.cli import _gt_feature_params

    path = _params_file(tmp_path, "gt_features:\n  conf_threshold: 0.9\n")

    assert "conf_threshold" not in _gt_feature_params(path)
