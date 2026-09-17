"""外観記述子の抽出のテスト。

合成画像だけを使い、モデルも実データも読まない。crop・マスク・骨格点は
すでにファイルにあるので、ここで試すのは分割・ヒストグラム・パッチの
切り出しと集約だけになる。
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from mct.appearance import (
    N_JOINTS,
    AppearanceDescriptor,
    FrameFeature,
    aggregate,
    appearance_params,
    color_histogram,
    joint_points,
    patch_window,
    sift_features,
    split_y,
)

PARAMS = {
    "hist_bins": [12, 12],
    "achromatic_threshold": 0.15,
    "sift_patch_ratio": 0.07,
    "sift_patch_size": 32,
    "sift_min_bbox_height": 200,
    "sift_min_gradient": 0.01,
    "sift_min_frames": 5,
    "lambda_missing": 0.9,
}

L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12


def _keypoints(**overrides: tuple[float, float, float]) -> list[list[float]]:
    """COCO 17点。指定しなかった点は未検出（信頼度0）にする。"""
    kps = [[0.0, 0.0, 0.0] for _ in range(17)]
    for index, value in overrides.items():
        kps[int(index[1:])] = list(value)
    return kps


# --- 上下分割 -------------------------------------------------------------


def test_split_uses_hip_midpoint() -> None:
    kps = _keypoints(i5=(100, 200, 0.9), i6=(140, 200, 0.9),
                     i11=(105, 300, 0.9), i12=(135, 320, 0.9))

    line, source = split_y(kps, [100.0, 100.0, 60.0, 400.0])

    assert source == "hip"
    assert line == pytest.approx(310.0)


def test_split_falls_back_to_shoulder() -> None:
    """腰が取れなければ、肩から bbox 下辺までの 45% を腰とみなす。"""
    kps = _keypoints(i5=(100, 200, 0.9), i6=(140, 200, 0.9))

    line, source = split_y(kps, [100.0, 100.0, 60.0, 400.0])

    assert source == "shoulder"
    # 肩 200、下辺 500 -> 200 + 300*0.45
    assert line == pytest.approx(335.0)


def test_split_falls_back_to_bbox_centre() -> None:
    line, source = split_y(_keypoints(), [100.0, 100.0, 60.0, 400.0])

    assert source == "bbox"
    assert line == pytest.approx(300.0)


def test_split_without_keypoints_uses_bbox() -> None:
    line, source = split_y(None, [0.0, 0.0, 50.0, 200.0])

    assert source == "bbox" and line == pytest.approx(100.0)


def test_shoulder_fallback_needs_shoulder_above_bottom() -> None:
    """肩が bbox 下辺より下にある壊れた骨格点では bbox に落ちる。"""
    kps = _keypoints(i5=(100, 900, 0.9), i6=(140, 900, 0.9))

    _, source = split_y(kps, [100.0, 100.0, 60.0, 400.0])

    assert source == "bbox"


# --- 色ヒストグラム -------------------------------------------------------


def _solid(bgr: tuple[int, int, int], size: int = 20) -> np.ndarray:
    return np.full((size, size, 3), bgr, dtype=np.uint8)


def test_low_saturation_goes_to_achromatic_bin() -> None:
    """彩度の低い灰色は無彩色ビン（末尾）に集まる。"""
    upper, _ = color_histogram(_solid((128, 128, 128)), None, 20, PARAMS)

    assert upper[-1] == pytest.approx(1.0)
    assert sum(upper[:-1]) == pytest.approx(0.0)


def test_high_saturation_stays_in_grid() -> None:
    upper, _ = color_histogram(_solid((0, 0, 255)), None, 20, PARAMS)

    assert upper[-1] == pytest.approx(0.0)
    assert sum(upper[:-1]) == pytest.approx(1.0)


def test_histogram_is_l1_normalised_and_sized() -> None:
    rng = np.random.default_rng(0)
    crop = rng.integers(0, 256, (40, 20, 3), dtype=np.uint8)

    upper, lower = color_histogram(crop, None, 20, PARAMS)

    assert len(upper) == 12 * 12 + 1
    assert sum(upper) == pytest.approx(1.0)
    assert sum(lower) == pytest.approx(1.0)


def test_histogram_respects_the_split_row() -> None:
    """分割行より上と下で別々のヒストグラムになる。"""
    crop = np.vstack([_solid((0, 0, 255), 20), _solid((255, 0, 0), 20)])

    upper, lower = color_histogram(crop, None, 20, PARAMS)

    assert np.argmax(upper) != np.argmax(lower)


def test_mask_restricts_the_pixels_used() -> None:
    crop = np.hstack([_solid((0, 0, 255), 20), _solid((255, 0, 0), 20)])
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    mask[:, :20] = 255                      # 左半分だけ人物

    masked, _ = color_histogram(crop, mask, 20, PARAMS)
    whole, _ = color_histogram(crop, None, 20, PARAMS)

    assert np.argmax(masked) != np.argmax(whole) or masked[np.argmax(masked)] > whole[np.argmax(whole)]
    assert masked[np.argmax(masked)] == pytest.approx(1.0)


def test_achromatic_threshold_can_be_disabled() -> None:
    """閾値を None にすると、灰色も色相の格子に入る。"""
    params = dict(PARAMS, achromatic_threshold=None)

    upper, _ = color_histogram(_solid((128, 128, 128)), None, 20, params)

    assert upper[-1] == pytest.approx(0.0)
    assert sum(upper[:-1]) == pytest.approx(1.0)


def test_empty_half_returns_none() -> None:
    """分割行が端に寄って画素が無い側は None になる。"""
    upper, lower = color_histogram(_solid((0, 0, 255)), None, 0, PARAMS)

    assert upper is None
    assert lower is not None


# --- パッチの切り出し -----------------------------------------------------


def test_patch_window_is_centred_when_it_fits() -> None:
    assert patch_window(50.0, 60.0, 10.0, 200, 200) == (40, 50, 20, 20)


def test_patch_window_slides_inside_instead_of_padding() -> None:
    """crop からはみ出すときは窓ごと内側へずらし、寸法を保つ。"""
    x, y, w, h = patch_window(3.0, 3.0, 10.0, 200, 200)

    assert (x, y) == (0, 0)
    assert (w, h) == (20, 20)

    x, y, w, h = patch_window(198.0, 198.0, 10.0, 200, 200)

    assert (x, y) == (180, 180)
    assert (w, h) == (20, 20)


def test_patch_window_rejects_windows_larger_than_the_crop() -> None:
    assert patch_window(50.0, 50.0, 200.0, 100, 100) is None


# --- 関節点 ---------------------------------------------------------------


def test_joint_points_derives_neck_and_torso() -> None:
    kps = _keypoints(i5=(100, 200, 0.9), i6=(140, 200, 0.9),
                     i11=(100, 400, 0.9), i12=(140, 400, 0.9))

    points = joint_points(kps)

    assert len(points) == N_JOINTS
    assert points[4] == pytest.approx((120.0, 200.0))     # 首元 = 肩の中点
    assert points[5] == pytest.approx((120.0, 300.0))     # 胴体中心


def test_derived_points_need_every_source_joint() -> None:
    kps = _keypoints(i5=(100, 200, 0.9), i6=(140, 200, 0.9), i11=(100, 400, 0.9))

    points = joint_points(kps)

    assert points[4] is not None          # 首元は肩2点だけで作れる
    assert points[5] is None              # 胴体中心は腰が片方欠けると作れない


# --- SIFT -----------------------------------------------------------------


def _textured_crop(size: int = 400) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, (size, size), dtype=np.uint8)


def _full_keypoints(centre: float = 200.0) -> list[list[float]]:
    return _keypoints(
        i5=(centre - 30, centre - 50, 0.9), i6=(centre + 30, centre - 50, 0.9),
        i11=(centre - 30, centre + 50, 0.9), i12=(centre + 30, centre + 50, 0.9),
    )


def _stripes(size: int = 400) -> np.ndarray:
    """向きのはっきりした縞。回転させれば記述子が変わるはずの模様。"""
    rows = np.arange(size)[:, None]
    return ((np.sin(rows / 4.0) * 0.5 + 0.5) * 255).astype(np.uint8).repeat(size, axis=1)


def test_sift_is_computed_upright() -> None:
    """縞を90度回すと記述子が変わる = 主方向を推定していない。

    `detectAndCompute` のように主方向を当てていれば、模様を回しても記述子は
    ほぼ変わらない。回転で変わることが upright の証拠になる。等方的な
    ランダムノイズでは回転しても統計が変わらず判定できないので、向きの
    ある模様を使う。
    """
    kps = _full_keypoints()
    box, bbox = [0.0, 0.0, 400.0, 400.0], [0.0, 0.0, 200.0, 400.0]

    straight, _ = sift_features(_stripes(), kps, box, bbox, PARAMS)
    rotated, _ = sift_features(np.rot90(_stripes()).copy(), kps, box, bbox, PARAMS)

    # どちらも中身のある記述子であることを確かめてから比べる
    assert float(np.linalg.norm(straight[0])) == pytest.approx(1.0)
    assert float(np.linalg.norm(rotated[0])) == pytest.approx(1.0)
    assert float(np.dot(straight[0], rotated[0])) < 0.5


def test_flat_patch_has_no_gradient() -> None:
    crop = np.full((400, 400), 128, dtype=np.uint8)

    descriptors, gradients = sift_features(
        crop, _full_keypoints(), [0.0, 0.0, 400.0, 400.0],
        [0.0, 0.0, 200.0, 400.0], PARAMS,
    )

    assert all(g == pytest.approx(0.0) for g in gradients)
    assert all(d is None for d in descriptors)


def test_small_bbox_skips_sift() -> None:
    descriptors, gradients = sift_features(
        _textured_crop(), _full_keypoints(), [0.0, 0.0, 400.0, 400.0],
        [0.0, 0.0, 100.0, 150.0], PARAMS,       # 高さ 150 < 200
    )

    assert all(d is None for d in descriptors)
    assert all(g == 0.0 for g in gradients)


def test_sift_descriptors_are_l2_normalised() -> None:
    descriptors, _ = sift_features(
        _textured_crop(), _full_keypoints(), [0.0, 0.0, 400.0, 400.0],
        [0.0, 0.0, 200.0, 400.0], PARAMS,
    )

    found = [d for d in descriptors if d is not None]
    assert found
    for d in found:
        assert float(np.linalg.norm(d)) == pytest.approx(1.0)


# --- 集約 -----------------------------------------------------------------


def _frame(hist_value: float, sift_value: float, gradient: float) -> FrameFeature:
    hist = np.full(145, 0.0)
    hist[0] = hist_value
    hist[1] = 1.0 - hist_value
    vector = np.zeros(128)
    vector[0] = sift_value
    vector[1] = np.sqrt(max(0.0, 1.0 - sift_value**2))
    return FrameFeature(
        hist_upper=hist, hist_lower=hist,
        sift=[vector] * N_JOINTS, gradients=[gradient] * N_JOINTS,
        split_source="hip", has_mask=True,
    )


def test_aggregate_uses_the_median_of_each_bin() -> None:
    """外れた1フレームに引きずられない。"""
    frames = [_frame(0.5, 1.0, 1.0) for _ in range(5)] + [_frame(0.0, 1.0, 1.0)]

    descriptor = aggregate("camA", 1, frames, PARAMS)

    assert descriptor.hist_upper[0] == pytest.approx(0.5)
    assert sum(descriptor.hist_upper) == pytest.approx(1.0)


def test_low_gradient_joints_become_invalid() -> None:
    frames = [_frame(0.5, 1.0, 0.001) for _ in range(30)]

    descriptor = aggregate("camA", 1, frames, PARAMS)

    assert descriptor.sift_valid == [False] * N_JOINTS
    assert descriptor.sift is None


def test_joint_needs_min_frames() -> None:
    frames = [_frame(0.5, 1.0, 1.0) for _ in range(4)]      # sift_min_frames=5

    descriptor = aggregate("camA", 1, frames, PARAMS)

    assert descriptor.n_valid_joints == 0


def test_joint_at_min_frames_is_valid() -> None:
    frames = [_frame(0.5, 1.0, 1.0) for _ in range(5)]

    descriptor = aggregate("camA", 1, frames, PARAMS)

    assert descriptor.n_valid_joints == N_JOINTS
    assert float(np.linalg.norm(descriptor.sift[0])) == pytest.approx(1.0)


def test_aggregate_without_frames() -> None:
    descriptor = aggregate("camA", 1, [], PARAMS)

    assert descriptor.hist_upper is None and descriptor.sift is None
    assert descriptor.n_frames_used == 0


# --- 直列化 ---------------------------------------------------------------


def test_descriptor_roundtrip() -> None:
    original = AppearanceDescriptor(
        camera="camA", track_id=3,
        hist_upper=[0.5, 0.5], hist_lower=None,
        sift=[[1.0] + [0.0] * 127] * N_JOINTS,
        sift_valid=[True] * N_JOINTS, n_frames_used=30,
    )

    restored = AppearanceDescriptor.from_dict(original.to_dict())

    assert restored.key == ("camA", 3)
    assert restored.hist_lower is None
    assert restored.n_valid_joints == N_JOINTS


def test_params_merge_defaults_and_overrides(tmp_path) -> None:
    path = tmp_path / "params.yaml"
    path.write_text("appearance:\n  sift_min_gradient: 0.05\n", encoding="utf-8")

    params = appearance_params(path, lambda_missing=0.5)

    assert params["sift_min_gradient"] == 0.05      # YAML
    assert params["hist_bins"] == [12, 12]          # 既定値
    assert params["lambda_missing"] == 0.5          # 明示的な上書き
