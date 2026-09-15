"""ホモグラフィと単位変換のテスト。

world_points は cm で記入されているので、H を作るときに必ず m に直す。
これを忘れると静止判定の閾値 1.0 m が 1 cm として働き、ほとんどの
トラックが静止扱いになる。単位はここで固定する。
"""

from __future__ import annotations

import numpy as np
import pytest

from mct.config import CameraConfig, Homography, load_cameras
from mct.geometry import (
    compute_homography,
    get_homography,
    reprojection_error,
    to_world,
)

# 3m x 2m の長方形を cm で表したもの
WORLD_CM = [[0, 0], [300, 0], [300, 200], [0, 200]]
IMAGE = [[100, 100], [400, 100], [400, 300], [100, 300]]


def test_cm_world_points_become_meters() -> None:
    H = compute_homography(IMAGE, WORLD_CM, "cm")
    world = to_world(IMAGE, H)

    assert world[0] == pytest.approx([0.0, 0.0], abs=1e-6)
    assert world[1] == pytest.approx([3.0, 0.0], abs=1e-6)   # 300cm -> 3m
    assert world[2] == pytest.approx([3.0, 2.0], abs=1e-6)
    assert world[3] == pytest.approx([0.0, 2.0], abs=1e-6)


def test_meter_world_points_pass_through() -> None:
    world_m = [[x / 100.0, y / 100.0] for x, y in WORLD_CM]
    H = compute_homography(IMAGE, world_m, "m")

    assert to_world(IMAGE, H)[1] == pytest.approx([3.0, 0.0], abs=1e-6)


def test_cm_and_m_give_the_same_result() -> None:
    """同じ配置を cm で書いても m で書いても同じ実空間座標になる。"""
    h_cm = compute_homography(IMAGE, WORLD_CM, "cm")
    h_m = compute_homography(IMAGE, [[x / 100, y / 100] for x, y in WORLD_CM], "m")

    assert to_world(IMAGE, h_cm) == pytest.approx(to_world(IMAGE, h_m), abs=1e-6)


def test_distance_in_world_is_meters() -> None:
    """画像上の2点の実距離がメートルで返ること（静止判定の閾値の前提）。"""
    H = compute_homography(IMAGE, WORLD_CM, "cm")
    a, b = to_world([IMAGE[0], IMAGE[1]], H)

    assert float(np.linalg.norm(b - a)) == pytest.approx(3.0, abs=1e-6)


def test_reprojection_error_is_centimetres() -> None:
    H = compute_homography(IMAGE, WORLD_CM, "cm")
    world_m = np.array(WORLD_CM, dtype=float) / 100.0
    mean, worst = reprojection_error(H, IMAGE, world_m)

    assert mean == pytest.approx(0.0, abs=1e-3)
    assert worst == pytest.approx(0.0, abs=1e-3)


def test_reprojection_error_grows_with_a_bad_point() -> None:
    """対応点を1つずらすと誤差が出る（計測ミスを検出できること）。"""
    H = compute_homography(IMAGE, WORLD_CM, "cm")
    world_m = np.array(WORLD_CM, dtype=float) / 100.0
    shifted = world_m.copy()
    shifted[2] += [0.10, 0.0]                      # 10 cm ずらす
    mean, worst = reprojection_error(H, IMAGE, shifted)

    assert worst == pytest.approx(10.0, abs=0.5)   # cm で返る
    assert mean > 0.0


def test_rejects_unknown_unit() -> None:
    with pytest.raises(ValueError, match="world_unit"):
        compute_homography(IMAGE, WORLD_CM, "mm")


def test_rejects_too_few_points() -> None:
    with pytest.raises(ValueError, match="4点以上"):
        compute_homography(IMAGE[:3], WORLD_CM[:3], "cm")


def test_rejects_mismatched_counts() -> None:
    with pytest.raises(ValueError, match="対応点の数"):
        compute_homography(IMAGE, WORLD_CM[:3], "cm")


def test_to_world_handles_empty_input() -> None:
    H = compute_homography(IMAGE, WORLD_CM, "cm")
    assert to_world([], H).shape == (0, 2)


def _cameras_with_homography() -> dict[str, CameraConfig]:
    return {
        "camA": CameraConfig(
            name="camA",
            homography=Homography(
                world_unit="cm",
                world_points=[list(map(float, p)) for p in WORLD_CM],
                image_points_by_date={"0720": [list(map(float, p)) for p in IMAGE]},
            ),
        )
    }


def test_get_homography_selects_the_date() -> None:
    cameras = _cameras_with_homography()

    assert get_homography(cameras, "camA", "0720") is not None
    assert get_homography(cameras, "camA", "0803") is None   # 日付のエントリ無し
    assert get_homography(cameras, "camZ", "0720") is None   # カメラ無し


def test_get_homography_returns_none_without_measurements() -> None:
    cameras = {"camA": CameraConfig(name="camA")}
    assert get_homography(cameras, "camA", "0720") is None


def test_real_config_reprojection_is_reasonable() -> None:
    """実際の計測点で、再投影誤差が静止判定の閾値より十分小さいこと。"""
    cameras = load_cameras("configs/cameras.yaml")

    for name, config in cameras.items():
        homography = config.homography
        world_m = np.array(homography.world_points, dtype=float) / 100.0
        for date in homography.image_points_by_date:
            H = get_homography(cameras, name, date)
            assert H is not None
            _, worst = reprojection_error(H, homography.image_points(date), world_m)
            # 静止判定は 1.0 m = 100 cm。誤差はその半分未満に収まっていること
            assert worst < 50.0, f"{name}/{date} の最大誤差 {worst:.1f}cm"
