"""領域と角度の判定。

観測区間（`exclude_regions`）と遷移領域（`transition_zones`）の適用に使う。
追跡モデルには依存しないので、追跡の実行中でも読み込める。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import cv2
import numpy as np

from mct.config import CameraConfig, Region

__all__ = [
    "point_in_regions",
    "angle_in_range",
    "region_center",
    "direction_deg",
    "compute_homography",
    "get_homography",
    "to_world",
    "reprojection_error",
]

# world_unit からメートルへの倍率。実空間の計測は cm 単位で記録してある。
UNIT_TO_METER = {"cm": 0.01, "m": 1.0}


def point_in_regions(pt: Sequence[float], regions: Iterable[Region]) -> bool:
    """点がいずれかの領域に入っていれば True。境界上は「入っている」とみなす。

    境界を含めるのは、観測区間の外側を切り取る用途で、境界ちょうどの点が
    どちらに転ぶか分からない状態を避けるため。
    """
    x, y = float(pt[0]), float(pt[1])
    for region in regions:
        if region.type == "rect":
            (x0, y0), (x1, y1) = region.points
            if min(x0, x1) <= x <= max(x0, x1) and min(y0, y1) <= y <= max(y0, y1):
                return True
        else:
            contour = np.asarray(region.points, dtype=np.float32)
            # 戻り値は内側が正・境界が0・外側が負
            if cv2.pointPolygonTest(contour, (x, y), False) >= 0:
                return True
    return False


def angle_in_range(deg: float, lo: float, hi: float) -> bool:
    """角度が [lo, hi] の範囲に入るか。0 をまたぐ区間に対応する。

    たとえば [290, 60] は 290 度から時計回りに 60 度までを指し、350 も 30 も
    含むが 180 は含まない。両端は含む。
    """
    d = deg % 360.0
    lo %= 360.0
    hi %= 360.0
    if lo <= hi:
        return lo <= d <= hi
    return d >= lo or d <= hi


def direction_deg(dx: float, dy: float) -> float:
    """進行方向を度で返す。右向きが 0 度、下向きが 90 度（画像座標系）。"""
    return float(np.degrees(np.arctan2(dy, dx)) % 360.0)


def region_center(region: Region) -> tuple[int, int]:
    """領域の中心。ラベルや扇形の起点に使う。"""
    points = np.asarray(region.points, dtype=np.float32)
    if region.type == "rect":
        return int(points[:, 0].mean()), int(points[:, 1].mean())

    moments = cv2.moments(points)
    if abs(moments["m00"]) < 1e-9:          # 面積0（一直線に並ぶ頂点）
        return int(points[:, 0].mean()), int(points[:, 1].mean())
    return int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])


def compute_homography(
    image_points: Sequence[Sequence[float]],
    world_points: Sequence[Sequence[float]],
    world_unit: str = "cm",
) -> np.ndarray:
    """画像座標を BEV 実空間座標（メートル）に写す 3x3 行列を求める。

    `world_points` は cm 単位で記録されているので、ここで必ずメートルに
    直してから当てはめる。これを忘れると静止判定の閾値 1.0 m が 1 cm として
    働き、ほとんどのトラックが静止扱いになる。単位はテストで固定してある。
    """
    if world_unit not in UNIT_TO_METER:
        raise ValueError(f'world_unit は "cm" か "m" ですが "{world_unit}" でした')
    if len(image_points) != len(world_points):
        raise ValueError(
            f"対応点の数が違います: image {len(image_points)} / world {len(world_points)}"
        )
    if len(world_points) < 4:
        raise ValueError(f"ホモグラフィには4点以上必要です（{len(world_points)} 点）")

    img = np.asarray(image_points, dtype=np.float64)
    world = np.asarray(world_points, dtype=np.float64) * UNIT_TO_METER[world_unit]

    H, _ = cv2.findHomography(img, world, cv2.RANSAC, 5.0)
    if H is None:
        raise ValueError("ホモグラフィを計算できませんでした（対応点を確認してください）")
    return H


def get_homography(
    cameras: dict[str, CameraConfig], cam: str, date: str
) -> np.ndarray | None:
    """指定したカメラ・日付のホモグラフィ。未設定なら None。

    カメラは日付ごとに画角がわずかにずれているので、画像側の対応点だけを
    日付で切り替える。
    """
    config = cameras.get(cam)
    if config is None:
        return None
    homography = config.homography
    image_points = homography.image_points(date)
    if not image_points or not homography.world_points:
        return None
    return compute_homography(image_points, homography.world_points, homography.world_unit)


def to_world(points: Sequence[Sequence[float]], H: np.ndarray) -> np.ndarray:
    """画像座標の並びを BEV 実空間座標（メートル）に変換する。"""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    if len(pts) == 0:
        return np.empty((0, 2), dtype=np.float64)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def reprojection_error(
    H: np.ndarray,
    image_points: Sequence[Sequence[float]],
    world_points_m: Sequence[Sequence[float]],
) -> tuple[float, float]:
    """計測点の再投影誤差（平均, 最大）を cm で返す。

    誤差が大きければ計測点の取り違えを疑える。`world_points_m` は
    メートル単位であることに注意（cm のまま渡すと 100 倍の誤差に見える）。
    """
    projected = to_world(image_points, H)
    world = np.asarray(world_points_m, dtype=np.float64)
    errors = np.linalg.norm(projected - world, axis=1) * 100.0   # m -> cm
    return float(errors.mean()), float(errors.max())
