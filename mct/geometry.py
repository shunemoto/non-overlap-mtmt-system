"""領域と角度の判定。

観測区間（`exclude_regions`）と遷移領域（`transition_zones`）の適用に使う。
追跡モデルには依存しないので、追跡の実行中でも読み込める。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import cv2
import numpy as np

from mct.config import Region

__all__ = ["point_in_regions", "angle_in_range", "region_center", "direction_deg"]


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
