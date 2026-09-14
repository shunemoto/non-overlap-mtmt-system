"""領域判定と角度判定のテスト。"""

from __future__ import annotations

import pytest

from mct.config import Region
from mct.geometry import angle_in_range, direction_deg, point_in_regions, region_center

RECT = Region(type="rect", points=[[100.0, 100.0], [300.0, 200.0]])
TRIANGLE = Region(type="polygon", points=[[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]])


def test_point_inside_rect() -> None:
    assert point_in_regions((200, 150), [RECT])
    assert point_in_regions((101, 101), [RECT])


def test_point_outside_rect() -> None:
    assert not point_in_regions((99, 150), [RECT])
    assert not point_in_regions((200, 201), [RECT])
    assert not point_in_regions((301, 150), [RECT])


def test_rect_boundary_counts_as_inside() -> None:
    for pt in [(100, 100), (300, 200), (100, 150), (200, 200)]:
        assert point_in_regions(pt, [RECT]), pt


def test_rect_accepts_reversed_corners() -> None:
    """右下・左上の順で書かれていても同じ矩形として扱う。"""
    reversed_rect = Region(type="rect", points=[[300.0, 200.0], [100.0, 100.0]])
    assert point_in_regions((200, 150), [reversed_rect])
    assert not point_in_regions((50, 150), [reversed_rect])


def test_point_inside_polygon() -> None:
    assert point_in_regions((10, 10), [TRIANGLE])
    assert not point_in_regions((90, 90), [TRIANGLE])   # 斜辺の外側


def test_polygon_boundary_counts_as_inside() -> None:
    for pt in [(0, 0), (100, 0), (0, 100), (50, 0), (50, 50)]:
        assert point_in_regions(pt, [TRIANGLE]), pt


def test_empty_regions_never_match() -> None:
    assert not point_in_regions((10, 10), [])


def test_matches_any_of_several_regions() -> None:
    regions = [RECT, TRIANGLE]
    assert point_in_regions((10, 10), regions)      # 多角形のほう
    assert point_in_regions((200, 150), regions)    # 矩形のほう
    assert not point_in_regions((800, 800), regions)


def test_angle_in_simple_range() -> None:
    assert angle_in_range(150, 110, 240)
    assert not angle_in_range(90, 110, 240)
    assert not angle_in_range(250, 110, 240)


def test_angle_in_wrapping_range() -> None:
    """[290, 60] は 0 をまたぐ区間。"""
    assert angle_in_range(350, 290, 60)
    assert angle_in_range(30, 290, 60)
    assert not angle_in_range(180, 290, 60)
    assert not angle_in_range(100, 290, 60)


def test_angle_range_includes_endpoints() -> None:
    assert angle_in_range(290, 290, 60)
    assert angle_in_range(60, 290, 60)
    assert angle_in_range(110, 110, 240)
    assert angle_in_range(240, 110, 240)


def test_angle_normalizes_out_of_turn_values() -> None:
    assert angle_in_range(-10, 290, 60)     # -10 == 350
    assert angle_in_range(360, 290, 60)     # 360 == 0
    assert angle_in_range(390, 290, 60)     # 390 == 30


def test_direction_deg_uses_image_axes() -> None:
    """右向きが0度、下向きが90度（y 軸が下向きの画像座標系）。"""
    assert direction_deg(1, 0) == pytest.approx(0.0)
    assert direction_deg(0, 1) == pytest.approx(90.0)
    assert direction_deg(-1, 0) == pytest.approx(180.0)
    assert direction_deg(0, -1) == pytest.approx(270.0)


def test_region_center() -> None:
    assert region_center(RECT) == (200, 150)
    cx, cy = region_center(TRIANGLE)
    assert (cx, cy) == (33, 33)     # 三角形の重心
