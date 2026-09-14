"""cameras.yaml の読み込みと検証のテスト。

定義は手作業で書いていくので、「途中の状態でも読める」ことと
「明らかな誤りは分かりやすく落ちる」ことの両方を確かめる。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mct.config import CameraConfigError, load_cameras

REPO_ROOT = Path(__file__).resolve().parent.parent


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "cameras.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_loads_repository_cameras_yaml() -> None:
    """リポジトリに入っている定義がそのまま読めること。"""
    cameras = load_cameras(REPO_ROOT / "configs" / "cameras.yaml")

    assert set(cameras) == {f"cam{c}" for c in "ABCDEFGH"}
    cam_a = cameras["camA"]
    assert cam_a.image_size == (1920, 1080)
    assert len(cam_a.exclude_regions) == 2
    assert [r.type for r in cam_a.exclude_regions] == ["rect", "polygon"]

    zone = cam_a.transition_zones[0]
    assert zone.id == "a_corridor"
    assert zone.exit_dir == (290.0, 60.0)
    assert zone.entry_dir == (110.0, 240.0)
    assert [(t.cam, t.room) for t in zone.to] == [("camB", None), ("camA", "design_room")]


def test_undefined_cameras_load_as_empty() -> None:
    cam_b = load_cameras(REPO_ROOT / "configs" / "cameras.yaml")["camB"]

    assert cam_b.exclude_regions == []
    assert cam_b.transition_zones == []
    assert cam_b.homography.image_points == []


def test_camera_with_no_body_at_all(tmp_path: Path) -> None:
    cameras = load_cameras(_write(tmp_path, "cameras:\n  camA:\n"))

    assert cameras["camA"].image_size is None
    assert cameras["camA"].exclude_regions == []


def test_empty_exclude_regions(tmp_path: Path) -> None:
    body = """
cameras:
  camA:
    image_size: [1920, 1080]
    exclude_regions:
    transition_zones:
    homography:
"""
    cameras = load_cameras(_write(tmp_path, body))

    assert cameras["camA"].exclude_regions == []
    assert cameras["camA"].transition_zones == []


def test_empty_cameras_key(tmp_path: Path) -> None:
    assert load_cameras(_write(tmp_path, "cameras:\n")) == {}


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_cameras("does/not/exist.yaml")


def test_rect_needs_exactly_two_points(tmp_path: Path) -> None:
    body = """
cameras:
  camA:
    image_size: [1920, 1080]
    exclude_regions:
      - type: rect
        points: [[0, 0], [100, 100], [200, 200]]
"""
    with pytest.raises(CameraConfigError) as e:
        load_cameras(_write(tmp_path, body))
    assert "camA" in str(e.value) and "2点" in str(e.value)


def test_polygon_needs_three_points(tmp_path: Path) -> None:
    body = """
cameras:
  camA:
    image_size: [1920, 1080]
    exclude_regions:
      - type: polygon
        points: [[0, 0], [100, 100]]
"""
    with pytest.raises(CameraConfigError) as e:
        load_cameras(_write(tmp_path, body))
    assert "camA" in str(e.value) and "3点以上" in str(e.value)


def test_unknown_region_type(tmp_path: Path) -> None:
    body = """
cameras:
  camA:
    exclude_regions:
      - type: circle
        points: [[0, 0], [10, 10]]
"""
    with pytest.raises(CameraConfigError, match="circle"):
        load_cameras(_write(tmp_path, body))


def test_point_outside_image_size(tmp_path: Path) -> None:
    body = """
cameras:
  camA:
    image_size: [1920, 1080]
    exclude_regions:
      - type: rect
        points: [[0, 0], [1920, 1200]]
"""
    with pytest.raises(CameraConfigError) as e:
        load_cameras(_write(tmp_path, body))
    assert "camA" in str(e.value) and "image_size" in str(e.value)


def test_bounds_not_checked_without_image_size(tmp_path: Path) -> None:
    """image_size 未記入なら範囲チェックは省く（定義途中を許すため）。"""
    body = """
cameras:
  camA:
    exclude_regions:
      - type: rect
        points: [[0, 0], [9999, 9999]]
"""
    assert len(load_cameras(_write(tmp_path, body))["camA"].exclude_regions) == 1


def test_angle_out_of_range(tmp_path: Path) -> None:
    body = """
cameras:
  camA:
    image_size: [1920, 1080]
    transition_zones:
      - id: z1
        region: {type: rect, points: [[0, 0], [100, 100]]}
        exit_dir: [290, 400]
"""
    with pytest.raises(CameraConfigError) as e:
        load_cameras(_write(tmp_path, body))
    assert "400" in str(e.value) and "0〜360" in str(e.value)


def test_wrapping_angle_is_accepted(tmp_path: Path) -> None:
    body = """
cameras:
  camA:
    image_size: [1920, 1080]
    transition_zones:
      - id: z1
        region: {type: rect, points: [[0, 0], [100, 100]]}
        exit_dir: [290, 60]
"""
    assert load_cameras(_write(tmp_path, body))["camA"].transition_zones[0].exit_dir == (290.0, 60.0)


def test_duplicate_zone_id(tmp_path: Path) -> None:
    body = """
cameras:
  camA:
    image_size: [1920, 1080]
    transition_zones:
      - id: same
        region: {type: rect, points: [[0, 0], [100, 100]]}
      - id: same
        region: {type: rect, points: [[200, 200], [300, 300]]}
"""
    with pytest.raises(CameraConfigError) as e:
        load_cameras(_write(tmp_path, body))
    assert "same" in str(e.value) and "重複" in str(e.value)


def test_same_zone_id_allowed_across_cameras(tmp_path: Path) -> None:
    """id の一意性はカメラ内だけで見る。"""
    body = """
cameras:
  camA:
    transition_zones:
      - id: corridor
        region: {type: rect, points: [[0, 0], [100, 100]]}
  camB:
    transition_zones:
      - id: corridor
        region: {type: rect, points: [[0, 0], [100, 100]]}
"""
    cameras = load_cameras(_write(tmp_path, body))
    assert cameras["camA"].transition_zones[0].id == cameras["camB"].transition_zones[0].id


def test_zone_without_id(tmp_path: Path) -> None:
    body = """
cameras:
  camA:
    transition_zones:
      - region: {type: rect, points: [[0, 0], [100, 100]]}
"""
    with pytest.raises(CameraConfigError, match="id"):
        load_cameras(_write(tmp_path, body))


def test_zone_optional_fields_default(tmp_path: Path) -> None:
    body = """
cameras:
  camA:
    transition_zones:
      - id: z1
        region: {type: rect, points: [[0, 0], [100, 100]]}
"""
    zone = load_cameras(_write(tmp_path, body))["camA"].transition_zones[0]
    assert zone.exit_dir is None and zone.entry_dir is None and zone.to == []
