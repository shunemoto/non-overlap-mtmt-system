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
    assert [r.type for r in cam_a.exclude_regions] == ["rect", "rect"]

    zone = cam_a.transition_zones[0]
    assert zone.id == "a_to_b"
    assert zone.exit_dir == (160.0, 234.0)
    assert zone.entry_dir == (336.0, 54.0)     # 0 をまたぐ区間
    assert [(t.cam, t.room) for t in zone.to] == [("camB", None), ("camA", "design_room")]


def test_every_camera_is_defined() -> None:
    """8台すべてに観測区間・遷移領域・ホモグラフィが入っていること。"""
    cameras = load_cameras(REPO_ROOT / "configs" / "cameras.yaml")

    assert sum(len(c.transition_zones) for c in cameras.values()) == 17
    for name, config in cameras.items():
        assert config.exclude_regions, name
        assert config.transition_zones, name
        assert config.homography.world_unit == "cm", name
        assert len(config.homography.world_points) >= 4, name
        assert set(config.homography.image_points_by_date) == {
            "0720", "0727", "0730", "0803"
        }, name


def test_homography_points_match_per_date() -> None:
    cameras = load_cameras(REPO_ROOT / "configs" / "cameras.yaml")

    for name, config in cameras.items():
        n_world = len(config.homography.world_points)
        for date, points in config.homography.image_points_by_date.items():
            assert len(points) == n_world, f"{name}/{date}"


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


# --- transitions.yaml ----------------------------------------------------


def _transitions(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "transitions.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def _two_zone_cameras(tmp_path: Path):
    body = """
cameras:
  camA:
    image_size: [1920, 1080]
    transition_zones:
      - id: a_to_b
        region: {type: rect, points: [[0, 0], [100, 100]]}
  camB:
    image_size: [1920, 1080]
    transition_zones:
      - id: b_to_a
        region: {type: rect, points: [[0, 0], [100, 100]]}
"""
    return load_cameras(_write(tmp_path, body))


def test_loads_repository_transitions() -> None:
    from mct.config import load_transitions

    cameras = load_cameras(REPO_ROOT / "configs" / "cameras.yaml")
    routes = load_transitions(REPO_ROOT / "configs" / "transitions.yaml", cameras)

    assert len(routes) == 30
    assert sum(1 for r in routes if r.is_self_loop) == 4
    assert sum(1 for r in routes if r.search_range_sec[1] is None) == 4


def test_unknown_zone_raises(tmp_path: Path) -> None:
    from mct.config import load_transitions

    cameras = _two_zone_cameras(tmp_path)
    body = """
transitions:
  - from: {cam: camA, zone: a_to_b}
    to: {cam: camB, zone: nope}
    distance_m: 10.0
    search_range_sec: [5, 10]
    c_dwell: 0.1
"""
    with pytest.raises(CameraConfigError) as e:
        load_transitions(_transitions(tmp_path, body), cameras)
    assert "nope" in str(e.value) and "camB" in str(e.value)


def test_unknown_camera_raises(tmp_path: Path) -> None:
    from mct.config import load_transitions

    cameras = _two_zone_cameras(tmp_path)
    body = """
transitions:
  - from: {cam: camZ, zone: a_to_b}
    to: {cam: camB, zone: b_to_a}
    distance_m: 10.0
    search_range_sec: [5, 10]
    c_dwell: 0.1
"""
    with pytest.raises(CameraConfigError, match="camZ"):
        load_transitions(_transitions(tmp_path, body), cameras)


def test_reversed_search_range_raises(tmp_path: Path) -> None:
    from mct.config import load_transitions

    cameras = _two_zone_cameras(tmp_path)
    body = """
transitions:
  - from: {cam: camA, zone: a_to_b}
    to: {cam: camB, zone: b_to_a}
    distance_m: 10.0
    search_range_sec: [30, 10]
    c_dwell: 0.1
"""
    with pytest.raises(CameraConfigError, match="下限"):
        load_transitions(_transitions(tmp_path, body), cameras)


def test_open_ended_search_range_is_allowed(tmp_path: Path) -> None:
    from mct.config import load_transitions

    cameras = _two_zone_cameras(tmp_path)
    body = """
transitions:
  - from: {cam: camA, zone: a_to_b}
    to: {cam: camA, zone: a_to_b}
    distance_m: 30.0
    search_range_sec: [300, null]
    c_dwell: 0.1
"""
    routes = load_transitions(_transitions(tmp_path, body), cameras)
    assert routes[0].search_range_sec == (300.0, None)
    assert routes[0].is_self_loop


def test_non_positive_values_raise(tmp_path: Path) -> None:
    from mct.config import load_transitions

    cameras = _two_zone_cameras(tmp_path)
    for key in ("distance_m", "c_dwell"):
        body = f"""
transitions:
  - from: {{cam: camA, zone: a_to_b}}
    to: {{cam: camB, zone: b_to_a}}
    distance_m: {0 if key == 'distance_m' else 10.0}
    search_range_sec: [5, 10]
    c_dwell: {0 if key == 'c_dwell' else 0.1}
"""
        with pytest.raises(CameraConfigError, match=key):
            load_transitions(_transitions(tmp_path, body), cameras)


def test_asymmetric_pair_warns(tmp_path: Path) -> None:
    import warnings
    from mct.config import load_transitions

    cameras = _two_zone_cameras(tmp_path)
    body = """
transitions:
  - from: {cam: camA, zone: a_to_b}
    to: {cam: camB, zone: b_to_a}
    distance_m: 10.0
    search_range_sec: [5, 10]
    c_dwell: 0.1
"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_transitions(_transitions(tmp_path, body), cameras)

    assert len(caught) == 1
    assert "逆向き" in str(caught[0].message)


def test_symmetric_pair_does_not_warn(tmp_path: Path) -> None:
    import warnings
    from mct.config import load_transitions

    cameras = _two_zone_cameras(tmp_path)
    body = """
transitions:
  - from: {cam: camA, zone: a_to_b}
    to: {cam: camB, zone: b_to_a}
    distance_m: 10.0
    search_range_sec: [5, 10]
    c_dwell: 0.1
  - from: {cam: camB, zone: b_to_a}
    to: {cam: camA, zone: a_to_b}
    distance_m: 10.0
    search_range_sec: [5, 10]
    c_dwell: 0.1
"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_transitions(_transitions(tmp_path, body), cameras)

    assert caught == []


def test_empty_transitions(tmp_path: Path) -> None:
    from mct.config import load_transitions

    cameras = _two_zone_cameras(tmp_path)
    assert load_transitions(_transitions(tmp_path, "transitions:\n"), cameras) == []


# --- homography の検証 ---------------------------------------------------


def test_homography_point_count_mismatch_raises(tmp_path: Path) -> None:
    body = """
cameras:
  camA:
    homography:
      world_unit: cm
      world_points: [[0, 0], [0, 100], [100, 100], [100, 0]]
      image_points_by_date:
        "0720": [[0, 0], [0, 10], [10, 10]]
"""
    with pytest.raises(CameraConfigError) as e:
        load_cameras(_write(tmp_path, body))
    assert "0720" in str(e.value) and "3 点" in str(e.value)


def test_homography_unknown_unit_raises(tmp_path: Path) -> None:
    body = """
cameras:
  camA:
    homography:
      world_unit: mm
      world_points: [[0, 0], [0, 100], [100, 100], [100, 0]]
"""
    with pytest.raises(CameraConfigError, match="world_unit"):
        load_cameras(_write(tmp_path, body))


def test_homography_too_few_points_raises(tmp_path: Path) -> None:
    body = """
cameras:
  camA:
    homography:
      world_points: [[0, 0], [0, 100], [100, 100]]
"""
    with pytest.raises(CameraConfigError, match="4点以上"):
        load_cameras(_write(tmp_path, body))


def test_homography_absent_is_allowed(tmp_path: Path) -> None:
    cameras = load_cameras(_write(tmp_path, "cameras:\n  camA:\n    image_size: [1920, 1080]\n"))
    homography = cameras["camA"].homography

    assert homography.world_points == []
    assert homography.image_points("0720") is None


def test_homography_image_points_are_not_bounds_checked(tmp_path: Path) -> None:
    """画面端の計測点が数 px はみ出しても通す（計算に実害がないため）。"""
    body = """
cameras:
  camA:
    image_size: [1920, 1080]
    homography:
      world_unit: cm
      world_points: [[0, 0], [0, 100], [100, 100], [100, 0]]
      image_points_by_date:
        "0720": [[0, 0], [0, 1082], [100, 1082], [100, 0]]
"""
    cameras = load_cameras(_write(tmp_path, body))
    assert cameras["camA"].homography.image_points("0720")[1] == [0.0, 1082.0]
