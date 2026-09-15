"""キャッシュキーのテスト。"""

from __future__ import annotations

from pathlib import Path

from mct.cache import cache_key, cache_path, clear, list_entries, meta_path, save_meta


def test_key_ignores_dict_order() -> None:
    """辞書に書いた順番が違ってもキーは同じ。"""
    a = cache_key({"alpha": 1, "beta": [1, 2], "gamma": {"x": 1, "y": 2}})
    b = cache_key({"gamma": {"y": 2, "x": 1}, "beta": [1, 2], "alpha": 1})

    assert a == b


def test_key_changes_with_values() -> None:
    assert cache_key({"x": 1}) != cache_key({"x": 2})
    assert cache_key({"x": 1}) != cache_key({"y": 1})


def test_list_order_matters() -> None:
    """並びに意味がある値（座標列など）は順番が変わればキーも変わる。"""
    assert cache_key({"points": [[0, 0], [1, 1]]}) != cache_key({"points": [[1, 1], [0, 0]]})


def test_parent_changes_the_key() -> None:
    base = cache_key({"x": 1})

    assert cache_key({"x": 1}, "parent1") != base
    assert cache_key({"x": 1}, "parent1") != cache_key({"x": 1}, "parent2")


def test_parent_is_stable() -> None:
    assert cache_key({"x": 1}, "p") == cache_key({"x": 1}, "p")


def test_key_length() -> None:
    assert len(cache_key({"x": 1})) == 12


def test_cache_path_layout() -> None:
    p = cache_path("L2", "0720", "camA", "abc123", root="cache")

    assert p == Path("cache/L2/0720_camA_abc123.json")
    assert meta_path(p) == Path("cache/L2/0720_camA_abc123.meta.json")


def test_save_meta_records_the_subset(tmp_path: Path) -> None:
    import json

    target = tmp_path / "L2" / "0720_camA_abc123.json"
    subset = {"filtering": {"min_track_len": 31}}
    written = save_meta(target, subset, parent="p1")

    saved = json.loads(written.read_text(encoding="utf-8"))
    assert saved["subset"] == subset
    assert saved["parent"] == "p1"
    assert saved["key"] == cache_key(subset, "p1")


def test_list_and_clear(tmp_path: Path) -> None:
    for layer in ("L2", "L3"):
        d = tmp_path / layer
        d.mkdir(parents=True)
        (d / f"0720_camA_key.json").write_text("{}", encoding="utf-8")
        (d / f"0720_camA_key.meta.json").write_text("{}", encoding="utf-8")

    entries = list_entries(root=tmp_path)
    assert len(entries) == 2                      # .meta.json は数えない
    assert {layer for layer, _, _ in entries} == {"L2", "L3"}
    assert len(list_entries("L2", root=tmp_path)) == 1

    assert clear("L2", root=tmp_path) == 1
    assert len(list_entries(root=tmp_path)) == 1
    assert clear(root=tmp_path) == 1
    assert list_entries(root=tmp_path) == []


def test_list_on_missing_root(tmp_path: Path) -> None:
    assert list_entries(root=tmp_path / "nope") == []
    assert clear(root=tmp_path / "nope") == 0
