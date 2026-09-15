"""中間結果のキャッシュ。

設定を変えながら何度も回す作業で、変わっていない層を計算し直さずに済ませる。
キーは「その層の結果を左右する設定」だけから作るので、無関係な設定を
いじってもキャッシュは生きたままになる。

上流の層が変われば下流も無効にしたいので、`parent` に上流のキーを渡して
連鎖させる。
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

__all__ = ["CACHE_ROOT", "cache_key", "cache_path", "meta_path", "save_meta", "list_entries", "clear"]

CACHE_ROOT = Path("cache")
KEY_LENGTH = 12


def cache_key(subset: dict[str, Any], parent: str | None = None) -> str:
    """設定の一部からキャッシュキーを作る。

    `json.dumps(sort_keys=True)` を挟むので、辞書に書いた順番が違っても
    同じキーになる。`parent` を含めることで、上流の層のキーが変われば
    こちらのキーも変わり、古い結果を読まずに済む。
    """
    payload = {"subset": subset, "parent": parent}
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:KEY_LENGTH]


def cache_path(
    layer: str, date: str, camera: str, key: str, ext: str = "json",
    root: str | Path = CACHE_ROOT,
) -> Path:
    """`cache/{layer}/{date}_{camera}_{key}.{ext}`"""
    return Path(root) / layer / f"{date}_{camera}_{key}.{ext}"


def meta_path(path: str | Path) -> Path:
    """キャッシュ本体に対応する「生成に使った設定」のパス。"""
    p = Path(path)
    return p.with_name(f"{p.stem}.meta.json")


def save_meta(path: str | Path, subset: dict[str, Any], parent: str | None = None) -> Path:
    """何を根拠に作られたキャッシュかを残す。

    キーはハッシュなので、これが無いと後から中身を判別できない。
    """
    dest = meta_path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        json.dump(
            {"subset": subset, "parent": parent, "key": cache_key(subset, parent)},
            f, ensure_ascii=False, indent=2, default=str,
        )
    return dest


def list_entries(
    layer: str | None = None, root: str | Path = CACHE_ROOT
) -> list[tuple[str, Path, int]]:
    """(層, パス, バイト数) の一覧。`.meta.json` は除く。"""
    base = Path(root)
    if not base.is_dir():
        return []

    layers = [base / layer] if layer else sorted(p for p in base.iterdir() if p.is_dir())
    entries = []
    for d in layers:
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and not p.name.endswith(".meta.json"):
                entries.append((d.name, p, p.stat().st_size))
    return entries


def clear(layer: str | None = None, root: str | Path = CACHE_ROOT) -> int:
    """キャッシュを消し、消したディレクトリ数を返す。層を指定しなければ全部。"""
    base = Path(root)
    if not base.is_dir():
        return 0

    targets = [base / layer] if layer else [p for p in base.iterdir() if p.is_dir()]
    removed = 0
    for d in targets:
        if d.is_dir():
            shutil.rmtree(d)
            removed += 1
    return removed
