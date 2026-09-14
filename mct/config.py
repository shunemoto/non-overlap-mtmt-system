"""設定 YAML の読み込み。

`params.yaml` は書かれていないキーを既定値で埋めるので、最小限の記述でも動く。
`cameras.yaml` は観測区間と遷移領域の定義で、これから手作業で書いていくため、
途中の状態でも読み込めるようにしてある（検証は明らかな誤りだけに絞る）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "DEFAULT_TRACKING",
    "load_params",
    "tracking_params",
    "CameraConfigError",
    "Region",
    "TransitionTarget",
    "TransitionZone",
    "Homography",
    "CameraConfig",
    "load_cameras",
]

DEFAULT_TRACKING: dict[str, Any] = {
    "model_pose": "yolo11l-pose.pt",
    "model_seg": "yolo11x-seg.pt",
    "tracker": "botsort.yaml",
    "conf_threshold": 0.25,
    "n_saved_per_track": 90,
    "crop_padding_ratio": 0.10,
    "device": "mps",
    "imgsz": 640,
    "stride": 1,
    "mask_iou_threshold": 0.30,
    "n_bins": 9,
    "jpeg_quality": 95,
}


def load_params(path: str | Path | None) -> dict[str, Any]:
    """パラメータ YAML を読む。`path` が None か存在しなければ空とみなす。"""
    if path is None:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def tracking_params(
    path: str | Path | None, **overrides: Any
) -> dict[str, Any]:
    """既定値 → YAML の `tracking:` → 明示的な上書き、の順にマージする。

    `overrides` の値が None のキーは「指定なし」として無視するので、
    argparse の未指定オプションをそのまま渡せる。
    """
    params = dict(DEFAULT_TRACKING)
    params.update(load_params(path).get("tracking") or {})
    params.update({k: v for k, v in overrides.items() if v is not None})
    return params


# --------------------------------------------------------------------------
# cameras.yaml — 観測区間と遷移領域
# --------------------------------------------------------------------------


class CameraConfigError(ValueError):
    """cameras.yaml の記述に誤りがあるときに送出する。"""


@dataclass
class Region:
    """画像上の領域。`rect` は [左上, 右下] の2点、`polygon` は3点以上の頂点列。"""

    type: str
    points: list[list[float]]


@dataclass
class TransitionTarget:
    """遷移先。`room` は部屋まで特定できる場合のみ。"""

    cam: str
    room: str | None = None


@dataclass
class TransitionZone:
    """遷移領域。出現・退出の位置と進行方向から遷移先候補を絞る。"""

    id: str
    region: Region
    exit_dir: tuple[float, float] | None = None
    entry_dir: tuple[float, float] | None = None
    to: list[TransitionTarget] = field(default_factory=list)


@dataclass
class Homography:
    """画像座標から BEV 実空間座標への対応点。未計測なら空のまま。"""

    image_points: list[list[float]] = field(default_factory=list)
    world_points: list[list[float]] = field(default_factory=list)


@dataclass
class CameraConfig:
    """1台分の定義。どの項目も未記入のまま置ける。"""

    name: str
    image_size: tuple[int, int] | None = None
    exclude_regions: list[Region] = field(default_factory=list)
    transition_zones: list[TransitionZone] = field(default_factory=list)
    homography: Homography = field(default_factory=Homography)


def _parse_region(raw: Any, camera: str, where: str) -> Region:
    if not isinstance(raw, dict):
        raise CameraConfigError(f"{camera}: {where} は type と points を持つ辞書にしてください")

    rtype = raw.get("type")
    if rtype not in ("rect", "polygon"):
        raise CameraConfigError(
            f'{camera}: {where} の type は "rect" か "polygon" ですが "{rtype}" でした'
        )

    points = raw.get("points") or []
    if not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in points):
        raise CameraConfigError(f"{camera}: {where} の points は [x, y] の並びにしてください")

    if rtype == "rect" and len(points) != 2:
        raise CameraConfigError(
            f"{camera}: {where} は rect なので points は [左上, 右下] の2点ですが "
            f"{len(points)} 点でした"
        )
    if rtype == "polygon" and len(points) < 3:
        raise CameraConfigError(
            f"{camera}: {where} は polygon なので points は3点以上必要ですが "
            f"{len(points)} 点でした"
        )

    return Region(type=rtype, points=[[float(x), float(y)] for x, y in points])


def _check_bounds(
    region: Region, image_size: tuple[int, int] | None, camera: str, where: str
) -> None:
    """座標が画像の範囲に収まっているか見る。image_size 未記入なら省略する。"""
    if image_size is None:
        return
    width, height = image_size
    for x, y in region.points:
        if not (0 <= x <= width and 0 <= y <= height):
            raise CameraConfigError(
                f"{camera}: {where} の点 ({x:g}, {y:g}) が "
                f"image_size [{width}, {height}] の外にあります"
            )


def _parse_dir(raw: Any, camera: str, where: str) -> tuple[float, float] | None:
    if raw is None:
        return None
    if not (isinstance(raw, (list, tuple)) and len(raw) == 2):
        raise CameraConfigError(
            f"{camera}: {where} は [開始角, 終了角] の2要素にしてください"
        )
    lo, hi = float(raw[0]), float(raw[1])
    for value in (lo, hi):
        if not 0.0 <= value <= 360.0:
            raise CameraConfigError(
                f"{camera}: {where} の角度 {value:g} が 0〜360 の範囲外です"
            )
    return lo, hi


def _parse_zone(raw: Any, camera: str, index: int) -> TransitionZone:
    if not isinstance(raw, dict):
        raise CameraConfigError(f"{camera}: transition_zones[{index}] は辞書にしてください")

    zone_id = raw.get("id")
    if not zone_id:
        raise CameraConfigError(f"{camera}: transition_zones[{index}] に id がありません")

    where = f"transition_zones[{index}] '{zone_id}'"
    region = _parse_region(raw.get("region"), camera, f"{where} の region")

    targets = []
    for j, t in enumerate(raw.get("to") or []):
        if not isinstance(t, dict) or "cam" not in t:
            raise CameraConfigError(f"{camera}: {where} の to[{j}] に cam がありません")
        targets.append(TransitionTarget(cam=str(t["cam"]), room=t.get("room")))

    return TransitionZone(
        id=str(zone_id),
        region=region,
        exit_dir=_parse_dir(raw.get("exit_dir"), camera, f"{where} の exit_dir"),
        entry_dir=_parse_dir(raw.get("entry_dir"), camera, f"{where} の entry_dir"),
        to=targets,
    )


def _parse_camera(name: str, raw: Any) -> CameraConfig:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise CameraConfigError(f"{name}: カメラの定義は辞書にしてください")

    size_raw = raw.get("image_size")
    image_size: tuple[int, int] | None = None
    if size_raw is not None:
        if not (isinstance(size_raw, (list, tuple)) and len(size_raw) == 2):
            raise CameraConfigError(f"{name}: image_size は [幅, 高さ] の2要素にしてください")
        image_size = (int(size_raw[0]), int(size_raw[1]))

    excludes = []
    for i, raw_region in enumerate(raw.get("exclude_regions") or []):
        region = _parse_region(raw_region, name, f"exclude_regions[{i}]")
        _check_bounds(region, image_size, name, f"exclude_regions[{i}]")
        excludes.append(region)

    zones = []
    seen: set[str] = set()
    for i, raw_zone in enumerate(raw.get("transition_zones") or []):
        zone = _parse_zone(raw_zone, name, i)
        if zone.id in seen:
            raise CameraConfigError(
                f"{name}: transition_zones の id '{zone.id}' が重複しています"
            )
        seen.add(zone.id)
        _check_bounds(
            zone.region, image_size, name, f"transition_zones[{i}] '{zone.id}' の region"
        )
        zones.append(zone)

    homo_raw = raw.get("homography") or {}
    homography = Homography(
        image_points=[[float(x), float(y)] for x, y in (homo_raw.get("image_points") or [])],
        world_points=[[float(x), float(y)] for x, y in (homo_raw.get("world_points") or [])],
    )

    return CameraConfig(
        name=name,
        image_size=image_size,
        exclude_regions=excludes,
        transition_zones=zones,
        homography=homography,
    )


def load_cameras(path: str | Path) -> dict[str, CameraConfig]:
    """cameras.yaml を読み、カメラ名から定義への辞書を返す。

    定義を書いている途中でも読めるよう、未記入の項目は空として扱う。
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"カメラ定義が見つかりません: {p}")

    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cameras = raw.get("cameras")
    if cameras is None:
        return {}
    if not isinstance(cameras, dict):
        raise CameraConfigError(f"{p}: cameras はカメラ名をキーとする辞書にしてください")

    return {name: _parse_camera(name, body) for name, body in cameras.items()}
