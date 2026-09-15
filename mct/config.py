"""設定 YAML の読み込み。

`params.yaml` は書かれていないキーを既定値で埋めるので、最小限の記述でも動く。
`cameras.yaml` は観測区間と遷移領域の定義で、これから手作業で書いていくため、
途中の状態でも読み込めるようにしてある（検証は明らかな誤りだけに絞る）。
"""

from __future__ import annotations

import warnings
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
    "TransitionRoute",
    "load_transitions",
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
    """画像座標から BEV 実空間座標への対応点。未計測なら空のまま。

    カメラは日付ごとに画角がわずかにずれているため、実空間側の計測点は
    共通で、画像側の対応点だけを日付ごとに持つ。観測区間と遷移領域は
    ずれを織り込んで定義してあるので補正しない。
    """

    world_unit: str = "cm"
    world_points: list[list[float]] = field(default_factory=list)
    image_points_by_date: dict[str, list[list[float]]] = field(default_factory=dict)

    def image_points(self, date: str) -> list[list[float]] | None:
        """その日付の画像側対応点。無ければ None。"""
        return self.image_points_by_date.get(date)


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


def _points(raw: Any, camera: str, where: str) -> list[list[float]]:
    if not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in raw or []):
        raise CameraConfigError(f"{camera}: {where} は [x, y] の並びにしてください")
    return [[float(x), float(y)] for x, y in (raw or [])]


def _parse_homography(raw: Any, camera: str) -> Homography:
    """homography 節を読む。未記入のカメラは空の Homography として通す。

    画像側の計測点には画像範囲の検査をかけない。画角の端で計測した点が
    数 px はみ出すことがあり、ホモグラフィの計算には実害がないため。
    """
    raw = raw or {}
    if not isinstance(raw, dict):
        raise CameraConfigError(f"{camera}: homography は辞書にしてください")

    unit = str(raw.get("world_unit", "cm"))
    if unit not in ("cm", "m"):
        raise CameraConfigError(
            f'{camera}: homography の world_unit は "cm" か "m" ですが "{unit}" でした'
        )

    world = _points(raw.get("world_points"), camera, "homography の world_points")
    by_date_raw = raw.get("image_points_by_date") or {}
    if not isinstance(by_date_raw, dict):
        raise CameraConfigError(
            f"{camera}: homography の image_points_by_date は日付をキーとする辞書にしてください"
        )

    by_date: dict[str, list[list[float]]] = {}
    for date, pts_raw in by_date_raw.items():
        date = str(date)
        pts = _points(pts_raw, camera, f"homography の image_points_by_date['{date}']")
        # 点数が食い違うと対応が1つずれたまま計算が通ってしまう
        if len(pts) != len(world):
            raise CameraConfigError(
                f"{camera}: homography の image_points_by_date['{date}'] は {len(pts)} 点ですが "
                f"world_points は {len(world)} 点です"
            )
        by_date[date] = pts

    # ホモグラフィの算出には4点以上が要る。未記入（0点）は許す。
    if world and len(world) < 4:
        raise CameraConfigError(
            f"{camera}: homography は4点以上必要ですが world_points が {len(world)} 点でした"
        )

    return Homography(world_unit=unit, world_points=world, image_points_by_date=by_date)


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

    homography = _parse_homography(raw.get("homography"), name)

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


# --------------------------------------------------------------------------
# transitions.yaml — カメラ間の遷移経路
# --------------------------------------------------------------------------


@dataclass
class TransitionRoute:
    """ある遷移領域から別の遷移領域への経路。

    `search_range_sec` の上限が None なら上限なし（居室に長時間滞在してから
    戻ってくる自己ループで使う）。
    """

    from_cam: str
    from_zone: str
    to_cam: str
    to_zone: str
    distance_m: float
    search_range_sec: tuple[float, float | None]
    c_dwell: float

    @property
    def is_self_loop(self) -> bool:
        return self.from_cam == self.to_cam and self.from_zone == self.to_zone


def _parse_endpoint(
    raw: Any, cameras: dict[str, CameraConfig], where: str
) -> tuple[str, str]:
    if not isinstance(raw, dict) or "cam" not in raw or "zone" not in raw:
        raise CameraConfigError(f"{where} は cam と zone を持つ辞書にしてください")

    cam, zone = str(raw["cam"]), str(raw["zone"])
    if cam not in cameras:
        raise CameraConfigError(f"{where}: カメラ '{cam}' が cameras.yaml にありません")
    if zone not in {z.id for z in cameras[cam].transition_zones}:
        known = ", ".join(z.id for z in cameras[cam].transition_zones) or "（定義なし）"
        raise CameraConfigError(
            f"{where}: {cam} に遷移領域 '{zone}' がありません（定義済み: {known}）"
        )
    return cam, zone


def load_transitions(
    path: str | Path, cameras: dict[str, CameraConfig]
) -> list[TransitionRoute]:
    """transitions.yaml を読む。参照先の遷移領域が実在するかを `cameras` で検証する。

    A→B があるのに B→A が無い場合は警告する。片側だけの定義は書き忘れの
    ことが多いが、一方通行の経路も有り得るので例外にはしない。
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"遷移定義が見つかりません: {p}")

    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    entries = raw.get("transitions")
    if entries is None:
        return []
    if not isinstance(entries, list):
        raise CameraConfigError(f"{p}: transitions は一覧にしてください")

    routes: list[TransitionRoute] = []
    for i, entry in enumerate(entries):
        where = f"transitions[{i}]"
        if not isinstance(entry, dict):
            raise CameraConfigError(f"{where} は辞書にしてください")

        from_cam, from_zone = _parse_endpoint(entry.get("from"), cameras, f"{where} の from")
        to_cam, to_zone = _parse_endpoint(entry.get("to"), cameras, f"{where} の to")

        rng = entry.get("search_range_sec")
        if not (isinstance(rng, (list, tuple)) and len(rng) == 2):
            raise CameraConfigError(
                f"{where}: search_range_sec は [下限, 上限] の2要素にしてください"
            )
        lo = float(rng[0])
        hi = None if rng[1] is None else float(rng[1])
        if hi is not None and lo > hi:
            raise CameraConfigError(
                f"{where}: search_range_sec の下限 {lo:g} が上限 {hi:g} を超えています"
            )

        distance_m = float(entry.get("distance_m", 0.0))
        c_dwell = float(entry.get("c_dwell", 0.0))
        for key, value in (("distance_m", distance_m), ("c_dwell", c_dwell)):
            if value <= 0:
                raise CameraConfigError(f"{where}: {key} は正の数にしてください（{value:g}）")

        routes.append(
            TransitionRoute(
                from_cam=from_cam, from_zone=from_zone,
                to_cam=to_cam, to_zone=to_zone,
                distance_m=distance_m,
                search_range_sec=(lo, hi),
                c_dwell=c_dwell,
            )
        )

    _warn_asymmetric(routes)
    return routes


def _warn_asymmetric(routes: list[TransitionRoute]) -> None:
    pairs = {(r.from_cam, r.from_zone, r.to_cam, r.to_zone) for r in routes}
    for route in routes:
        if route.is_self_loop:
            continue
        back = (route.to_cam, route.to_zone, route.from_cam, route.from_zone)
        if back not in pairs:
            warnings.warn(
                f"transitions: {route.from_cam}/{route.from_zone} -> "
                f"{route.to_cam}/{route.to_zone} の逆向きが定義されていません",
                stacklevel=2,
            )
