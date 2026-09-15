"""フェーズ2の実行手順。

`tracks.json` を読み、観測区間を適用し、静止誤検出を除き、
`tracks_filtered.json` を書く。同じ設定で2回目を回したときは L2 キャッシュから
読み、追跡も再計算もしない。

追跡モデルを import しないので、追跡を回しながら並行して実行できる。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from mct.cache import cache_key, cache_path, save_meta
from mct.config import CameraConfig, load_cameras, tracking_params
from mct.diagnostics import CameraReport, analyse, print_camera_report
from mct.filtering import (
    apply_observation_region,
    copy_removed_crops,
    remove_static_detections,
)
from mct.geometry import get_homography
from mct.io_utils import load_tracking_result, output_dir, result_path, save_tracking_result
from mct.schema import TrackingResult

__all__ = ["FILTERED_FILENAME", "filter_camera", "l2_subset"]

FILTERED_FILENAME = "tracks_filtered.json"
LAYER = "L2"


def l2_subset(cam_cfg: CameraConfig, params: dict[str, Any], date: str) -> dict[str, Any]:
    """L2 キャッシュのキーに使う設定だけを取り出す。

    この層の結果を左右するのは観測区間・filtering の設定・その日の
    ホモグラフィ計測点だけ。追跡モデルや可視化の設定を変えても
    キャッシュは生きたままにする。
    """
    return {
        "exclude_regions": [
            {"type": r.type, "points": r.points} for r in cam_cfg.exclude_regions
        ],
        "filtering": dict(sorted(params.items())),
        "image_points": cam_cfg.homography.image_points(date),
        "world_points": cam_cfg.homography.world_points,
        "world_unit": cam_cfg.homography.world_unit,
    }


def filter_camera(
    date: str,
    camera: str,
    out_root: str | Path,
    cameras_path: str | Path,
    params_path: str | Path,
    *,
    force: bool = False,
) -> tuple[TrackingResult, CameraReport, list[dict[str, Any]], bool]:
    """1カメラ分を絞り込む。

    戻り値は (絞り込み後の結果, 診断, 静止として除外した一覧,
    キャッシュから読んだか)。
    """
    cameras = load_cameras(cameras_path)
    cam_cfg = cameras.get(camera) or CameraConfig(name=camera)
    params = _filtering_params(params_path)

    out_dir = output_dir(out_root, date, camera)
    source = result_path(out_dir)
    if not source.is_file():
        raise FileNotFoundError(f"{source} がありません。先に track を実行してください。")

    subset = l2_subset(cam_cfg, params, date)
    cached = cache_path(LAYER, date, camera, cache_key(subset))
    dest = out_dir / FILTERED_FILENAME

    result = load_tracking_result(source)
    H = get_homography(cameras, camera, date)

    # 絞り込み自体は数秒で終わるが、診断に必要な内訳も同時に得られるので
    # キャッシュの有無にかかわらず通す。キャッシュが効くのは書き出しの方。
    after_region, region_stats = apply_observation_region(result, cam_cfg, params)
    filtered, removed = remove_static_detections(after_region, H, params)

    from_cache = cached.is_file() and not force
    if from_cache:
        filtered = load_tracking_result(cached)
        if not dest.is_file():
            shutil.copy2(cached, dest)
    else:
        copy_removed_crops(removed, after_region, out_dir)
        save_tracking_result(filtered, dest)
        save_tracking_result(filtered, cached)
        save_meta(cached, subset)

    report = analyse(date, camera, filtered, region_stats, removed, cam_cfg, params)
    return filtered, report, removed, from_cache


def _filtering_params(params_path: str | Path) -> dict[str, Any]:
    """params.yaml の filtering 節を既定値とマージして返す。"""
    from mct.config import load_params

    defaults = {
        "min_track_len": 31,
        "n_frames_per_track": 30,
        "n_bins": 9,
        "static_max_distance_m": 1.0,
        "static_min_len": 60,
        "static_max_std_m": 0.3,
        "static_max_conf": 0.7,
        "static_max_distance_px": 80,
        "static_max_std_px": 25,
        "dir_frames": 30,
    }
    defaults.update(load_params(params_path).get("filtering") or {})
    return defaults


def run(
    dates: list[str],
    cameras_list: list[str],
    out_root: str | Path,
    cameras_path: str | Path,
    params_path: str | Path,
    *,
    force: bool = False,
) -> list[CameraReport]:
    """複数カメラをまとめて処理し、カメラごとに診断を出す。"""
    reports: list[CameraReport] = []
    for date in dates:
        for camera in cameras_list:
            source = result_path(Path(out_root) / date / camera)
            if not source.is_file():
                print(f"\n--- {date}/{camera} --- (tracks.json が無いので飛ばします)")
                continue
            _, report, removed, from_cache = filter_camera(
                date, camera, out_root, cameras_path, params_path, force=force
            )
            print_camera_report(report, removed)
            if from_cache:
                print("  （キャッシュから読み込み）")
            reports.append(report)
    return reports
