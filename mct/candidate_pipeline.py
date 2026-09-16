"""候補列挙の実行と診断。

`tracks_filtered.json` を読んで要約し、候補を列挙して
`outputs/{date}/candidates.json` に書く。GT があれば再現率まで出す。
"""

from __future__ import annotations

import collections
import json
import statistics
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from mct.cache import cache_key
from mct.candidates import Candidate, candidate_counts, enumerate_candidates
from mct.config import CameraConfig, load_cameras, load_params, load_transitions
from mct.geometry import get_homography
from mct.io_utils import load_tracking_result
from mct.summarize import TrackSummary, harmonic_mean_speed, summarize_camera

__all__ = ["CANDIDATES_FILENAME", "build_candidates", "run"]

CANDIDATES_FILENAME = "candidates.json"
ALL_CAMERAS = ("camA", "camB", "camC", "camD", "camE", "camF", "camG", "camH")
FILTERED = "tracks_filtered.json"

# 歩行速度として妥当とみなす範囲。外れていれば距離か対応点を疑う。
WALKING_RANGE = (0.5, 2.5)


def _spatiotemporal(params_path: str | Path) -> tuple[dict[str, Any], int]:
    """spatiotemporal の設定と、進行方向に使うフレーム数を返す。

    `dir_frames` は filtering 側の値をそのまま使う。遷移領域の判定と
    候補列挙で別の値を使うと、診断で見えているものと候補がずれる。
    """
    raw = load_params(params_path)
    params = {
        "fps": 30,
        "speed_target_sec": 1.5,
        "speed_min_segments": 3,
        "speed_stop_threshold": 0.3,
        "speed_min_mps": 0.6,
        "speed_max_mps": 2.0,
        "calib_min_samples": 5,
        "self_loop_dir_tol": 60,
    }
    params.update(raw.get("spatiotemporal") or {})
    dir_frames = int((raw.get("filtering") or {}).get("dir_frames", 30))
    return params, dir_frames


def build_candidates(
    date: str,
    out_root: str | Path,
    cameras_path: str | Path,
    transitions_path: str | Path,
    params_path: str | Path,
    *,
    no_topology: bool = False,
) -> tuple[list[TrackSummary], list[Candidate], dict[str, Any]]:
    cameras = load_cameras(cameras_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # 対称性の警告は候補列挙では不要
        routes = load_transitions(transitions_path, cameras)
    params, dir_frames = _spatiotemporal(params_path)

    summaries: list[TrackSummary] = []
    for camera in ALL_CAMERAS:
        source = Path(out_root) / date / camera / FILTERED
        if not source.is_file():
            continue
        result = load_tracking_result(source)
        summaries.extend(
            summarize_camera(
                date, camera, result.tracks,
                cameras.get(camera) or CameraConfig(name=camera),
                get_homography(cameras, camera, date),
                params, dir_frames,
            )
        )

    candidates = enumerate_candidates(summaries, routes, params, no_topology=no_topology)
    meta = {
        "date": date,
        "n_tracks": len(summaries),
        "n_candidates": len(candidates),
        "config_hash": cache_key(
            {
                "spatiotemporal": dict(sorted(params.items())),
                "dir_frames": dir_frames,
                "routes": [
                    [r.from_cam, r.from_zone, r.to_cam, r.to_zone,
                     r.distance_m, list(r.search_range_sec), r.c_dwell]
                    for r in routes
                ],
            },
            parent="no_topology" if no_topology else None,
        ),
        "no_topology": no_topology,
    }
    return summaries, candidates, meta


def write_candidates(
    date: str,
    out_root: str | Path,
    summaries: list[TrackSummary],
    candidates: list[Candidate],
    meta: dict[str, Any],
) -> Path:
    dest = Path(out_root) / date / CANDIDATES_FILENAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": meta,
        "summaries": [s.to_dict() for s in summaries],
        "candidates": [c.to_dict() for c in candidates],
    }
    with dest.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    return dest


# --------------------------------------------------------------------------
# 診断
# --------------------------------------------------------------------------


def print_counts(date: str, summaries: list[TrackSummary], candidates: list[Candidate]) -> None:
    counts = candidate_counts(candidates)
    print(f"\n--- {date} ---")
    print(f"  トラック {len(summaries)} / 候補 {counts['total']}")
    print(
        f"    探索時間内 {counts['in_range']} / 範囲外 {counts['out_of_range']}"
        f" / 自己ループ {counts['self_loop']}"
    )
    if counts["total"]:
        print("    カメラ対ごとの候補数（上位8）:")
        for (a, b), n in counts["by_camera_pair"].most_common(8):
            loop = "（自己ループ）" if a == b else ""
            print(f"      {a} -> {b}{loop}: {n}")


def print_speed_stats(summaries: list[TrackSummary]) -> None:
    speeds = [s.speed_mps for s in summaries if s.speed_mps is not None]
    n_none = sum(1 for s in summaries if s.speed_mps is None)
    n_fallback = sum(1 for s in summaries if s.speed_fallback)
    n_clipped = sum(1 for s in summaries if s.speed_clipped)

    print(f"\n速度の統計（トラック {len(summaries)}）")
    print(
        f"  算出できず {n_none} / 停止のみでカメラ中央値を代用 {n_fallback} "
        f"({100*n_fallback/max(1,len(summaries)):.1f}%) / クリップ {n_clipped} "
        f"({100*n_clipped/max(1,len(summaries)):.1f}%)"
    )
    if not speeds:
        return

    a = np.array(speeds)
    lo, hi = WALKING_RANGE
    inside = int(((a >= lo) & (a <= hi)).sum())
    print(
        f"  最小 {a.min():.2f} / p25 {np.percentile(a,25):.2f} / "
        f"中央 {np.median(a):.2f} / p75 {np.percentile(a,75):.2f} / 最大 {a.max():.2f} m/s"
    )
    print(
        f"  歩行速度として妥当な {lo}〜{hi} m/s に入る割合: "
        f"{100*inside/len(a):.1f}% ({inside}/{len(a)})"
    )


def print_arrival_error(
    candidates: list[Candidate],
    summaries: dict[tuple[str, int], TrackSummary],
    correct_pairs: set[tuple[tuple[str, int], tuple[str, int]]],
    fps: float,
) -> None:
    """GT で正解の対について、予測到着フレームと実際のずれを集計する。

    フェーズ4で到着時刻のばらつき σ を決める材料。ゾーン対ごとの中央値が
    0 から大きくずれていれば、距離か速度にバイアスがある。
    """
    errors: list[float] = []
    by_zone: dict[tuple[str, str], list[float]] = collections.defaultdict(list)

    for c in candidates:
        if c.route is None or (c.tgt, c.src) not in correct_pairs:
            continue
        tgt, src = summaries.get(c.tgt), summaries.get(c.src)
        if tgt is None or src is None:
            continue
        # 到着時刻は距離÷速度なので、平均すべきは 1m あたりの所要時間。
        # その逆数、すなわち調和平均を使う。
        speed = harmonic_mean_speed(tgt.speed_mps, src.speed_mps)
        if not speed or speed <= 0:
            continue
        predicted = c.route.distance_m / speed * fps
        errors.append(float(predicted - c.delta_frames))
        by_zone[(c.tgt_zone, c.src_zone)].append(float(predicted - c.delta_frames))

    print(f"\n予測到着時刻の誤差（正解の対 {len(errors)} 件）")
    if not errors:
        print("  対象がありません")
        return

    a = np.array(errors)
    print(
        f"  中央 {np.median(a):+.0f} フレーム ({np.median(a)/fps:+.2f} 秒) / "
        f"p25 {np.percentile(a,25):+.0f} / p75 {np.percentile(a,75):+.0f}"
    )
    print(f"  （正なら予測が遅い＝実際より時間がかかると見込んでいる）")
    print("  ゾーン対ごとの中央値（件数の多い順・上位10）:")
    for (tz, sz), values in sorted(by_zone.items(), key=lambda kv: -len(kv[1]))[:10]:
        print(
            f"    {tz} -> {sz:<12} {len(values):>3} 件  "
            f"中央 {statistics.median(values):+7.0f} フレーム "
            f"({statistics.median(values)/fps:+.2f} 秒)"
        )
