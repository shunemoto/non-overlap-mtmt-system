"""外観パラメータの掃引。

crop を**1回だけ読んで全変種の記述子を同時に作る**。設定ごとに画像を
読み直すと、変種の数だけ実行時間が伸びる。1トラック分の画素を持っている
間に全変種を計算して捨てるので、メモリも増えない。

記述子の計算は `sift_patch_ratio` の数だけで足りる。集約（フレーム平均）は
画素を持たなくてもできるので、切り出しと分けてある。
"""

from __future__ import annotations

import collections
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mct.appearance import (
    N_JOINTS,
    AppearanceDescriptor,
    FrameFeature,
    aggregate,
    color_histogram,
    sift_features,
    split_y,
)
from mct.appearance_sim import score_pairs
from mct.candidates import Candidate
from mct.io_utils import crop_path, load_tracking_result, mask_path
from mct.metrics import auc

__all__ = [
    "HIST_BINS", "ACHROMATIC", "PATCH_RATIOS",
    "SweepResult", "collect_variants", "evaluate", "print_report",
]

HIST_BINS = ((8, 8), (12, 12), (16, 16))
ACHROMATIC = (0.10, 0.15, 0.20, None)
# 0.09 を超えると、カメラ間では尺度と視点の差を拾って下がる。両端が下がる
# ことを確かめられるよう、頂点の前後を含めた範囲にしてある。
PATCH_RATIOS = (0.05, 0.07, 0.09, 0.11, 0.13)


def _color_key(bins: tuple[int, int], threshold: float | None) -> str:
    label = "なし" if threshold is None else f"{threshold:.2f}"
    return f"bins={bins[0]}x{bins[1]} achrom={label}"


def _sift_key(ratio: float) -> str:
    return f"ratio={ratio:.2f}"


@dataclass
class SweepResult:
    """1変種ぶんの AUC。"""

    key: str
    auc_all: float
    auc_in_range: float
    auc_self_loop: float
    extra: str = ""


def collect_variants(
    out_root: str | Path, date: str, cameras: list[str], base: dict[str, Any]
) -> tuple[dict[str, dict], dict[str, dict]]:
    """(色の変種, SIFT の変種)。それぞれ 変種キー -> {(cam, tid): 記述子}。"""
    color_frames: dict[str, dict[tuple[str, int], list[FrameFeature]]] = {
        _color_key(b, t): collections.defaultdict(list)
        for b in HIST_BINS for t in ACHROMATIC
    }
    sift_frames: dict[float, dict[tuple[str, int], list[FrameFeature]]] = {
        r: collections.defaultdict(list) for r in PATCH_RATIOS
    }

    for camera in cameras:
        source = Path(out_root) / date / camera / "tracks_filtered.json"
        if not source.is_file():
            continue
        result = load_tracking_result(source)
        directory = source.parent

        for track_id, points in result.tracks.items():
            key = (camera, int(track_id))
            for point in points:
                if not point.saved or point.crop_box is None:
                    continue
                crop = cv2.imread(str(crop_path(directory, track_id, point.frame_idx)))
                if crop is None:
                    continue
                mask_file = mask_path(directory, track_id, point.frame_idx)
                mask = (
                    cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                    if mask_file.is_file() else None
                )
                if mask is not None and mask.shape[:2] != crop.shape[:2]:
                    mask = None

                line, _ = split_y(point.keypoints, point.bbox)
                row = int(round(line - float(point.crop_box[1])))
                for bins in HIST_BINS:
                    for threshold in ACHROMATIC:
                        upper, lower = color_histogram(
                            crop, mask, row,
                            {"hist_bins": bins, "achromatic_threshold": threshold},
                        )
                        color_frames[_color_key(bins, threshold)][key].append(
                            FrameFeature(upper, lower, [None] * N_JOINTS, "", True)
                        )

                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                for ratio in PATCH_RATIOS:
                    sift = sift_features(
                        gray, point.keypoints, point.crop_box, point.bbox,
                        dict(base, sift_patch_ratio=ratio),
                    )
                    sift_frames[ratio][key].append(
                        FrameFeature(None, None, sift, "", True)
                    )

    colors = {
        key: {
            k: aggregate(k[0], k[1], frames, dict(base, sift_min_frames=10**9))
            for k, frames in table.items()
        }
        for key, table in color_frames.items()
    }
    sifts = {
        _sift_key(ratio): {
            k: aggregate(k[0], k[1], frames, base) for k, frames in table.items()
        }
        for ratio, table in sift_frames.items()
    }
    return colors, sifts


def _merge(
    color: dict[tuple[str, int], AppearanceDescriptor],
    sift: dict[tuple[str, int], AppearanceDescriptor],
) -> dict[tuple[str, int], AppearanceDescriptor]:
    """色側と SIFT 側の記述子を1つにまとめる。"""
    out = {}
    for key, c in color.items():
        s = sift.get(key)
        out[key] = AppearanceDescriptor(
            camera=key[0], track_id=key[1],
            hist_upper=c.hist_upper, hist_lower=c.hist_lower,
            sift=None if s is None else s.sift,
            sift_valid=[False] * N_JOINTS if s is None else s.sift_valid,
            n_frames_used=c.n_frames_used,
        )
    return out


def evaluate(
    candidates: list[tuple[str, Candidate]],
    correct: set[tuple[str, tuple[str, int], tuple[str, int]]],
    descriptors: dict[str, dict[tuple[str, int], AppearanceDescriptor]],
    attr: str,
    params: dict[str, Any],
) -> tuple[float, float, float]:
    """(全候補, in_range, 自己ループ) の AUC。"""
    buckets = {k: ([], []) for k in ("all", "in_range", "self_loop")}
    by_date: dict[str, list[Candidate]] = collections.defaultdict(list)
    for date, candidate in candidates:
        by_date[date].append(candidate)

    for date, items in by_date.items():
        table = descriptors.get(date) or {}
        for candidate, score in zip(items, score_pairs(items, table, params)):
            if score is None:
                continue
            value = getattr(score, attr)
            if value is None:
                continue
            is_correct = (date, candidate.tgt, candidate.src) in correct
            for name in ("all",
                         "self_loop" if candidate.is_self_loop
                         else ("in_range" if candidate.in_search_range else None)):
                if name is None:
                    continue
                buckets[name][0 if is_correct else 1].append(value)

    return tuple(auc(pos, neg) for pos, neg in
                 (buckets["all"], buckets["in_range"], buckets["self_loop"]))


def print_report(title: str, results: list[SweepResult], baseline: str | None) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(f"  {'設定':<32}{'全候補':>9}{'in_range':>10}{'自己ループ':>11}   備考")
    print("  " + "-" * 74)
    for r in sorted(results, key=lambda x: -(0.0 if np.isnan(x.auc_all) else x.auc_all)):
        mark = "  ← 現行" if r.key == baseline else ""
        print(
            f"  {r.key:<32}{r.auc_all:>9.3f}{r.auc_in_range:>10.3f}"
            f"{r.auc_self_loop:>11.3f}   {r.extra}{mark}"
        )
    best = max(results, key=lambda x: -1.0 if np.isnan(x.auc_all) else x.auc_all)
    print(f"\n  AUC（全候補）が最大: {best.key}  {best.auc_all:.3f}")
