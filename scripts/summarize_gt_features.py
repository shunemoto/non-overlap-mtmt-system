"""GT 由来の特徴データを集計する。

観測区間を適用したあとに何トラック残り、そこから何件の正解遷移が構成できるかが
再同定単体評価の上限になる。予測側（outputs/）との比較も出す。

    python scripts/summarize_gt_features.py
"""

from __future__ import annotations

import collections
import statistics
import sys
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mct.config import load_cameras, load_params  # noqa: E402
from mct.geometry import point_in_regions  # noqa: E402
from mct.gt import ALL_CAMERAS, gt_frame_offsets, gt_transitions, load_ground_truth  # noqa: E402
from mct.io_utils import load_tracking_result, mask_path  # noqa: E402

DATES = ("0720", "0727", "0730", "0803")
OUT_GT = REPO_ROOT / "outputs_gt"


def main() -> int:
    cams = load_cameras(REPO_ROOT / "configs" / "cameras.yaml")
    params = load_params(REPO_ROOT / "configs" / "params.yaml")

    print(f"{'date':<7}{'cam':<6}{'GT前':>6}{'GT後':>6}{'除外点':>9}{'saved':>7}"
          f"{'骨格なし':>9}{'マスクなし':>11}")
    print("-" * 64)

    totals = collections.Counter()
    selected_counts: list[int] = []
    survivors: dict[str, dict[tuple[str, int], int]] = {}

    for date in DATES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gt = load_ground_truth(
                REPO_ROOT / "data" / "gt", date, frame_offsets=gt_frame_offsets(params, date)
            )
        survivors[date] = {}
        for camera in ALL_CAMERAS:
            cfg = cams[camera]
            n_before = len(gt.evaluable(camera))
            pts_before = sum(len(gt.boxes[camera][t]) for t in gt.evaluable(camera))
            pts_after = 0
            for t in gt.evaluable(camera):
                pts_after += sum(
                    1 for b in gt.boxes[camera][t]
                    if not point_in_regions(
                        (b.bbox[0] + b.bbox[2] / 2, b.bbox[1] + b.bbox[3]), cfg.exclude_regions
                    )
                )

            path = OUT_GT / date / camera / "tracks_filtered.json"
            n_after = n_saved = n_no_kp = n_no_mask = 0
            if path.is_file():
                result = load_tracking_result(path)
                n_after = len(result.tracks)
                for tid, points in result.tracks.items():
                    saved = [p for p in points if p.saved]
                    selected_counts.append(len(saved))
                    survivors[date][(camera, int(tid))] = len(points)
                    n_saved += len(saved)
                    n_no_kp += sum(1 for p in saved if p.keypoints is None)
                    n_no_mask += sum(
                        1 for p in saved if not mask_path(path.parent, tid, p.frame_idx).is_file()
                    )

            print(
                f"{date:<7}{camera:<6}{n_before:>6}{n_after:>6}"
                f"{pts_before - pts_after:>9}{n_saved:>7}{n_no_kp:>9}{n_no_mask:>11}"
            )
            for key, value in (
                ("before", n_before), ("after", n_after), ("saved", n_saved),
                ("no_kp", n_no_kp), ("no_mask", n_no_mask),
                ("pts_before", pts_before), ("pts_after", pts_after),
            ):
                totals[key] += value

    print("-" * 64)
    print(
        f"{'合計':<13}{totals['before']:>6}{totals['after']:>6}"
        f"{totals['pts_before']-totals['pts_after']:>9}{totals['saved']:>7}"
        f"{totals['no_kp']:>9}{totals['no_mask']:>11}"
    )
    if totals["saved"]:
        print(
            f"\n骨格点が取れなかった割合: {100*totals['no_kp']/totals['saved']:.2f}%"
            f" / マスク: {100*totals['no_mask']/totals['saved']:.2f}%"
        )
    if selected_counts:
        a = np.array(selected_counts)
        print(
            f"選抜フレーム数: 最小 {a.min()} / 中央 {int(np.median(a))} / 最大 {a.max()}"
            f" / 30枚に達したトラック {int((a == 30).sum())} / {len(a)}"
        )

    _print_transitions(survivors, params)
    return 0


def _print_transitions(survivors, params) -> None:
    """残ったトラックから構成できる正解遷移の件数。単体評価の上限になる。"""
    print(f"\n{'date':<7}{'残トラック':>11}{'正解遷移':>10}{'GT 全体':>10}")
    print("-" * 40)
    total_kept = total_tr = total_all = 0
    for date in DATES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gt = load_ground_truth(
                REPO_ROOT / "data" / "gt", date, frame_offsets=gt_frame_offsets(params, date)
            )
        all_tr = len(gt_transitions(gt))
        kept = survivors[date]

        by_gid: dict[str, list[tuple[int, str, int]]] = collections.defaultdict(list)
        for (camera, track_id) in kept:
            gid = gt.global_ids.get((camera, track_id))
            if gid:
                boxes = gt.boxes[camera][track_id]
                by_gid[gid].append((boxes[0].frame_idx, camera, track_id))
        n_tr = sum(max(0, len(v) - 1) for v in by_gid.values())

        print(f"{date:<7}{len(kept):>11}{n_tr:>10}{all_tr:>10}")
        total_kept += len(kept); total_tr += n_tr; total_all += all_tr

    print("-" * 40)
    print(f"{'合計':<7}{total_kept:>11}{total_tr:>10}{total_all:>10}")
    print(
        f"\n再同定単体評価の上限: 正解遷移 {total_tr} / {total_all} = "
        f"{100*total_tr/max(1,total_all):.1f}%"
    )


if __name__ == "__main__":
    raise SystemExit(main())
