"""ゾーンに該当しなかった端点を、正常な出入りと分断の疑いに分ける。

観測区間の境界ぎわで始まる（終わる）トラックは、区間の外から入ってきた
正常な出入りで、遷移領域を足せば拾える。区間の内側深くで始まるトラックは、
そこに出入口が無いので追跡が途切れた疑いが強い。打ち手が違うので分けて数える。

    python scripts/analyse_zone_misses.py
"""

from __future__ import annotations

import collections
import sys
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mct.config import load_cameras, load_params, load_transitions  # noqa: E402
from mct.diagnostics import _find_fragments, match_zones  # noqa: E402
from mct.evaluation import match_tracks  # noqa: E402
from mct.gt import ALL_CAMERAS, gt_transitions, load_ground_truth  # noqa: E402
from mct.io_utils import load_tracking_result  # noqa: E402

DATES = ("0720", "0727", "0730", "0803")
BOUNDARY_PX = 50.0


def distance_to_boundary(point, cam_cfg) -> float:
    """観測区間の境界までの距離。

    境界は「除外領域のふち」と「画像のふち」の両方。どちらか近い方を取る。
    """
    x, y = point
    width, height = cam_cfg.image_size or (1920, 1080)
    best = min(x, y, width - x, height - y)          # 画像のふち

    for region in cam_cfg.exclude_regions:
        pts = np.asarray(region.points, dtype=float)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        dx = max(x0 - x, 0.0, x - x1)
        dy = max(y0 - y, 0.0, y - y1)
        best = min(best, float(np.hypot(dx, dy)))
    return best


def main() -> int:
    cameras = load_cameras(REPO_ROOT / "configs" / "cameras.yaml")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        routes = load_transitions(REPO_ROOT / "configs" / "transitions.yaml", cameras)
    params = load_params(REPO_ROOT / "configs" / "params.yaml")
    ev_params = params["evaluation"]
    dir_frames = int(params["filtering"]["dir_frames"])

    near = collections.Counter()
    deep = collections.Counter()
    distances: list[float] = []
    deep_examples: list[str] = []
    in_fragments = 0
    n_endpoints = 0

    for date in DATES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gt = load_ground_truth(REPO_ROOT / "data" / "gt", date)

        to_pred, summaries, fragments = {}, {}, {}
        for camera in ALL_CAMERAS:
            source = REPO_ROOT / "outputs" / date / camera / "tracks_filtered.json"
            if not source.is_file():
                continue
            pred = load_tracking_result(source)
            for p, g in match_tracks(pred, gt, camera, ev_params).pairs.items():
                to_pred[(camera, g)] = (camera, p)
            cfg = cameras[camera]
            for tid, points in pred.tracks.items():
                summaries[(camera, tid)] = (
                    match_zones(points, cfg, dir_frames, entering=True),
                    match_zones(points, cfg, dir_frames, entering=False),
                )
            # フェーズ2と同じ判定で「分断の疑い」の組を拾う
            fragments[camera] = {
                tid for pair in _find_fragments(pred.tracks) for tid in pair[:2]
            }

        for t in gt_transitions(gt):
            tgt = to_pred.get((t.from_cam, t.from_track))
            src = to_pred.get((t.to_cam, t.to_track))
            if tgt is None or src is None:
                continue

            # 遷移の両端を見る。退出側は exit、出現側は entry。
            for key, entering in ((tgt, False), (src, True)):
                camera, tid = key
                entry_match, exit_match = summaries[key]
                match = entry_match if entering else exit_match
                if match.zones:
                    continue

                n_endpoints += 1
                cfg = cameras[camera]
                d = distance_to_boundary(match.point, cfg)
                distances.append(d)
                side = "出現" if entering else "退出"
                if d <= BOUNDARY_PX:
                    near[(camera, side)] += 1
                else:
                    deep[(camera, side)] += 1
                    if tid in fragments.get(camera, set()):
                        in_fragments += 1
                    if len(deep_examples) < 8:
                        deep_examples.append(
                            f"{date}/{camera}#{tid} {side} "
                            f"({match.point[0]:.0f},{match.point[1]:.0f}) 境界まで {d:.0f}px"
                        )

    n_near, n_deep = sum(near.values()), sum(deep.values())
    print(f"ゾーンに該当しなかった端点: {n_endpoints} 件")
    print(f"  (a) 境界から {BOUNDARY_PX:.0f}px 以内（正常な出入り）: {n_near} 件 "
          f"({100*n_near/max(1,n_endpoints):.1f}%)")
    print(f"  (b) 内側 {BOUNDARY_PX:.0f}px より深い（分断の疑い）  : {n_deep} 件 "
          f"({100*n_deep/max(1,n_endpoints):.1f}%)")
    if distances:
        a = np.array(distances)
        print(f"\n境界までの距離: 中央 {np.median(a):.0f}px / p75 {np.percentile(a,75):.0f}px "
              f"/ 最大 {a.max():.0f}px")

    print(f"\n(b) のうち、フェーズ2の「分断の疑いがあるペア」にも現れたもの: "
          f"{in_fragments} / {n_deep} 件")

    print("\n(a) の内訳（カメラ・側）:")
    for (camera, side), n in near.most_common(10):
        print(f"  {camera} {side}: {n} 件")
    if deep:
        print("\n(b) の内訳（カメラ・側）:")
        for (camera, side), n in deep.most_common(10):
            print(f"  {camera} {side}: {n} 件")
        print("\n(b) の例:")
        for e in deep_examples:
            print(f"  {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
