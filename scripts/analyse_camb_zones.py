"""camB の b_to_cf と b_to_d をどう分離するかの判断材料。

2つの遷移領域は矩形が y 730-860 で重なっており、角度の分布も近い。端点が
どちらのゾーンにも該当し得る状態なので、角度で分けるか領域で分けるかを
決める必要がある。

GT の正解遷移の端点について、座標・進行方向・現在の該当ゾーンを集め、
案1（角度で分離）と案2（領域で分離）それぞれの回収件数と重複件数を出す。

    python scripts/analyse_camb_zones.py
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mct.config import Region, load_cameras, load_params  # noqa: E402
from mct.diagnostics import track_direction  # noqa: E402
from mct.evaluation import match_tracks  # noqa: E402
from mct.geometry import angle_in_range, point_in_regions  # noqa: E402
from mct.gt import ALL_CAMERAS, gt_frame_offsets, gt_transitions, load_ground_truth  # noqa: E402
from mct.io_utils import load_tracking_result  # noqa: E402

DATES = ("0720", "0727", "0730", "0803")
ZONES = ("b_to_cf", "b_to_d")

# 現行定義（configs/cameras.yaml の camB より）
CURRENT = {
    "b_to_cf": {
        "region": [[0, 280], [360, 860]],
        "出現": (0.0, 80.0),
        "退出": (186.0, 250.0),
    },
    "b_to_d": {
        "region": [[0, 730], [750, 1080]],
        "出現": (310.0, 0.0),
        "退出": (130.0, 185.0),
    },
}

# 案1: 角度で分離する（領域はそのまま）
PLAN1 = {
    "b_to_cf": {
        "region": [[0, 280], [360, 860]],
        "出現": (340.0, 30.0),
        "退出": (170.0, 250.0),
    },
    "b_to_d": {
        "region": [[0, 730], [750, 1080]],
        "出現": (295.0, 339.0),
        "退出": (120.0, 168.0),
    },
}

# 案2: 領域で分離する（b_to_cf の下端を 860 -> 720、角度は現行のまま）
PLAN2 = {
    "b_to_cf": {
        "region": [[0, 280], [360, 720]],
        "出現": (0.0, 80.0),
        "退出": (186.0, 250.0),
    },
    "b_to_d": {
        "region": [[0, 730], [750, 1080]],
        "出現": (310.0, 0.0),
        "退出": (130.0, 185.0),
    },
}

# 案3: 角度での分離をあきらめ、b_to_cf の角度を実測に合わせる。
# 2つのゾーンの進行方向はほぼ同じで、実際に効いているのは y 位置による
# 分離である。重なるのは y 730-860 の帯だけなので、そこは重複を許す。
PLAN3 = {
    "b_to_cf": {
        "region": [[0, 280], [360, 860]],
        "出現": (330.0, 80.0),
        "退出": (145.0, 250.0),
    },
    "b_to_d": {
        "region": [[0, 730], [750, 1080]],
        "出現": (310.0, 0.0),
        "退出": (130.0, 185.0),
    },
}


@dataclass
class Endpoint:
    date: str
    track_id: int
    side: str                       # 出現 / 退出
    x: float
    y: float
    direction: float


def collect(out_root: Path, use_gt: bool) -> list[Endpoint]:
    """GT の正解遷移のうち camB 側の端点を集める。"""
    cams = load_cameras(REPO_ROOT / "configs" / "cameras.yaml")
    params = load_params(REPO_ROOT / "configs" / "params.yaml")
    dir_frames = int(params["filtering"]["dir_frames"])
    ev_params = params["evaluation"]

    out: list[Endpoint] = []
    for date in DATES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gt = load_ground_truth(
                REPO_ROOT / "data" / "gt", date, frame_offsets=gt_frame_offsets(params, date)
            )

        to_pred, tracks = {}, {}
        for camera in ALL_CAMERAS:
            source = out_root / date / camera / "tracks_filtered.json"
            if not source.is_file():
                continue
            result = load_tracking_result(source)
            tracks[camera] = result.tracks
            if use_gt:
                for tid in result.tracks:
                    to_pred[(camera, int(tid))] = int(tid)
            else:
                for pred, gt_track in match_tracks(result, gt, camera, ev_params).pairs.items():
                    to_pred[(camera, gt_track)] = int(pred)

        seen: set[tuple[int, str]] = set()
        for transition in gt_transitions(gt):
            ends = (
                (transition.from_cam, transition.from_track, False),
                (transition.to_cam, transition.to_track, True),
            )
            for camera, gt_track, entering in ends:
                if camera != "camB":
                    continue
                track_id = to_pred.get((camera, gt_track))
                if track_id is None:
                    continue
                side = "出現" if entering else "退出"
                if (track_id, side) in seen:
                    continue                      # 同じ端点が複数の遷移に現れる
                points = tracks["camB"].get(str(track_id))
                if not points:
                    continue
                direction = track_direction(points, dir_frames, entering=entering)
                if direction is None:
                    continue
                point = points[0].foot_point if entering else points[-1].foot_point
                seen.add((track_id, side))
                out.append(
                    Endpoint(date, track_id, side, point[0], point[1], direction)
                )
    return out


def _in_region(rect, x: float, y: float) -> bool:
    return point_in_regions((x, y), [Region(type="rect", points=rect)])


def matches(plan: dict, e: Endpoint) -> list[str]:
    """その端点が該当するゾーンを全部返す。"""
    hit = []
    for zone_id in ZONES:
        spec = plan[zone_id]
        if not _in_region(spec["region"], e.x, e.y):
            continue
        if angle_in_range(e.direction, *spec[e.side]):
            hit.append(zone_id)
    return hit


def in_any_region(plan: dict, e: Endpoint) -> list[str]:
    return [z for z in ZONES if _in_region(plan[z]["region"], e.x, e.y)]


def print_coordinates(endpoints: list[Endpoint]) -> None:
    print("\n【1】b_to_cf / b_to_d の領域に入る端点の y 座標分布")
    print(f"  {'ゾーン':<10}{'側':>5}{'件数':>5}{'最小':>7}{'p25':>7}"
          f"{'中央':>7}{'p75':>7}{'最大':>7}   現在の該当")
    print("  " + "-" * 74)
    for zone_id in ZONES:
        for side in ("出現", "退出"):
            sel = [
                e for e in endpoints
                if e.side == side and _in_region(CURRENT[zone_id]["region"], e.x, e.y)
            ]
            if not sel:
                continue
            ys = np.array([e.y for e in sel])
            n_hit = sum(1 for e in sel if zone_id in matches(CURRENT, e))
            print(
                f"  {zone_id:<10}{side:>5}{len(sel):>5}{ys.min():>7.0f}"
                f"{np.percentile(ys, 25):>7.0f}{np.median(ys):>7.0f}"
                f"{np.percentile(ys, 75):>7.0f}{ys.max():>7.0f}"
                f"   角度も合致 {n_hit} 件"
            )

    print("\n  端点ごとの明細（y 昇順）")
    print(f"  {'日付':<6}{'track':>6}{'側':>5}{'x':>7}{'y':>7}{'方向':>7}"
          f"   {'領域に入るゾーン':<22}現在の該当")
    print("  " + "-" * 82)
    for e in sorted(endpoints, key=lambda x: x.y):
        regions = in_any_region(CURRENT, e)
        if not regions:
            continue
        hit = matches(CURRENT, e)
        print(
            f"  {e.date:<6}{e.track_id:>6}{e.side:>5}{e.x:>7.0f}{e.y:>7.0f}"
            f"{e.direction:>6.0f}°   {'+'.join(regions):<22}"
            f"{'+'.join(hit) if hit else '(なし)'}"
        )


def print_overlap(endpoints: list[Endpoint]) -> None:
    print("\n【2】重複領域（x 0-360, y 730-860）に端点があるトラック")
    overlap = [
        e for e in endpoints
        if 0 <= e.x <= 360 and 730 <= e.y <= 860
    ]
    print(f"  {len(overlap)} 件 / camB の端点 {len(endpoints)} 件")
    if not overlap:
        return
    print(f"  {'日付':<6}{'track':>6}{'側':>5}{'x':>7}{'y':>7}{'方向':>7}   現在の該当")
    print("  " + "-" * 62)
    for e in sorted(overlap, key=lambda x: x.y):
        hit = matches(CURRENT, e)
        print(
            f"  {e.date:<6}{e.track_id:>6}{e.side:>5}{e.x:>7.0f}{e.y:>7.0f}"
            f"{e.direction:>6.0f}°   {'+'.join(hit) if hit else '(なし)'}"
        )


def print_plans(endpoints: list[Endpoint]) -> None:
    print("\n【3】案ごとの端点の該当状況（camB の端点のみ。遷移の再現率とは別）")
    print(f"  {'案':<8}{'該当あり':>9}{'該当なし':>9}{'2ゾーン重複':>12}"
          f"{'回収':>6}{'喪失':>6}")
    print("  " + "-" * 52)

    base = {(e.date, e.track_id, e.side): bool(matches(CURRENT, e)) for e in endpoints}
    for label, plan in (("現行", CURRENT), ("案1", PLAN1), ("案2", PLAN2), ("案3", PLAN3)):
        hits = {(e.date, e.track_id, e.side): matches(plan, e) for e in endpoints}
        n_hit = sum(1 for v in hits.values() if v)
        n_dup = sum(1 for v in hits.values() if len(v) > 1)
        recovered = sum(1 for k, v in hits.items() if v and not base[k])
        lost = sum(1 for k, v in hits.items() if not v and base[k])
        print(
            f"  {label:<8}{n_hit:>9}{len(endpoints) - n_hit:>9}{n_dup:>12}"
            f"{recovered:>6}{lost:>6}"
        )

    for label, plan in (("案1", PLAN1), ("案2", PLAN2), ("案3", PLAN3)):
        changed = [
            e for e in endpoints
            if bool(matches(plan, e)) != base[(e.date, e.track_id, e.side)]
        ]
        dups = [e for e in endpoints if len(matches(plan, e)) > 1]
        if changed:
            print(f"\n  {label} で変化した端点")
            for e in sorted(changed, key=lambda x: (x.side, x.y)):
                before = matches(CURRENT, e) or ["(なし)"]
                after = matches(plan, e) or ["(なし)"]
                print(
                    f"    {e.date} track{e.track_id:>3} {e.side} "
                    f"y={e.y:>4.0f} {e.direction:>4.0f}°  "
                    f"{'+'.join(before)} -> {'+'.join(after)}"
                )
        if dups:
            print(f"  {label} で2ゾーンに重複する端点")
            for e in dups:
                print(
                    f"    {e.date} track{e.track_id:>3} {e.side} "
                    f"y={e.y:>4.0f} {e.direction:>4.0f}°  {'+'.join(matches(plan, e))}"
                )


def main() -> int:
    for label, root, use_gt in (
        ("GT 側 (outputs_gt)", REPO_ROOT / "outputs_gt", True),
        ("予測側 (outputs)", REPO_ROOT / "outputs", False),
    ):
        print("\n" + "=" * 86)
        print(label)
        print("=" * 86)
        endpoints = collect(root, use_gt)
        print_coordinates(endpoints)
        print_overlap(endpoints)
        print_plans(endpoints)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
