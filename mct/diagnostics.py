"""観測区間・遷移領域の定義が妥当かを確かめるための診断。

観測区間と遷移領域は手作業で定義したばかりで、定義漏れや角度範囲の誤りが
残っている可能性が高い。「どの遷移領域にも該当しなかったトラック」を
**位置が外れた場合と角度が外れた場合に分けて数える**のは、定義漏れ
（領域を足すべき）と角度誤り（範囲を広げるべき）を切り分けるためである。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mct.config import CameraConfig
from mct.geometry import angle_in_range, direction_deg, point_in_regions
from mct.schema import FramePoint, TrackingResult

__all__ = ["ZoneReport", "CameraReport", "analyse", "print_camera_report", "print_summary"]

FRAGMENT_MAX_PX = 150.0
FRAGMENT_MAX_FRAMES = 30


@dataclass
class ZoneReport:
    """遷移領域ごとの該当数。0件なら定義が実態に合っていない。"""

    zone_id: str
    n_entry: int = 0
    n_exit: int = 0


@dataclass
class CameraReport:
    date: str
    camera: str
    n_before: int = 0
    n_after: int = 0
    n_dropped_empty: int = 0
    n_dropped_short: int = 0
    n_removed_static: int = 0
    n_under_target_saved: int = 0
    lengths: list[int] = field(default_factory=list)

    # 遷移領域に該当しなかったトラック。位置と角度で分けて数える。
    entry_outside_region: int = 0
    entry_angle_mismatch: int = 0
    exit_outside_region: int = 0
    exit_angle_mismatch: int = 0

    zones: list[ZoneReport] = field(default_factory=list)
    fragments: list[tuple[str, str, float, int]] = field(default_factory=list)

    @property
    def n_matched_entry(self) -> int:
        return self.n_after - self.entry_outside_region - self.entry_angle_mismatch

    @property
    def n_matched_exit(self) -> int:
        return self.n_after - self.exit_outside_region - self.exit_angle_mismatch


def _direction(points: list[FramePoint], dir_frames: int, *, entering: bool) -> float | None:
    """進入（または退出）方向を度で返す。動きが無ければ None。

    先頭/末尾の1フレーム差だと検出のぶれがそのまま角度になるので、
    dir_frames 分の変位を使う。
    """
    if len(points) < 2:
        return None

    if entering:
        anchor = points[0]
        target = next(
            (p for p in points if p.frame_idx - anchor.frame_idx >= dir_frames), points[-1]
        )
        dx = target.foot_point[0] - anchor.foot_point[0]
        dy = target.foot_point[1] - anchor.foot_point[1]
    else:
        anchor = points[-1]
        target = next(
            (p for p in reversed(points) if anchor.frame_idx - p.frame_idx >= dir_frames),
            points[0],
        )
        dx = anchor.foot_point[0] - target.foot_point[0]
        dy = anchor.foot_point[1] - target.foot_point[1]

    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None
    return direction_deg(dx, dy)


def _classify(
    points: list[FramePoint],
    cam_cfg: CameraConfig,
    dir_frames: int,
    *,
    entering: bool,
) -> tuple[str | None, bool]:
    """(該当した遷移領域の id, 位置は合っていたか) を返す。

    位置が合っていて角度だけ外れた場合は (None, True) になり、
    定義漏れ（位置も外れ）と区別できる。
    """
    point = points[0] if entering else points[-1]
    deg = _direction(points, dir_frames, entering=entering)

    position_matched = False
    for zone in cam_cfg.transition_zones:
        if not point_in_regions(point.foot_point, [zone.region]):
            continue
        position_matched = True

        rng = zone.entry_dir if entering else zone.exit_dir
        if rng is None:              # 角度の指定が無ければ位置だけで合致とみなす
            return zone.id, True
        if deg is not None and angle_in_range(deg, *rng):
            return zone.id, True

    return None, position_matched


def _find_fragments(
    tracks: dict[str, list[FramePoint]]
) -> list[tuple[str, str, float, int]]:
    """分断の疑いがあるペアを探す。

    あるトラックの終了直後に、近い位置で別のトラックが始まっていれば、
    同一人物が ID を振り直された可能性が高い。
    """
    ends = [(tid, ps[-1]) for tid, ps in tracks.items() if ps]
    starts = [(tid, ps[0]) for tid, ps in tracks.items() if ps]

    found = []
    for end_id, end in ends:
        for start_id, start in starts:
            if start_id == end_id:
                continue
            gap = start.frame_idx - end.frame_idx
            if not 0 <= gap <= FRAGMENT_MAX_FRAMES:
                continue
            dist = float(
                np.hypot(
                    start.foot_point[0] - end.foot_point[0],
                    start.foot_point[1] - end.foot_point[1],
                )
            )
            if dist <= FRAGMENT_MAX_PX:
                found.append((end_id, start_id, dist, gap))
    return sorted(found, key=lambda x: (x[3], x[2]))


def analyse(
    date: str,
    camera: str,
    filtered: TrackingResult,
    region_stats: dict[str, Any],
    removed_static: list[dict[str, Any]],
    cam_cfg: CameraConfig,
    params: dict[str, Any],
) -> CameraReport:
    """1カメラ分の診断をまとめる。"""
    dir_frames = int(params.get("dir_frames", 30))
    report = CameraReport(
        date=date,
        camera=camera,
        n_before=region_stats["n_tracks_before"],
        n_after=len(filtered.tracks),
        n_dropped_empty=len(region_stats["dropped_empty"]),
        n_dropped_short=len(region_stats["dropped_short"]),
        n_removed_static=len(removed_static),
        n_under_target_saved=region_stats["n_under_target_saved"],
        lengths=sorted(len(ps) for ps in filtered.tracks.values()),
        zones=[ZoneReport(z.id) for z in cam_cfg.transition_zones],
    )
    by_id = {z.zone_id: z for z in report.zones}

    for points in filtered.tracks.values():
        for entering in (True, False):
            zone_id, position_matched = _classify(points, cam_cfg, dir_frames, entering=entering)
            if zone_id is not None:
                counter = by_id[zone_id]
                if entering:
                    counter.n_entry += 1
                else:
                    counter.n_exit += 1
            elif position_matched:      # 位置は合うが角度が範囲外
                if entering:
                    report.entry_angle_mismatch += 1
                else:
                    report.exit_angle_mismatch += 1
            else:                       # そもそも領域の外
                if entering:
                    report.entry_outside_region += 1
                else:
                    report.exit_outside_region += 1

    report.fragments = _find_fragments(filtered.tracks)
    return report


def _quantiles(values: list[int]) -> str:
    if not values:
        return "-"
    a = np.asarray(values)
    return (
        f"{a.min()}/{int(np.percentile(a, 25))}/{int(statistics.median(values))}/"
        f"{int(np.percentile(a, 75))}/{a.max()}"
    )


def print_camera_report(report: CameraReport, removed_static: list[dict[str, Any]]) -> None:
    r = report
    print(f"\n--- {r.date}/{r.camera} ---")
    print(f"  トラック数        : {r.n_before} -> {r.n_after}")
    print(
        f"    削除の内訳      : 観測区間で全点除外 {r.n_dropped_empty} / "
        f"min_track_len 未満 {r.n_dropped_short} / 静止誤検出 {r.n_removed_static}"
    )
    print(f"  トラック長        : {_quantiles(r.lengths)}  (最小/p25/中央/p75/最大)")
    print(f"  saved が30枚未満  : {r.n_under_target_saved} トラック")

    total = max(1, r.n_after)
    print(
        f"  遷移領域に不一致  : 出現 {r.entry_outside_region + r.entry_angle_mismatch}"
        f"（位置外 {r.entry_outside_region} / 角度外 {r.entry_angle_mismatch}）"
        f" = {100*(r.entry_outside_region + r.entry_angle_mismatch)/total:.0f}%"
    )
    print(
        f"                      退出 {r.exit_outside_region + r.exit_angle_mismatch}"
        f"（位置外 {r.exit_outside_region} / 角度外 {r.exit_angle_mismatch}）"
        f" = {100*(r.exit_outside_region + r.exit_angle_mismatch)/total:.0f}%"
    )

    for zone in r.zones:
        mark = "  ⚠ 該当0件" if zone.n_entry == 0 and zone.n_exit == 0 else ""
        print(f"    {zone.zone_id:<12} 出現 {zone.n_entry:>4} / 退出 {zone.n_exit:>4}{mark}")

    if removed_static:
        print(f"  静止誤検出として除外した {len(removed_static)} 件:")
        for info in removed_static[:10]:
            lo, hi = info["frame_range"]
            print(
                f"    track {info['track_id']:<5} f{lo}-{hi} ({info['n_points']}点) "
                f"移動 {info['distance']:.2f}{info['unit']} "
                f"散らばり {info['spread']:.2f}{info['unit']} "
                f"conf {info['conf_median']:.2f}  [{' / '.join(info['reasons'])}]"
            )
        if len(removed_static) > 10:
            print(f"    ... 他 {len(removed_static) - 10} 件")

    if r.fragments:
        print(f"  分断の疑い {len(r.fragments)} 組（150px 以内・30フレーム以内）:")
        for end_id, start_id, dist, gap in r.fragments[:10]:
            print(f"    track {end_id} の終了 -> track {start_id} の開始  {dist:.0f}px / {gap}フレーム")
        if len(r.fragments) > 10:
            print(f"    ... 他 {len(r.fragments) - 10} 組")


def print_summary(reports: list[CameraReport]) -> None:
    """全カメラの集計表。"""
    if not reports:
        return

    print("\n" + "=" * 100)
    print("全カメラ集計")
    print("=" * 100)
    head = (
        f"{'date':<6}{'cam':<6}{'前':>5}{'後':>5}{'全除外':>7}{'短い':>6}{'静止':>6}"
        f"{'長さ 最小/p25/中央/p75/最大':>26}{'出現不一致':>12}{'退出不一致':>12}"
    )
    print(head)
    print("-" * len(head))
    for r in reports:
        print(
            f"{r.date:<6}{r.camera:<6}{r.n_before:>5}{r.n_after:>5}{r.n_dropped_empty:>7}"
            f"{r.n_dropped_short:>6}{r.n_removed_static:>6}{_quantiles(r.lengths):>26}"
            f"{r.entry_outside_region:>6}/{r.entry_angle_mismatch:<5}"
            f"{r.exit_outside_region:>6}/{r.exit_angle_mismatch:<5}"
        )

    n_after = sum(r.n_after for r in reports)
    eo = sum(r.entry_outside_region for r in reports)
    ea = sum(r.entry_angle_mismatch for r in reports)
    xo = sum(r.exit_outside_region for r in reports)
    xa = sum(r.exit_angle_mismatch for r in reports)
    print("-" * len(head))
    print(
        f"{'合計':<12}{sum(r.n_before for r in reports):>5}{n_after:>5}"
        f"{sum(r.n_dropped_empty for r in reports):>7}"
        f"{sum(r.n_dropped_short for r in reports):>6}"
        f"{sum(r.n_removed_static for r in reports):>6}"
        f"{'':>26}{eo:>6}/{ea:<5}{xo:>6}/{xa:<5}"
    )

    if n_after:
        print(
            f"\n遷移領域に該当しなかった割合: "
            f"出現 {100*(eo+ea)/n_after:.1f}%（位置外 {100*eo/n_after:.1f}% / "
            f"角度外 {100*ea/n_after:.1f}%） / "
            f"退出 {100*(xo+xa)/n_after:.1f}%（位置外 {100*xo/n_after:.1f}% / "
            f"角度外 {100*xa/n_after:.1f}%）"
        )

    empty = [
        f"{r.camera}/{z.zone_id}"
        for r in reports
        for z in r.zones
        if z.n_entry == 0 and z.n_exit == 0
    ]
    if empty:
        counts: dict[str, int] = {}
        for name in empty:
            counts[name] = counts.get(name, 0) + 1
        print(f"\n⚠ 該当0件の遷移領域（日付数）: " + ", ".join(
            f"{k}×{v}" for k, v in sorted(counts.items())
        ))
    else:
        print("\n該当0件の遷移領域はありません。")
