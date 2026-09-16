"""遷移領域の角度範囲を実測から見直すための分析。

位置は遷移領域に入っているのに角度が範囲外で落ちた端点を一覧にし、ゾーンごとの
実測角度の分布と現在の定義範囲を突き合わせる。範囲を広げた場合に何件回収でき、
その代わりに他のゾーンとの重複がどれだけ増えるかも試算する。

角度は円周上の量なので、平均や分位点は素直に計算できない（350度と10度の平均は
180度ではなく0度）。円周平均を基準に ±180度へ展開してから分位点を取る。

    python scripts/analyse_zone_angles.py [outputs|outputs_gt|both]
"""

from __future__ import annotations

import collections
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mct.config import load_cameras, load_params  # noqa: E402
from mct.diagnostics import angle_gap, circular_diff, match_zones  # noqa: E402
from mct.evaluation import match_tracks  # noqa: E402
from mct.geometry import angle_in_range, point_in_regions  # noqa: E402
from mct.gt import ALL_CAMERAS, gt_frame_offsets, gt_transitions, load_ground_truth  # noqa: E402
from mct.io_utils import load_tracking_result  # noqa: E402

DATES = ("0720", "0727", "0730", "0803")
PERCENTILES = (5, 25, 50, 75, 95)


@dataclass
class Endpoint:
    """GT の遷移の端点1つ分。"""

    date: str
    camera: str
    track_id: int
    side: str                    # 出現 / 退出
    angle: float
    zone_id: str
    zone_range: tuple[float, float]
    matched: bool

    @property
    def gap(self) -> float:
        return angle_gap(self.angle, *self.zone_range)


@dataclass
class ZoneSamples:
    camera: str
    zone_id: str
    side: str
    zone_range: tuple[float, float]
    matched: list[float] = field(default_factory=list)
    missed: list[float] = field(default_factory=list)

    @property
    def all_angles(self) -> list[float]:
        return self.matched + self.missed


def circular_percentiles(angles: list[float], qs=PERCENTILES) -> dict[int, float]:
    """円周上の分位点。円周平均を基準に ±180度へ展開してから求める。"""
    if not angles:
        return {}
    radians = np.radians(angles)
    mean = float(np.degrees(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())) % 360)
    offsets = [((a - mean + 180) % 360) - 180 for a in angles]
    return {q: float((mean + np.percentile(offsets, q)) % 360) for q in qs}


def _load_side(out_root: Path, use_gt: bool):
    """(端点の一覧, ゾーンごとの標本) を返す。"""
    cams = load_cameras(REPO_ROOT / "configs" / "cameras.yaml")
    params = load_params(REPO_ROOT / "configs" / "params.yaml")
    dir_frames = int(params["filtering"]["dir_frames"])
    ev_params = params["evaluation"]

    endpoints: list[Endpoint] = []
    samples: dict[tuple[str, str, str], ZoneSamples] = {}

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
                    to_pred[(camera, int(tid))] = (camera, int(tid))
            else:
                for pred, gt_track in match_tracks(result, gt, camera, ev_params).pairs.items():
                    to_pred[(camera, gt_track)] = (camera, int(pred))

        for transition in gt_transitions(gt):
            ends = (
                (to_pred.get((transition.from_cam, transition.from_track)), False),
                (to_pred.get((transition.to_cam, transition.to_track)), True),
            )
            for key, entering in ends:
                if key is None:
                    continue
                camera, track_id = key
                points = tracks.get(camera, {}).get(str(track_id))
                if not points:
                    continue

                cfg = cams[camera]
                match = match_zones(points, cfg, dir_frames, entering=entering)
                if match.direction is None:
                    continue
                side = "出現" if entering else "退出"

                # 位置が入っているゾーンすべてについて標本に加える
                for zone in cfg.transition_zones:
                    if not point_in_regions(match.point, [zone.region]):
                        continue
                    rng = zone.entry_dir if entering else zone.exit_dir
                    if rng is None:
                        continue
                    bucket = samples.setdefault(
                        (camera, zone.id, side), ZoneSamples(camera, zone.id, side, rng)
                    )
                    hit = angle_in_range(match.direction, *rng)
                    (bucket.matched if hit else bucket.missed).append(match.direction)

                if match.zones:
                    continue                       # 角度も合っている端点は一覧に出さない
                if match.reason != "angle_mismatch":
                    continue
                endpoints.append(
                    Endpoint(
                        date=date, camera=camera, track_id=track_id, side=side,
                        angle=match.direction, zone_id=match.nearest_zone or "?",
                        zone_range=match.nearest_range or (0.0, 0.0), matched=False,
                    )
                )
    return endpoints, samples


def print_endpoints(endpoints: list[Endpoint]) -> None:
    print(f"\n角度が範囲外で落ちた端点: {len(endpoints)} 件")
    print(f"  {'日付':<6}{'カメラ':<7}{'track':>6}{'側':>5}{'実測角度':>9}"
          f"  {'ゾーンid':<13}{'定義範囲':>14}{'差(度)':>8}")
    print("  " + "-" * 74)
    for e in sorted(endpoints, key=lambda x: (x.camera, x.zone_id, x.side, x.date)):
        lo, hi = e.zone_range
        print(
            f"  {e.date:<6}{e.camera:<7}{e.track_id:>6}{e.side:>5}{e.angle:>8.0f}°"
            f"  {e.zone_id:<13}{f'[{lo:.0f}, {hi:.0f}]':>14}{e.gap:>7.0f}°"
        )


def print_zone_distributions(samples: dict) -> None:
    print("\nゾーンごとの実測角度（GT の正解遷移の端点。位置がそのゾーンに入るもの）")
    header = (
        f"  {'カメラ':<7}{'ゾーンid':<13}{'側':>5}{'該当':>5}{'角度外':>7}"
        f"{'定義範囲':>13}   p5/p25/中央/p75/p95"
    )
    print(header)
    print("  " + "-" * (len(header) + 8))
    for key in sorted(samples):
        s = samples[key]
        pct = circular_percentiles(s.all_angles)
        lo, hi = s.zone_range
        spread = "  ".join(f"{pct[q]:.0f}" for q in PERCENTILES) if pct else "-"
        print(
            f"  {s.camera:<7}{s.zone_id:<13}{s.side:>5}{len(s.matched):>5}"
            f"{len(s.missed):>7}{f'[{lo:.0f}, {hi:.0f}]':>13}   {spread}"
        )
        if s.missed:
            print(f"        角度外の実測: "
                  + ", ".join(f"{a:.0f}°" for a in sorted(s.missed)))


def simulate_widening(samples: dict, cams) -> None:
    """定義範囲を p5〜p95 に広げた場合の回収数と、重複の増減。"""
    proposed: dict[tuple[str, str, str], tuple[float, float]] = {}
    recovered = 0
    for key, s in samples.items():
        pct = circular_percentiles(s.all_angles)
        if not pct:
            continue
        new = (pct[5], pct[95])
        proposed[key] = new
        recovered += sum(1 for a in s.missed if angle_in_range(a, *new))

    print(f"\n定義範囲を p5〜p95 に広げた場合")
    print(f"  回収できる端点: {recovered} / {sum(len(s.missed) for s in samples.values())} 件")

    print(f"\n  {'カメラ':<7}{'ゾーンid':<13}{'側':>5}{'現在':>14}{'提案':>14}{'幅の変化':>11}")
    print("  " + "-" * 66)
    for key in sorted(proposed):
        s = samples[key]
        lo, hi = s.zone_range
        nlo, nhi = proposed[key]
        width_now = (hi - lo) % 360
        width_new = (nhi - nlo) % 360
        print(
            f"  {s.camera:<7}{s.zone_id:<13}{s.side:>5}"
            f"{f'[{lo:.0f}, {hi:.0f}]':>14}{f'[{nlo:.0f}, {nhi:.0f}]':>14}"
            f"{f'{width_now:.0f}° -> {width_new:.0f}°':>11}"
        )

    _check_overlap(samples, proposed, cams)


def _check_overlap(samples: dict, proposed: dict, cams) -> None:
    """広げた範囲が、領域の重なる別ゾーンと角度でも重なるか。

    領域が離れていれば角度が重なっても実害は無い。領域が重なっている対だけを
    見て、角度まで重なるなら1つの端点が2つのゾーンに該当し得る。
    """
    print("\n  重複の確認（領域が重なるゾーン対のみ）")
    found = False
    for camera, cfg in cams.items():
        zones = cfg.transition_zones
        for i, a in enumerate(zones):
            for b in zones[i + 1:]:
                if not _regions_overlap(a.region, b.region):
                    continue
                for side in ("出現", "退出"):
                    ka = (camera, a.id, side)
                    kb = (camera, b.id, side)
                    if ka not in proposed or kb not in proposed:
                        continue
                    now = _ranges_overlap(samples[ka].zone_range, samples[kb].zone_range)
                    new = _ranges_overlap(proposed[ka], proposed[kb])
                    if new:
                        found = True
                        change = "（現在も重複）" if now else "  ← 新たに重複"
                        print(f"    {camera} {a.id} と {b.id}（{side}）{change}")
    if not found:
        print("    領域と角度の両方が重なるゾーン対はありません")


def _regions_overlap(a, b) -> bool:
    pa, pb = np.asarray(a.points, float), np.asarray(b.points, float)
    ax0, ay0 = pa.min(axis=0); ax1, ay1 = pa.max(axis=0)
    bx0, by0 = pb.min(axis=0); bx1, by1 = pb.max(axis=0)
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def _ranges_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """2つの角度範囲が重なるか。1度刻みで見る。"""
    return any(
        angle_in_range(float(d), *a) and angle_in_range(float(d), *b)
        for d in range(0, 360)
    )


def main(argv: list[str]) -> int:
    which = argv[1] if len(argv) > 1 else "both"
    cams = load_cameras(REPO_ROOT / "configs" / "cameras.yaml")

    for label, root, use_gt in (
        ("予測側 (outputs)", REPO_ROOT / "outputs", False),
        ("GT 側 (outputs_gt)", REPO_ROOT / "outputs_gt", True),
    ):
        if which not in ("both", root.name):
            continue
        print("\n" + "=" * 86)
        print(label)
        print("=" * 86)
        endpoints, samples = _load_side(root, use_gt)
        print_endpoints(endpoints)
        print_zone_distributions(samples)
        simulate_widening(samples, cams)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
