"""正解遷移からカメラ間距離を校正する。

`transitions.yaml` の `distance_m` は現地で測った直線距離だが、実際の通行は
曲がったり回り込んだりするので、到着時刻の予測が一貫して早すぎる方向に
ずれていた。正解遷移から距離を逆算し、実効的な距離に直す。

これは GT を使ったパラメータ調整なので、`holdout_date` で1日を除いて校正し、
除いた日で評価できるようにしてある（leave-one-day-out）。
"""

from __future__ import annotations

import collections
import csv
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mct.candidates import Candidate
from mct.config import TransitionRoute
from mct.summarize import TrackSummary, harmonic_mean_speed

__all__ = ["ZoneCalibration", "collect_samples", "calibrate", "write_csv", "write_yaml"]

CSV_FILENAME = "distance_calibration.csv"
YAML_FILENAME = "transitions_calibrated.yaml"


@dataclass
class ZoneCalibration:
    """1ゾーン対の校正結果。

    サンプルは (調和平均の速度 [m/s], 実際の経過フレーム数) の対で持つ。
    ここから逆算距離も校正前後の誤差も導けるので、別々に貯めなくて済む。
    """

    from_cam: str
    from_zone: str
    to_cam: str
    to_zone: str
    current_distance_m: float
    fps: float = 30.0
    samples: list[tuple[float, int]] = field(default_factory=list)
    adopted: bool = False

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    @property
    def implied(self) -> list[float]:
        """各サンプルから逆算した距離 [m]。"""
        return [speed * (delta / self.fps) for speed, delta in self.samples]

    @property
    def median(self) -> float | None:
        values = self.implied
        return statistics.median(values) if values else None

    @property
    def p25(self) -> float | None:
        return _percentile(self.implied, 25)

    @property
    def p75(self) -> float | None:
        return _percentile(self.implied, 75)

    def errors(self, distance_m: float) -> list[float]:
        """その距離を使ったときの到着予測の誤差 [フレーム]。"""
        return [distance_m / speed * self.fps - delta for speed, delta in self.samples]

    @property
    def current_error_median(self) -> float | None:
        values = self.errors(self.current_distance_m)
        return statistics.median(values) if values else None

    @property
    def adopted_distance_m(self) -> float:
        return self.median if (self.adopted and self.median is not None) else self.current_distance_m

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.from_cam, self.from_zone, self.to_cam, self.to_zone)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100.0
    lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def collect_samples(
    candidates: list[Candidate],
    summaries: dict[tuple[str, int], TrackSummary],
    correct_pairs: set[tuple[tuple[str, int], tuple[str, int]]],
    fps: float,
) -> dict[tuple[str, str, str, str], ZoneCalibration]:
    """正解と分かっている候補から、距離の逆算値を集める。

    自己ループは除く。居室に滞在する時間は距離と速度から決まらないので、
    逆算すると滞在時間がそのまま距離になってしまう。
    """
    out: dict[tuple[str, str, str, str], ZoneCalibration] = {}

    for c in candidates:
        if c.route is None or c.is_self_loop:
            continue
        if (c.tgt, c.src) not in correct_pairs:
            continue
        tgt, src = summaries.get(c.tgt), summaries.get(c.src)
        if tgt is None or src is None:
            continue
        speed = harmonic_mean_speed(tgt.speed_mps, src.speed_mps)
        if not speed or speed <= 0 or c.delta_frames <= 0:
            continue

        key = (c.route.from_cam, c.route.from_zone, c.route.to_cam, c.route.to_zone)
        entry = out.setdefault(
            key,
            ZoneCalibration(*key, current_distance_m=c.route.distance_m, fps=fps),
        )
        entry.samples.append((speed, c.delta_frames))
    return out


def calibrate(
    per_date_samples: list[dict[tuple[str, str, str, str], ZoneCalibration]],
    routes: list[TransitionRoute],
    min_samples: int,
) -> dict[tuple[str, str, str, str], ZoneCalibration]:
    """日付をまたいでサンプルを束ね、採用の可否を決める。

    サンプルが少ないゾーン対で中央値を取ると1〜2件の外れ値に引きずられるので、
    `min_samples` に満たなければ手で測った値を残す。
    """
    merged: dict[tuple[str, str, str, str], ZoneCalibration] = {}
    for samples in per_date_samples:
        for key, entry in samples.items():
            target = merged.setdefault(
                key,
                ZoneCalibration(
                    *key, current_distance_m=entry.current_distance_m, fps=entry.fps
                ),
            )
            target.samples.extend(entry.samples)

    # サンプルが1件も無かった経路も、現状の値として表に載せる
    for r in routes:
        if r.from_cam == r.to_cam and r.from_zone == r.to_zone:
            continue
        merged.setdefault(
            (r.from_cam, r.from_zone, r.to_cam, r.to_zone),
            ZoneCalibration(
                r.from_cam, r.from_zone, r.to_cam, r.to_zone,
                current_distance_m=r.distance_m,
            ),
        )

    for entry in merged.values():
        entry.adopted = entry.n_samples >= min_samples
    return merged


def write_csv(
    calibrations: dict[tuple[str, str, str, str], ZoneCalibration],
    out_root: str | Path,
) -> Path:
    dest = Path(out_root) / CSV_FILENAME
    dest.parent.mkdir(parents=True, exist_ok=True)

    def fmt(v: float | None) -> str:
        return "" if v is None else f"{v:.2f}"

    with dest.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "from_cam", "from_zone", "to_cam", "to_zone", "n_samples",
            "current_distance_m", "implied_distance_m_median",
            "implied_p25", "implied_p75", "current_error_frames_median", "adopted",
        ])
        for key in sorted(calibrations):
            e = calibrations[key]
            writer.writerow([
                e.from_cam, e.from_zone, e.to_cam, e.to_zone, e.n_samples,
                fmt(e.current_distance_m), fmt(e.median), fmt(e.p25), fmt(e.p75),
                fmt(e.current_error_median), e.adopted,
            ])
    return dest


def write_yaml(
    calibrations: dict[tuple[str, str, str, str], ZoneCalibration],
    routes: list[TransitionRoute],
    dest: str | Path,
) -> Path:
    """校正後の距離を入れた transitions を別ファイルとして書く。

    元の `transitions.yaml` は手で測った記録なので変更しない。
    """
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# 正解遷移から距離を校正した transitions。",
        "# scripts ではなく `python -m mct.cli calibrate-distance` が生成する。",
        "# distance_m 以外は configs/transitions.yaml と同じ。",
        "# adopted=False のゾーン対（サンプル数が足りない）は手動値のまま。",
        "",
        "transitions:",
    ]
    for r in routes:
        key = (r.from_cam, r.from_zone, r.to_cam, r.to_zone)
        entry = calibrations.get(key)
        distance = entry.adopted_distance_m if entry else r.distance_m
        note = ""
        if entry and entry.adopted:
            note = f"  # 校正済み（{entry.n_samples} 件、元 {r.distance_m:g}）"
        hi = "null" if r.search_range_sec[1] is None else f"{r.search_range_sec[1]:g}"
        lines += [
            f"  - from: {{cam: {r.from_cam}, zone: {r.from_zone}}}",
            f"    to: {{cam: {r.to_cam}, zone: {r.to_zone}}}",
            f"    distance_m: {distance:.1f}{note}",
            f"    search_range_sec: [{r.search_range_sec[0]:g}, {hi}]",
            f"    c_dwell: {r.c_dwell:g}",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def print_report(
    calibrations: dict[tuple[str, str, str, str], ZoneCalibration],
    min_samples: int,
) -> None:
    adopted = [e for e in calibrations.values() if e.adopted]
    kept = [e for e in calibrations.values() if not e.adopted]

    print(f"\n距離の校正（自己ループを除く {len(calibrations)} ゾーン対）")
    print(f"  採用 {len(adopted)} / 手動値のまま {len(kept)}（サンプル {min_samples} 件未満）")

    before = [v for e in calibrations.values() for v in e.errors(e.current_distance_m)]
    after = [v for e in calibrations.values() for v in e.errors(e.adopted_distance_m)]
    if before:
        print(
            f"  予測誤差の中央値: 校正前 {statistics.median(before):+.0f} フレーム "
            f"-> 校正後 {statistics.median(after):+.0f} フレーム"
        )
        print(
            f"  誤差の絶対値の中央値: "
            f"{statistics.median(map(abs, before)):.0f} -> "
            f"{statistics.median(map(abs, after)):.0f} フレーム"
        )
    if adopted:
        print("\n  採用したゾーン対（現在値との差が大きい順）:")
        for e in sorted(adopted, key=lambda x: -abs((x.median or 0) - x.current_distance_m))[:12]:
            ratio = (e.median or 0) / e.current_distance_m if e.current_distance_m else 0
            print(
                f"    {e.from_cam}/{e.from_zone} -> {e.to_cam}/{e.to_zone:<12} "
                f"{e.n_samples:>3} 件  {e.current_distance_m:5.1f} -> {e.median:5.1f} m  "
                f"(差 {e.median - e.current_distance_m:+5.1f} m / 比 {ratio:.2f})"
            )
    if kept:
        print("\n  手動値のまま残したゾーン対:")
        for e in sorted(kept, key=lambda x: -x.n_samples)[:12]:
            print(
                f"    {e.from_cam}/{e.from_zone} -> {e.to_cam}/{e.to_zone:<12} "
                f"{e.n_samples:>3} 件  {e.current_distance_m:5.1f} m"
            )
