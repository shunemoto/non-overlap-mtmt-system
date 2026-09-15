"""トラックごとの遷移領域の該当結果を CSV に書き出す。

割合だけでは、該当しなかったトラックが定義の誤りによるものか、対象エリア外
への出入りという正常な事象かを見分けられない。1トラック1行で出し、位置と
角度を並べて確認できるようにする。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from mct.config import CameraConfig
from mct.diagnostics import PointMatch, match_zones
from mct.io_utils import output_dir
from mct.schema import TrackingResult

__all__ = ["REPORT_FILENAME", "COLUMNS", "build_rows", "write_zone_report"]

REPORT_FILENAME = "zone_report.csv"

COLUMNS = [
    "track_id", "start_frame", "end_frame", "n_points",
    "start_x", "start_y", "entry_dir_deg", "entry_zones", "entry_reason",
    "end_x", "end_y", "exit_dir_deg", "exit_zones", "exit_reason",
]


def _reason_text(match: PointMatch) -> str:
    """該当しなかった理由。角度違いなら最寄りのゾーンと定義との差を添える。"""
    if match.matched:
        return ""
    if match.reason != "angle_mismatch":
        return "outside_region"

    lo, hi = match.nearest_range or (0.0, 0.0)
    gap = match.nearest_gap
    return (
        f"angle_mismatch({match.nearest_zone} 定義[{lo:.0f},{hi:.0f}] "
        f"との差{gap:.0f}度)" if gap is not None else f"angle_mismatch({match.nearest_zone})"
    )


def build_rows(
    filtered: TrackingResult, cam_cfg: CameraConfig, params: dict[str, Any]
) -> list[dict[str, Any]]:
    dir_frames = int(params.get("dir_frames", 30))
    rows = []
    for tid, points in sorted(filtered.tracks.items(), key=lambda kv: int(kv[0])):
        entry = match_zones(points, cam_cfg, dir_frames, entering=True)
        exit_ = match_zones(points, cam_cfg, dir_frames, entering=False)
        rows.append(
            {
                "track_id": tid,
                "start_frame": points[0].frame_idx,
                "end_frame": points[-1].frame_idx,
                "n_points": len(points),
                "start_x": f"{entry.point[0]:.1f}",
                "start_y": f"{entry.point[1]:.1f}",
                "entry_dir_deg": "" if entry.direction is None else f"{entry.direction:.1f}",
                # 複数該当はすべて並べる（重複の検出が目的）
                "entry_zones": ";".join(entry.zones),
                "entry_reason": _reason_text(entry),
                "end_x": f"{exit_.point[0]:.1f}",
                "end_y": f"{exit_.point[1]:.1f}",
                "exit_dir_deg": "" if exit_.direction is None else f"{exit_.direction:.1f}",
                "exit_zones": ";".join(exit_.zones),
                "exit_reason": _reason_text(exit_),
            }
        )
    return rows


def write_zone_report(
    date: str,
    camera: str,
    filtered: TrackingResult,
    cam_cfg: CameraConfig,
    params: dict[str, Any],
    out_root: str | Path,
) -> Path:
    """`outputs/{date}/{camera}/zone_report.csv` を書き、パスを返す。"""
    rows = build_rows(filtered, cam_cfg, params)
    dest = output_dir(out_root, date, camera) / REPORT_FILENAME

    # Excel で開いても文字化けしないよう BOM 付きにする
    with dest.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    n_entry = sum(1 for r in rows if r["entry_zones"])
    n_exit = sum(1 for r in rows if r["exit_zones"])
    n_dup = sum(
        1 for r in rows if ";" in r["entry_zones"] or ";" in r["exit_zones"]
    )
    print(
        f"wrote {dest} ({len(rows)} トラック / 出現該当 {n_entry} / "
        f"退出該当 {n_exit} / 複数該当 {n_dup})"
    )
    return dest
