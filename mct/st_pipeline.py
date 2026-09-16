"""時空間尤度の実行と σ の校正。

候補と要約を読み、σ を当てて `outputs/{date}/st_scores.json` を書く。
σ は GT の正解遷移から推定するので、距離校正と同様に `holdout_date` で
1日を除いて推定できる（leave-one-day-out）。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from mct.cache import cache_key
from mct.candidates import Candidate
from mct.config import TransitionRoute
from mct.spatiotemporal import (
    SigmaEstimate,
    SigmaTable,
    STScore,
    collect_errors,
    estimate_sigma,
    score_candidates,
)
from mct.summarize import TrackSummary

__all__ = [
    "SCORES_FILENAME",
    "SIGMA_CSV",
    "load_candidates",
    "write_scores",
    "write_sigma_csv",
    "print_sigma_report",
]

SCORES_FILENAME = "st_scores.json"
SIGMA_CSV = "sigma_calibration.csv"


def load_candidates(
    date: str, out_root: str | Path
) -> tuple[list[TrackSummary], list[Candidate], dict[str, Any]]:
    """`candidates.json` を読み戻す。

    候補の列挙をやり直さずに済ませるため、フェーズ3の出力をそのまま使う。
    """
    path = Path(out_root) / date / "candidates.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} がありません。先に candidates を実行してください。"
        )
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    summaries = [TrackSummary.from_dict(d) for d in payload["summaries"]]
    candidates = [_candidate_from_dict(d) for d in payload["candidates"]]
    return summaries, candidates, payload.get("meta") or {}


def _candidate_from_dict(d: dict[str, Any]) -> Candidate:
    raw = d.get("route")
    route = None
    if raw:
        lo, hi = raw["search_range_sec"]
        route = TransitionRoute(
            from_cam=raw["from_cam"], from_zone=raw["from_zone"],
            to_cam=raw["to_cam"], to_zone=raw["to_zone"],
            distance_m=float(raw["distance_m"]),
            search_range_sec=(float(lo), None if hi is None else float(hi)),
            c_dwell=float(raw["c_dwell"]),
        )
    return Candidate(
        src=(d["src"]["cam"], int(d["src"]["track"])),
        tgt=(d["tgt"]["cam"], int(d["tgt"]["track"])),
        src_zone=d.get("src_zone", ""), tgt_zone=d.get("tgt_zone", ""),
        route=route,
        delta_frames=int(d["delta_frames"]),
        in_search_range=bool(d["in_search_range"]),
    )


def write_scores(
    date: str,
    out_root: str | Path,
    scores: list[STScore],
    sigma: SigmaTable,
    params: dict[str, Any],
) -> Path:
    dest = Path(out_root) / date / SCORES_FILENAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "date": date,
            "n_candidates": len(scores),
            "sigma_default_frames": sigma.default,
            "n_sigma_per_zone": len(sigma.per_zone),
            "config_hash": cache_key(
                {
                    "spatiotemporal": dict(sorted(params.items())),
                    "sigma_default": round(sigma.default, 4),
                    "sigma_per_zone": {
                        "/".join(k): round(v, 4) for k, v in sorted(sigma.per_zone.items())
                    },
                }
            ),
        },
        "scores": [s.to_dict() for s in scores],
    }
    with dest.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    return dest


def write_sigma_csv(
    estimates: dict[tuple[str, str, str, str], SigmaEstimate],
    pooled: float,
    out_root: str | Path,
) -> Path:
    dest = Path(out_root) / SIGMA_CSV
    dest.parent.mkdir(parents=True, exist_ok=True)

    def fmt(v: float | None) -> str:
        return "" if v is None else f"{v:.2f}"

    with dest.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "from_cam", "from_zone", "to_cam", "to_zone", "n_samples",
            "sigma_frames", "median_e", "mad_e", "adopted",
        ])
        for key in sorted(estimates):
            e = estimates[key]
            writer.writerow([
                e.from_cam, e.from_zone, e.to_cam, e.to_zone, e.n_samples,
                fmt(e.sigma if e.adopted else pooled),
                fmt(e.median_e), fmt(e.mad_e), e.adopted,
            ])
    return dest


def print_sigma_report(
    estimates: dict[tuple[str, str, str, str], SigmaEstimate],
    pooled: float,
    min_samples: int,
) -> None:
    adopted = [e for e in estimates.values() if e.adopted]
    kept = [e for e in estimates.values() if not e.adopted]
    n_samples = sum(e.n_samples for e in estimates.values())

    print(f"\nσ の校正（正解・探索時間範囲内の {n_samples} 件から）")
    print(f"  共通 σ: {pooled:.0f} フレーム ({pooled/30:.2f} 秒)")
    print(f"  個別推定 {len(adopted)} / 共通を使う {len(kept)}（サンプル {min_samples} 件未満）")

    if adopted:
        print("\n  ゾーン対ごとの σ（サンプルの多い順）:")
        for e in sorted(adopted, key=lambda x: -x.n_samples):
            print(
                f"    {e.from_cam}/{e.from_zone} -> {e.to_cam}/{e.to_zone:<12} "
                f"{e.n_samples:>3} 件  σ={e.sigma:5.0f}  "
                f"中央 {e.median_e:+6.0f} / MAD {e.mad_e:5.0f}"
            )
    if kept:
        print(f"\n  共通 σ を使うゾーン対: "
              + ", ".join(f"{e.from_zone}->{e.to_zone}({e.n_samples})"
                          for e in sorted(kept, key=lambda x: -x.n_samples)[:10]))
