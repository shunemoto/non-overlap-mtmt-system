"""1日分の再同定を実行する。

候補・時空間尤度・外観類似度を読み、融合して割当を解き、Global ID を付ける。
手法ごとに変わるのは「どの候補集合を使うか」と「α をいくつにするか」だけで、
処理の流れは共通にしてある。

手法①（トポロジーなし）だけは候補集合そのものが違うので、ここで列挙し直して
外観を採点する。記述子は `appearance.json` にあるので画像は読まない。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mct.appearance_sim import AppScore, SCORES_FILENAME as APP_SCORES_FILENAME
from mct.candidates import Candidate, _enumerate_without_topology
from mct.fusion import METHODS, Method, app_value, fuse
from mct.predictions import Link, Predictions
from mct.spatiotemporal import KIND_SELF_LOOP
from mct.st_pipeline import SCORES_FILENAME as ST_SCORES_FILENAME, load_candidates

__all__ = ["DayResult", "run_date", "result_path"]

TrackKey = tuple[str, int]


@dataclass
class DayResult:
    """1日・1手法ぶんの実行結果と、その内訳。"""

    date: str
    method: int
    predictions: Predictions
    candidates: list[Candidate] = field(default_factory=list)
    scored: dict[tuple[TrackKey, TrackKey], float] = field(default_factory=dict)
    kinds: dict[tuple[TrackKey, TrackKey], str] = field(default_factory=dict)
    n_tracks: int = 0
    n_unscored: int = 0
    """候補ではあるが採点できなかった数（外観も時空間も無い）。"""
    sec_candidates: float = 0.0
    sec_similarity: float = 0.0
    sec_solve: float = 0.0

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)

    @property
    def n_links(self) -> int:
        return len(self.predictions.links)


def result_path(out_root: str | Path, date: str, method: int) -> Path:
    return Path(out_root) / date / f"global_ids_m{method}.json"


def _load_scores(path: Path, key: str) -> dict[tuple[TrackKey, TrackKey], Any]:
    """`st_scores.json` / `app_scores.json` を (tgt, src) で引ける形にする。"""
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    out = {}
    for row in payload.get("scores", []):
        pair = ((row["tgt"][0], int(row["tgt"][1])), (row["src"][0], int(row["src"][1])))
        out[pair] = row[key] if key else row
    return out


def _appearance_for(
    candidates: list[Candidate], out_root: str | Path, date: str, params: dict[str, Any]
) -> dict[tuple[TrackKey, TrackKey], AppScore]:
    """候補集合に対して外観をその場で採点する（手法①用）。

    記述子はトラック単位で作り置きしてあるので、ここは内積と
    ヒストグラム比較だけで済む。画像は読まない。
    """
    from mct.appearance import DESCRIPTOR_FILENAME, load_descriptors
    from mct.appearance_sim import score_pairs

    path = Path(out_root) / date / DESCRIPTOR_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} がありません。先に appearance を実行してください。"
        )
    descriptors = load_descriptors(path)
    return {
        (c.tgt, c.src): s
        for c, s in zip(candidates, score_pairs(candidates, descriptors, params))
        if s is not None
    }


def run_date(
    date: str,
    out_root: str | Path,
    method_number: int,
    params: dict[str, Any],
    app_params: dict[str, Any],
) -> DayResult:
    """1日・1手法を実行して結果を返す（書き込みは呼び出し側）。"""
    from mct.assignment import assign_global_ids, solve

    method: Method = METHODS[method_number]
    root = Path(out_root)

    t0 = time.perf_counter()
    summaries, candidates, _ = load_candidates(date, root)
    tracks: list[TrackKey] = sorted(s.key for s in summaries)

    if not method.topology:
        ordered = sorted(summaries, key=lambda s: (s.start_frame, s.camera, s.track_id))
        candidates = _enumerate_without_topology(ordered)
    t1 = time.perf_counter()

    st_scores = _load_scores(root / date / ST_SCORES_FILENAME, "likelihood") if method.st else {}
    st_kinds = _load_scores(root / date / ST_SCORES_FILENAME, "kind")
    if method.uses_appearance:
        app_scores = (
            _appearance_for(candidates, root, date, app_params)
            if not method.topology
            else {
                k: AppScore.from_dict(v)
                for k, v in _load_scores(root / date / APP_SCORES_FILENAME, "").items()
            }
        )
    else:
        app_scores = {}

    scored: dict[tuple[TrackKey, TrackKey], float] = {}
    kinds: dict[tuple[TrackKey, TrackKey], str] = {}
    detail: dict[tuple[TrackKey, TrackKey], tuple[float | None, float | None]] = {}
    n_unscored = 0

    for candidate in candidates:
        pair = (candidate.tgt, candidate.src)
        s_st = st_scores.get(pair)
        s_app = app_value(method, app_scores.get(pair))
        kind = st_kinds.get(pair, KIND_SELF_LOOP if candidate.is_self_loop else None)
        fused = fuse(
            s_st, s_app, kind=kind,
            is_self_loop=candidate.is_self_loop, method=method, params=params,
        )
        if fused is None:
            n_unscored += 1
            continue
        scored[pair] = fused[0]
        detail[pair] = (s_st, s_app)
        kinds[pair] = kind or ""
    t2 = time.perf_counter()

    chosen = solve(tracks, scored, float(params["no_match_score"]))
    t3 = time.perf_counter()

    zones = {(c.tgt, c.src): (c.tgt_zone, c.src_zone) for c in candidates}
    links = []
    for tgt, src in chosen:
        s_st, s_app = detail.get((tgt, src), (None, None))
        zone_from, zone_to = zones.get((tgt, src), ("", ""))
        links.append(
            Link(
                from_cam=tgt[0], from_track=str(tgt[1]),
                to_cam=src[0], to_track=str(src[1]),
                score=round(scored[(tgt, src)], 6),
                st=None if s_st is None else round(s_st, 6),
                app=None if s_app is None else round(s_app, 6),
                kind=kinds.get((tgt, src)) or None,
                zone_from=zone_from or None, zone_to=zone_to or None,
            )
        )

    predictions = Predictions(
        date=date, method=str(method_number),
        assignments=assign_global_ids(links, tracks), links=links,
        meta_extra=_meta(method_number, params, len(tracks), len(candidates)),
    )
    return DayResult(
        date=date, method=method_number, predictions=predictions,
        candidates=candidates, scored=scored, kinds=kinds,
        n_tracks=len(tracks), n_unscored=n_unscored,
        sec_candidates=t1 - t0, sec_similarity=t2 - t1, sec_solve=t3 - t2,
    )


def _meta(
    method: int, params: dict[str, Any], n_tracks: int, n_candidates: int
) -> dict[str, Any]:
    from mct.cache import cache_key

    subset = {
        "method": method,
        "alpha": float(params["alpha"]),
        "alpha_self": float(params["alpha_self"]),
        "no_match_score": float(params["no_match_score"]),
    }
    return {
        **subset,
        "n_tracks": n_tracks,
        "n_candidates": n_candidates,
        "config_hash": cache_key({"fusion": subset}),
    }
