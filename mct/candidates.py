"""対応候補の列挙。

カメラ間トポロジー（`transitions.yaml`）を使い、突き合わせる価値のある
トラック対だけを残す。類似度の算出と ID 統合はここでは行わない。

ここで落とした正解遷移は後段でどう工夫しても拾えないので、制約は
「明らかに有り得ない対」だけを弾く強さに留める。探索時間の上限を超えた対も
捨てずに `in_search_range=False` を付けて残すのはそのため。
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any

from mct.config import TransitionRoute
from mct.diagnostics import circular_diff
from mct.summarize import TrackSummary

__all__ = ["Candidate", "enumerate_candidates", "candidate_counts"]


@dataclass
class Candidate:
    """後から出現した `src` と、先に退出した `tgt` の対応仮説。"""

    src: tuple[str, int]        # (camera, track_id) 後に出現する側
    tgt: tuple[str, int]        # (camera, track_id) 先に退出する側
    src_zone: str
    tgt_zone: str
    route: TransitionRoute | None
    delta_frames: int           # src.start_frame - tgt.end_frame
    in_search_range: bool

    @property
    def is_self_loop(self) -> bool:
        return self.src[0] == self.tgt[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "src": {"cam": self.src[0], "track": self.src[1]},
            "tgt": {"cam": self.tgt[0], "track": self.tgt[1]},
            "src_zone": self.src_zone,
            "tgt_zone": self.tgt_zone,
            "route": None if self.route is None else {
                "from_cam": self.route.from_cam, "from_zone": self.route.from_zone,
                "to_cam": self.route.to_cam, "to_zone": self.route.to_zone,
                "distance_m": self.route.distance_m,
                "search_range_sec": list(self.route.search_range_sec),
                "c_dwell": self.route.c_dwell,
            },
            "delta_frames": self.delta_frames,
            "in_search_range": self.in_search_range,
        }

    def as_link(self) -> dict[str, Any]:
        """`candidate_recall` が読む形（時間順に from=tgt, to=src）。"""
        return {
            "from": {"cam": self.tgt[0], "track": self.tgt[1]},
            "to": {"cam": self.src[0], "track": self.src[1]},
        }


def _overlaps(a: TrackSummary, b: TrackSummary) -> bool:
    """出現区間が重なっているか。重なっていれば同一人物ではあり得ない。"""
    return a.start_frame <= b.end_frame and b.start_frame <= a.end_frame


def _direction_is_reversed(
    tgt: TrackSummary, src: TrackSummary, tolerance: float
) -> bool:
    """自己ループで、出ていった向きと戻ってきた向きが逆かどうか。

    同じカメラに再び現れる人物は、出た方向から戻ってくる。向きが同じままなら
    別人が同じ経路を通っただけとみなす。
    """
    if tgt.exit_dir_deg is None or src.entry_dir_deg is None:
        return False
    return circular_diff(tgt.exit_dir_deg, src.entry_dir_deg) >= 180.0 - tolerance


def enumerate_candidates(
    summaries: list[TrackSummary],
    routes: list[TransitionRoute],
    params: dict[str, Any],
    *,
    no_topology: bool = False,
) -> list[Candidate]:
    """候補を列挙する。

    `no_topology=True` はトポロジーを使わない比較手法用で、時間順序と
    区間非重複だけを課す。全カメラの全トラック対が対象になる。
    """
    fps = float(params.get("fps", 30))
    tolerance = float(params.get("self_loop_dir_tol", 60))

    by_key = {s.key: s for s in summaries}
    ordered = sorted(summaries, key=lambda s: (s.start_frame, s.camera, s.track_id))

    if no_topology:
        return _enumerate_without_topology(ordered)

    # 退出ゾーンごとに引けるようにしておく
    by_exit: dict[tuple[str, str], list[TrackSummary]] = collections.defaultdict(list)
    for s in summaries:
        for zone in s.exit_zones:
            by_exit[(s.camera, zone)].append(s)

    routes_by_dest: dict[tuple[str, str], list[TransitionRoute]] = collections.defaultdict(list)
    for r in routes:
        routes_by_dest[(r.to_cam, r.to_zone)].append(r)

    candidates: list[Candidate] = []
    seen: set[tuple[tuple[str, int], tuple[str, int], str, str]] = set()

    for src in ordered:
        for src_zone in src.entry_zones:
            # transitions.yaml の from が tgt 側、to が src 側。時間の向きと逆
            for route in routes_by_dest[(src.camera, src_zone)]:
                for tgt in by_exit[(route.from_cam, route.from_zone)]:
                    if tgt.key == src.key:
                        continue
                    if tgt.end_frame >= src.start_frame:
                        continue
                    if _overlaps(tgt, src):
                        continue

                    delta = src.start_frame - tgt.end_frame
                    lo, hi = route.search_range_sec
                    if delta < lo * fps:
                        continue
                    if route.is_self_loop and not _direction_is_reversed(tgt, src, tolerance):
                        continue

                    key = (src.key, tgt.key, src_zone, route.from_zone)
                    if key in seen:
                        continue
                    seen.add(key)

                    candidates.append(
                        Candidate(
                            src=src.key, tgt=tgt.key,
                            src_zone=src_zone, tgt_zone=route.from_zone,
                            route=route, delta_frames=delta,
                            # 上限を超えた対も捨てずに残し、印だけ付ける
                            in_search_range=(hi is None or delta <= hi * fps),
                        )
                    )
    return candidates


def _enumerate_without_topology(ordered: list[TrackSummary]) -> list[Candidate]:
    """時間順序と区間非重複だけで列挙する（比較手法①）。"""
    candidates = []
    for i, src in enumerate(ordered):
        for tgt in ordered[:i]:
            if tgt.end_frame >= src.start_frame or _overlaps(tgt, src):
                continue
            candidates.append(
                Candidate(
                    src=src.key, tgt=tgt.key, src_zone="", tgt_zone="",
                    route=None, delta_frames=src.start_frame - tgt.end_frame,
                    in_search_range=True,
                )
            )
    return candidates


def candidate_counts(candidates: list[Candidate]) -> dict[str, Any]:
    """診断用の内訳。"""
    by_pair = collections.Counter((c.tgt[0], c.src[0]) for c in candidates)
    return {
        "total": len(candidates),
        "in_range": sum(1 for c in candidates if c.in_search_range),
        "out_of_range": sum(1 for c in candidates if not c.in_search_range),
        "self_loop": sum(1 for c in candidates if c.is_self_loop),
        "by_camera_pair": by_pair,
        "n_src": len({c.src for c in candidates}),
        "n_tgt": len({c.tgt for c in candidates}),
    }
