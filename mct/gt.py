"""Ground Truth の読み込み。

`scripts/import_gt.py` が `data/gt/` に置いた MOT 形式のアノテーションと
Global ID 対応表を読む。対応表は取り込みの時点で個人名の列を落としてあるので、
ここでは4列しか存在しない。

GT の track_id は CVAT の display_id で、対応表の `local_track_id` と直接
対応する。別途の変換表は要らない。
"""

from __future__ import annotations

import collections
import csv
import warnings
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "GTBox",
    "GroundTruth",
    "GTTransition",
    "ALL_CAMERAS",
    "load_ground_truth",
    "gt_transitions",
]

ALL_CAMERAS = ("camA", "camB", "camC", "camD", "camE", "camF", "camG", "camH")
GT_FILENAME = "gt.txt"
GLOBAL_ID_FILENAME = "globalID.csv"


def _camera_name(camera_id: str) -> str:
    """対応表の "A" を追跡側の "camA" に合わせる。"""
    return camera_id if camera_id.startswith("cam") else f"cam{camera_id}"


def _date_from_sequence(sequence_id: str) -> str:
    """対応表の "720" を日付 "0720" に合わせる（4桁に0詰め）。"""
    return str(sequence_id).zfill(4)


@dataclass(frozen=True)
class GTBox:
    frame_idx: int                 # 1始まり。追跡結果と同じ基準
    bbox: tuple[float, float, float, float]   # [x, y, w, h]
    not_ignored: bool
    visibility: float


@dataclass
class GroundTruth:
    date: str
    boxes: dict[str, dict[int, list[GTBox]]] = field(default_factory=dict)
    global_ids: dict[tuple[str, int], str] = field(default_factory=dict)
    ignored: set[tuple[str, int]] = field(default_factory=set)

    def tracks(self, camera: str) -> dict[int, list[GTBox]]:
        return self.boxes.get(camera, {})

    def evaluable(self, camera: str) -> list[int]:
        """評価対象の GT トラック（ignore を除いたもの）。"""
        return [t for t in self.tracks(camera) if (camera, t) not in self.ignored]

    def frames_with_gt(self, camera: str) -> set[int]:
        """GT がトラックを1つでも持つフレーム。

        GT が無いフレームは、そこに何が写っていたか分からないので
        評価対象にしない。
        """
        return {b.frame_idx for boxes in self.tracks(camera).values() for b in boxes}


@dataclass(frozen=True)
class GTTransition:
    """同一 Global ID の隣り合うトラック対。カメラ間の移動1回にあたる。"""

    global_id: str
    from_cam: str
    from_track: int
    to_cam: str
    to_track: int

    @property
    def is_self_loop(self) -> bool:
        return self.from_cam == self.to_cam


def _read_gt_file(path: Path) -> dict[int, list[GTBox]]:
    """MOT 形式の gt.txt を track_id ごとにまとめる。"""
    tracks: dict[int, list[GTBox]] = collections.defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cells = line.split(",")
            tracks[int(cells[1])].append(
                GTBox(
                    frame_idx=int(cells[0]),
                    bbox=(float(cells[2]), float(cells[3]), float(cells[4]), float(cells[5])),
                    not_ignored=int(cells[6]) != 0,
                    visibility=float(cells[8]) if len(cells) > 8 else 1.0,
                )
            )
    return {t: sorted(b, key=lambda x: x.frame_idx) for t, b in tracks.items()}


def _read_global_ids(path: Path, date: str) -> dict[tuple[str, int], str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {
            (_camera_name(r["camera_id"]), int(r["local_track_id"])): r["global_id"]
            for r in csv.DictReader(f)
            if _date_from_sequence(r["sequence_id"]) == date
        }


def load_ground_truth(
    gt_root: str | Path,
    date: str,
    cameras: tuple[str, ...] = ALL_CAMERAS,
    *,
    ignore_without_global_id: bool = True,
) -> GroundTruth:
    """1日分の GT を読む。

    評価から外すトラック（`ignored`）に入れるのは2種類。

    - `not_ignored=0` のトラック。シナリオ外の人物なので FP に数えない
    - Global ID を持たないトラック。全32本で56件あり、95%が30フレーム未満・
      長さの中央値が2フレームというアノテーションの断片で、Global ID が
      無い以上そもそも正誤を判定できない
    """
    root = Path(gt_root)
    gid_path = root / GLOBAL_ID_FILENAME
    if not gid_path.is_file():
        raise FileNotFoundError(
            f"{gid_path} がありません。先に scripts/import_gt.py を実行してください。"
        )

    gt = GroundTruth(date=date, global_ids=_read_global_ids(gid_path, date))

    for camera in cameras:
        path = root / date / camera / GT_FILENAME
        if not path.is_file():
            warnings.warn(f"{path} がありません（{date}/{camera} は評価対象0件）", stacklevel=2)
            gt.boxes[camera] = {}
            continue
        gt.boxes[camera] = _read_gt_file(path)

    for (camera, track_id), _ in list(gt.global_ids.items()):
        if track_id not in gt.boxes.get(camera, {}):
            warnings.warn(
                f"{date}/{camera}: 対応表の local_track_id {track_id} が gt.txt にありません",
                stacklevel=2,
            )
            del gt.global_ids[(camera, track_id)]

    for camera, tracks in gt.boxes.items():
        for track_id, boxes in tracks.items():
            if not any(b.not_ignored for b in boxes):
                gt.ignored.add((camera, track_id))
            elif ignore_without_global_id and (camera, track_id) not in gt.global_ids:
                gt.ignored.add((camera, track_id))

    return gt


def gt_transitions(gt: GroundTruth) -> list[GTTransition]:
    """正解のカメラ間遷移。

    同一 Global ID のトラックを出現フレーム順に並べ、隣り合う対を1件とする。
    同じカメラ内で2回現れる場合（居室に入って戻ってくる自己ループ）も含む。
    """
    by_gid: dict[str, list[tuple[int, str, int]]] = collections.defaultdict(list)
    for (camera, track_id), global_id in gt.global_ids.items():
        boxes = gt.boxes.get(camera, {}).get(track_id)
        if not boxes:
            continue
        by_gid[global_id].append((boxes[0].frame_idx, camera, track_id))

    transitions = []
    for global_id, entries in by_gid.items():
        entries.sort()
        for (_, cam_a, track_a), (_, cam_b, track_b) in zip(entries, entries[1:]):
            transitions.append(
                GTTransition(
                    global_id=global_id,
                    from_cam=cam_a, from_track=track_a,
                    to_cam=cam_b, to_track=track_b,
                )
            )
    return transitions
