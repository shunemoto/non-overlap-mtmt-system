"""追跡結果の保存・読み込みと、出力ファイルのパス規約。

crop / mask のファイル名規約をここに一元化しておくことで、段階2以降が
パスをベタ書きせずに済む。標準ライブラリのみに依存する。
"""

from __future__ import annotations

import json
from pathlib import Path

from mct.schema import TrackingResult

__all__ = [
    "save_tracking_result",
    "load_tracking_result",
    "output_dir",
    "result_path",
    "crop_path",
    "mask_path",
]

RESULT_FILENAME = "tracks.json"


def save_tracking_result(result: TrackingResult, path: str | Path) -> Path:
    """追跡結果を JSON として保存し、書き込んだパスを返す。

    ファイルサイズを抑えるため、インデントも区切りのスペースも入れない。
    日本語などの非 ASCII 文字はエスケープせずそのまま書く。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, ensure_ascii=False, separators=(",", ":"))
    return p


def load_tracking_result(path: str | Path) -> TrackingResult:
    """`save_tracking_result` が書いた JSON を読み戻す。"""
    with Path(path).open("r", encoding="utf-8") as f:
        return TrackingResult.from_dict(json.load(f))


def output_dir(out_root: str | Path, date: str, camera: str) -> Path:
    """`{out_root}/{date}/{camera}/` を返す。無ければ作成する。"""
    d = Path(out_root) / date / camera
    d.mkdir(parents=True, exist_ok=True)
    return d


def result_path(out_dir: str | Path) -> Path:
    """出力ディレクトリ内の追跡結果 JSON のパス。"""
    return Path(out_dir) / RESULT_FILENAME


def crop_path(out_dir: str | Path, track_id: int | str, frame_idx: int) -> Path:
    """`{out_dir}/crops/t{track_id:03d}_f{frame_idx:06d}.jpg`

    パスを組み立てるだけで、ディレクトリは作らない。
    """
    return Path(out_dir) / "crops" / f"t{int(track_id):03d}_f{int(frame_idx):06d}.jpg"


def mask_path(out_dir: str | Path, track_id: int | str, frame_idx: int) -> Path:
    """`{out_dir}/masks/t{track_id:03d}_f{frame_idx:06d}.png`

    パスを組み立てるだけで、ディレクトリは作らない。
    """
    return Path(out_dir) / "masks" / f"t{int(track_id):03d}_f{int(frame_idx):06d}.png"
