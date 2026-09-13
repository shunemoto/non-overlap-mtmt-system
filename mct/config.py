"""パラメータ YAML の読み込み。

YAML に書かれていないキーは既定値で埋めるので、`configs/params.yaml` を
最小限にしても動く。CLI から個別に上書きすることも想定している。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

__all__ = ["DEFAULT_TRACKING", "load_params", "tracking_params"]

DEFAULT_TRACKING: dict[str, Any] = {
    "model_pose": "yolo11l-pose.pt",
    "model_seg": "yolo11x-seg.pt",
    "tracker": "botsort.yaml",
    "conf_threshold": 0.25,
    "n_saved_per_track": 90,
    "crop_padding_ratio": 0.10,
    "device": "mps",
    "imgsz": 640,
    "stride": 1,
    "mask_iou_threshold": 0.30,
    "n_bins": 9,
    "jpeg_quality": 95,
}


def load_params(path: str | Path | None) -> dict[str, Any]:
    """パラメータ YAML を読む。`path` が None か存在しなければ空とみなす。"""
    if path is None:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def tracking_params(
    path: str | Path | None, **overrides: Any
) -> dict[str, Any]:
    """既定値 → YAML の `tracking:` → 明示的な上書き、の順にマージする。

    `overrides` の値が None のキーは「指定なし」として無視するので、
    argparse の未指定オプションをそのまま渡せる。
    """
    params = dict(DEFAULT_TRACKING)
    params.update(load_params(path).get("tracking") or {})
    params.update({k: v for k, v in overrides.items() if v is not None})
    return params
