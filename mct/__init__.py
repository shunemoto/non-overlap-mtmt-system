"""非重複視野マルチカメラ人物追跡システム。"""

from mct.io_utils import (
    crop_path,
    load_tracking_result,
    mask_path,
    output_dir,
    result_path,
    save_tracking_result,
)
from mct.schema import FramePoint, TrackingResult, TrackMeta

__version__ = "0.1.0"

__all__ = [
    "FramePoint",
    "TrackMeta",
    "TrackingResult",
    "save_tracking_result",
    "load_tracking_result",
    "output_dir",
    "result_path",
    "crop_path",
    "mask_path",
    "__version__",
]
