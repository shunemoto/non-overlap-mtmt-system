"""Ground Truth の bbox から外観特徴の元データを作る。

検出・追跡の誤差を排除した条件で再同定だけを測るためのもの。出力は予測側と
同じ `TrackingResult` 形式なので、以降のフェーズは入力ディレクトリを
切り替えるだけで両方に対応できる。

モデルを使う関数（`extract_features`）と使わない関数（観測区間の適用・選抜・
座標変換）を分けてある。後者は torch を読まずに単体テストできる。
"""

from __future__ import annotations

import os

# MPS に未実装の演算を CPU に落とす。ultralytics より前に設定する必要がある。
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import statistics  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import numpy as np  # noqa: E402

from mct.config import CameraConfig  # noqa: E402
from mct.geometry import point_in_regions  # noqa: E402
from mct.gt import GroundTruth, GTBox  # noqa: E402
from mct.io_utils import crop_path, mask_path, output_dir  # noqa: E402
from mct.schema import FramePoint, TrackingResult, TrackMeta  # noqa: E402
from mct.select import select_frames  # noqa: E402

__all__ = [
    "GTStats",
    "foot_point",
    "apply_observation_region",
    "select_by_height",
    "padded_crop_box",
    "keypoints_to_image",
    "build_tracks",
    "extract_features",
]

PERSON_CLASS = 0


@dataclass
class GTStats:
    """1カメラ分の診断。"""

    date: str = ""
    camera: str = ""
    n_tracks_before: int = 0
    n_tracks_after: int = 0
    n_points_before: int = 0
    n_points_after: int = 0
    selected: list[int] = field(default_factory=list)
    n_crops: int = 0
    n_no_keypoints: int = 0
    n_no_mask: int = 0
    seconds: float = 0.0

    @property
    def n_selected(self) -> int:
        return sum(self.selected)


def foot_point(box: GTBox) -> tuple[float, float]:
    """bbox の下辺中央。予測側の `foot_point` と同じ定義。"""
    x, y, w, h = box.bbox
    return (x + w / 2.0, y + h)


def apply_observation_region(
    boxes: list[GTBox], cam_cfg: CameraConfig, min_track_len: int
) -> list[GTBox] | None:
    """観測区間の外側を落とす。短くなりすぎたら None。

    予測側の `mct.filtering.apply_observation_region` と同じ判断を、
    GT の箱に対して行う。
    """
    inside = [b for b in boxes if not point_in_regions(foot_point(b), cam_cfg.exclude_regions)]
    return inside if len(inside) >= min_track_len else None


def select_by_height(boxes: list[GTBox], n_target: int, n_bins: int) -> list[int]:
    """保存するフレームの位置を選ぶ。

    GT には骨格点が無いので、予測側の「骨格信頼度の合計」は使えない。
    代わりに **bbox の高さ**を評価値にする。人物が大きく写るフレームほど
    外観特徴が安定するため。`select_frames` はスコアの意味に依存しないので、
    9等分して各区画の上位を取る仕組みをそのまま使える。
    """
    return select_frames([b.bbox[3] for b in boxes], n_target=n_target, n_bins=n_bins)


def padded_crop_box(
    bbox: tuple[float, float, float, float], ratio: float, width: int, height: int
) -> tuple[int, int, int, int]:
    """bbox を四方に広げ、画像内にクリップした切り出し矩形 (x, y, w, h)。"""
    x, y, w, h = bbox
    px, py = w * ratio, h * ratio
    x0 = int(round(max(0.0, x - px)))
    y0 = int(round(max(0.0, y - py)))
    x1 = int(round(min(float(width), x + w + px)))
    y1 = int(round(min(float(height), y + h + py)))
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def keypoints_to_image(
    kps: np.ndarray, crop_origin: tuple[int, int]
) -> list[list[float]]:
    """crop 座標系の骨格点を元画像座標系に戻す。

    予測側は元画像で推論しているので、そちらと同じ座標系にそろえないと
    後段が使い分けを強いられる。信頼度0の点（未検出）は動かさない。
    """
    ox, oy = crop_origin
    out = []
    for x, y, c in np.asarray(kps, dtype=np.float64):
        out.append([float(x + ox), float(y + oy), float(c)] if c > 0 else [0.0, 0.0, 0.0])
    return out


def build_tracks(
    gt: GroundTruth, camera: str, cam_cfg: CameraConfig, params: dict[str, Any]
) -> tuple[dict[int, list[GTBox]], dict[int, list[int]], GTStats]:
    """観測区間を適用し、保存フレームを選ぶ。モデルは使わない。

    戻り値は (残ったトラック, トラックごとの選抜位置, 診断)。
    """
    min_len = int(params["min_track_len"])
    n_target = int(params["n_frames_per_track"])
    n_bins = int(params["n_bins"])

    stats = GTStats(date=gt.date, camera=camera)
    kept: dict[int, list[GTBox]] = {}
    selected: dict[int, list[int]] = {}

    for track_id in sorted(gt.evaluable(camera)):
        boxes = gt.boxes[camera][track_id]
        stats.n_tracks_before += 1
        stats.n_points_before += len(boxes)

        inside = apply_observation_region(boxes, cam_cfg, min_len)
        if inside is None:
            continue

        kept[track_id] = inside
        selected[track_id] = select_by_height(inside, n_target, n_bins)
        stats.n_tracks_after += 1
        stats.n_points_after += len(inside)
        stats.selected.append(len(selected[track_id]))

    return kept, selected, stats


def _iou_xywh(a: tuple[float, float, float, float], boxes: np.ndarray) -> np.ndarray:
    """[x, y, w, h] と xyxy の並びの IoU。"""
    if len(boxes) == 0:
        return np.empty(0, dtype=np.float64)
    ax0, ay0, aw, ah = a
    ax1, ay1 = ax0 + aw, ay0 + ah
    x0 = np.maximum(ax0, boxes[:, 0])
    y0 = np.maximum(ay0, boxes[:, 1])
    x1 = np.minimum(ax1, boxes[:, 2])
    y1 = np.minimum(ay1, boxes[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = aw * ah + areas - inter
    return np.where(union > 0, inter / union, 0.0)


def extract_features(
    video: str | Path,
    gt: GroundTruth,
    camera: str,
    cam_cfg: CameraConfig,
    params: dict[str, Any],
    out_root: str | Path,
) -> tuple[TrackingResult | None, GTStats]:
    """GT の bbox から crop・マスク・骨格点を作り、TrackingResult を書き出す。

    選抜フレームだけを対象に、動画を先頭から順次読む1パスで処理する
    （`CAP_PROP_POS_FRAMES` のシークは H.264 でフレームずれを起こす）。
    推論は切り出した crop に対して行う。GT の bbox が分かっているので全画面に
    かける必要が無く、他人物が混ざる余地も減る。
    """
    import cv2
    from ultralytics import YOLO

    kept, selected, stats = build_tracks(gt, camera, cam_cfg, params)
    out_dir = output_dir(out_root, gt.date, camera)
    if not kept:
        return None, stats

    started = time.perf_counter()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # フレーム番号 -> そのフレームで切り出す (track_id, トラック内の位置)
    todo: dict[int, list[tuple[int, int]]] = {}
    for track_id, positions in selected.items():
        for pos in positions:
            todo.setdefault(kept[track_id][pos].frame_idx, []).append((track_id, pos))

    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    pose = YOLO(params["model_pose"])
    seg = YOLO(params["model_seg"])
    ratio = float(params["crop_padding_ratio"])
    imgsz = int(params["imgsz"])
    device = params["device"]
    jpeg_quality = int(params["jpeg_quality"])
    iou_threshold = float(params["mask_iou_threshold"])

    results: dict[tuple[int, int], tuple[list[float], list[list[float]] | None]] = {}
    frame_idx = 0
    remaining = len(todo)
    try:
        while remaining > 0:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            targets = todo.get(frame_idx)
            if not targets:
                continue
            remaining -= 1

            for track_id, pos in targets:
                box = kept[track_id][pos]
                cx, cy, cw, ch = padded_crop_box(box.bbox, ratio, width, height)
                crop = frame[cy : cy + ch, cx : cx + cw]
                cv2.imwrite(
                    str(crop_path(out_dir, track_id, frame_idx)), crop,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                )
                stats.n_crops += 1

                keypoints = _run_pose(pose, crop, (cx, cy), imgsz, device)
                if keypoints is None:
                    stats.n_no_keypoints += 1

                binary = _run_seg(
                    seg, crop, box.bbox, (cx, cy), imgsz, device, iou_threshold
                )
                if binary is None:
                    stats.n_no_mask += 1
                else:
                    cv2.imwrite(str(mask_path(out_dir, track_id, frame_idx)), binary)

                results[(track_id, pos)] = ([float(cx), float(cy), float(cw), float(ch)],
                                            keypoints)
    finally:
        cap.release()

    stats.seconds = time.perf_counter() - started
    meta = TrackMeta(
        date=gt.date, camera=camera, width=width, height=height, fps=fps,
        n_frames=n_frames,
        model_pose=str(params["model_pose"]), model_seg=str(params["model_seg"]),
        tracker="ground_truth",
        n_saved_per_track=int(params["n_frames_per_track"]),
        created_at=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    )
    return TrackingResult(meta=meta, tracks=_to_frame_points(kept, selected, results)), stats


def _run_pose(model, crop, origin, imgsz, device) -> list[list[float]] | None:
    """crop に pose をかけ、骨格点を元画像座標系で返す。取れなければ None。"""
    result = model(crop, imgsz=imgsz, device=device, classes=[PERSON_CLASS], verbose=False)[0]
    if result.keypoints is None or len(result.keypoints.data) == 0:
        return None

    # crop はほぼ対象人物なので、最も大きい検出を採る
    boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else None
    index = 0
    if boxes is not None and len(boxes) > 1:
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        index = int(np.argmax(areas))
    return keypoints_to_image(result.keypoints.data[index].cpu().numpy(), origin)


def _run_seg(model, crop, bbox, origin, imgsz, device, iou_threshold):
    """crop に seg をかけ、人物のマスク（0/255）を返す。取れなければ None。"""
    result = model(
        crop, imgsz=imgsz, device=device, classes=[PERSON_CLASS],
        retina_masks=True, verbose=False,
    )[0]
    if result.masks is None or len(result.masks.data) == 0:
        return None

    boxes = result.boxes.xyxy.cpu().numpy()
    # GT の bbox を crop 座標系に直して突き合わせる
    x, y, w, h = bbox
    local = (x - origin[0], y - origin[1], w, h)
    ious = _iou_xywh(local, boxes)
    best = int(np.argmax(ious))
    if ious[best] < iou_threshold:
        return None
    return (result.masks.data[best].cpu().numpy() > 0.5).astype(np.uint8) * 255


def _to_frame_points(
    kept: dict[int, list[GTBox]],
    selected: dict[int, list[int]],
    results: dict[tuple[int, int], tuple[list[float], list[list[float]] | None]],
) -> dict[str, list[FramePoint]]:
    """GT の箱を FramePoint に変換する。

    骨格点やマスクが取れなくても `saved=True` のままにする。色ヒストグラムは
    bbox の中だけで計算できるので、crop があれば外観特徴は作れる。
    """
    tracks: dict[str, list[FramePoint]] = {}
    for track_id, boxes in sorted(kept.items()):
        chosen = set(selected.get(track_id, []))
        points = []
        for pos, box in enumerate(boxes):
            x, y, w, h = box.bbox
            entry = results.get((track_id, pos))
            saved = pos in chosen and entry is not None
            points.append(
                FramePoint(
                    frame_idx=box.frame_idx,
                    bbox=[float(x), float(y), float(w), float(h)],
                    conf=float(box.visibility),
                    foot_point=list(foot_point(box)),
                    saved=saved,
                    keypoints=entry[1] if saved else None,
                    crop_box=entry[0] if saved else None,
                )
            )
        tracks[str(track_id)] = points
    return tracks


def print_stats(stats: GTStats) -> None:
    s = stats
    print(f"\n--- {s.date}/{s.camera} ---")
    print(f"  GT トラック: {s.n_tracks_before} -> {s.n_tracks_after}")
    removed = s.n_points_before - s.n_points_after
    if s.n_points_before:
        print(
            f"  観測区間で除外した点: {removed} / {s.n_points_before} "
            f"({100*removed/s.n_points_before:.1f}%)"
        )
    if s.selected:
        print(
            f"  選抜フレーム: 合計 {s.n_selected} / 最小 {min(s.selected)} "
            f"/ 中央 {int(statistics.median(s.selected))} / 最大 {max(s.selected)}"
        )
    if s.n_crops:
        print(
            f"  crop {s.n_crops} 枚 / 骨格点なし {s.n_no_keypoints} "
            f"({100*s.n_no_keypoints/s.n_crops:.1f}%) / マスクなし {s.n_no_mask} "
            f"({100*s.n_no_mask/s.n_crops:.1f}%)"
        )
    print(f"  所要 {s.seconds/60:.1f} 分")
