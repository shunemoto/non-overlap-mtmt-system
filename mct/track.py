"""1カメラ分の追跡処理（2パス構成）。

1パス目で pose + BoT-SORT により全フレームの軌跡を取り、トラックごとに
保存するフレームを選抜する。2パス目でその選抜フレームにだけ seg を適用して
crop とマスクを保存する。全フレームに seg をかけると時間にもディスクにも
見合わないため、この2段構えにしている。
"""

from __future__ import annotations

import os

# MPS に未実装の演算を CPU に落とす。torch の import 後では効かないので、
# ultralytics（内部で torch を読む）より前に設定する必要がある。
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import statistics  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from mct.io_utils import crop_path, mask_path, output_dir, result_path, save_tracking_result  # noqa: E402
from mct.schema import FramePoint, TrackingResult, TrackMeta  # noqa: E402
from mct.select import pose_score, select_frames  # noqa: E402

__all__ = ["run_track"]

PERSON_CLASS = 0


@dataclass
class _RawTrack:
    """1パス目でトラックごとに貯める中間表現。

    `FramePoint` は saved / keypoints / crop_box の不変条件を持つので、
    2パス目が終わって内容が確定してから一度だけ構築する。
    """

    frame_idx: list[int] = field(default_factory=list)
    bbox: list[list[float]] = field(default_factory=list)
    conf: list[float] = field(default_factory=list)
    keypoints: list[np.ndarray] = field(default_factory=list)
    score: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.frame_idx)


def _video_meta(video: str | Path) -> tuple[int, int, float, int]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return width, height, fps, n_frames


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """xyxy の1矩形と複数矩形の IoU。`boxes` が空なら空配列を返す。"""
    if len(boxes) == 0:
        return np.empty(0, dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)

    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area + areas - inter
    return np.where(union > 0, inter / union, 0.0)


def _padded_crop_box(
    bbox: list[float], ratio: float, width: int, height: int
) -> tuple[int, int, int, int]:
    """bbox を四方に ratio だけ広げ、画像内にクリップした矩形を返す。

    SIFT の記述領域が肩付近で bbox からはみ出し得るため余白を取る。
    返すのは実際に切り出せる整数座標の (x, y, w, h)。
    """
    x, y, w, h = bbox
    px, py = w * ratio, h * ratio
    x0 = int(round(max(0.0, x - px)))
    y0 = int(round(max(0.0, y - py)))
    x1 = int(round(min(float(width), x + w + px)))
    y1 = int(round(min(float(height), y + h + py)))
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def _pass1(
    video: str | Path, params: dict[str, Any], max_frames: int | None
) -> tuple[dict[int, _RawTrack], int]:
    """pose + BoT-SORT で全フレームを追跡し、トラックごとの生データを集める。"""
    model = YOLO(params["model_pose"])
    stride = int(params["stride"])

    stream = model.track(
        source=str(video),
        stream=True,  # フレームを逐次処理し、全フレームをメモリに載せない
        persist=True,
        tracker=params["tracker"],
        conf=float(params["conf_threshold"]),
        imgsz=int(params["imgsz"]),
        device=params["device"],
        classes=[PERSON_CLASS],
        vid_stride=stride,
        verbose=False,
    )

    tracks: dict[int, _RawTrack] = {}
    n_processed = 0
    for i, r in enumerate(stream):
        if max_frames is not None and n_processed >= max_frames:
            break
        # vid_stride で間引いても frame_idx は元動画の通し番号（1始まり）を保つ
        frame_idx = i * stride + 1
        n_processed += 1

        boxes = r.boxes
        if boxes is None or boxes.id is None:
            continue

        ids = boxes.id.int().cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        kps_all = r.keypoints.data.cpu().numpy().astype(np.float32)

        for tid, (x1, y1, x2, y2), conf, kps in zip(ids, xyxy, confs, kps_all):
            w, h = float(x2 - x1), float(y2 - y1)
            t = tracks.setdefault(int(tid), _RawTrack())
            t.frame_idx.append(frame_idx)
            t.bbox.append([float(x1), float(y1), w, h])
            t.conf.append(float(conf))
            t.keypoints.append(kps)
            t.score.append(pose_score(kps))

    return tracks, n_processed


def _pass2(
    video: str | Path,
    tracks: dict[int, _RawTrack],
    selected: dict[int, list[int]],
    params: dict[str, Any],
    out_dir: Path,
    width: int,
    height: int,
) -> tuple[dict[tuple[int, int], list[float]], int, list[tuple[int, int]]]:
    """選抜フレームに seg をかけ、crop とマスクを保存する。

    戻り値は (実際に切り出した crop 矩形, 保存した crop 枚数,
    マスクを採れなかった (track_id, frame_idx) の列)。crop 矩形は
    `(track_id, トラック内の位置)` をキーとし、`FramePoint.crop_box` になる。
    """
    model = YOLO(params["model_seg"])
    ratio = float(params["crop_padding_ratio"])
    iou_threshold = float(params["mask_iou_threshold"])
    jpeg_quality = int(params["jpeg_quality"])

    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)

    # フレーム番号 -> そのフレームで保存すべき (track_id, トラック内の位置)
    todo: dict[int, list[tuple[int, int]]] = {}
    for tid, positions in selected.items():
        for pos in positions:
            todo.setdefault(tracks[tid].frame_idx[pos], []).append((tid, pos))

    crop_boxes: dict[tuple[int, int], list[float]] = {}
    n_crops = 0
    no_mask: list[tuple[int, int]] = []

    # ランダムシークは H.264 でフレームずれを起こしやすいので、
    # 先頭から順次読み出す1パスにして対象フレームだけ推論する。
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")

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

            # seg 推論はフレームにつき1回だけ。retina_masks で元解像度のマスクを得る
            # （既定ではモデル入力解像度のままで、crop 座標系に合わせられない）。
            r = model(
                frame,
                imgsz=int(params["imgsz"]),
                device=params["device"],
                classes=[PERSON_CLASS],
                retina_masks=True,
                verbose=False,
            )[0]
            seg_boxes = (
                r.boxes.xyxy.cpu().numpy()
                if r.boxes is not None and len(r.boxes)
                else np.empty((0, 4), dtype=np.float32)
            )
            seg_masks = r.masks.data.cpu().numpy() if r.masks is not None else None

            for tid, pos in targets:
                t = tracks[tid]
                x, y, w, h = t.bbox[pos]
                cx, cy, cw, ch = _padded_crop_box([x, y, w, h], ratio, width, height)
                crop_boxes[(tid, pos)] = [float(cx), float(cy), float(cw), float(ch)]

                cv2.imwrite(
                    str(crop_path(out_dir, tid, frame_idx)),
                    frame[cy : cy + ch, cx : cx + cw],
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                )
                n_crops += 1

                if seg_masks is None or len(seg_boxes) == 0:
                    no_mask.append((tid, frame_idx))
                    continue

                ious = _iou(np.array([x, y, x + w, y + h], dtype=np.float32), seg_boxes)
                best = int(np.argmax(ious))
                if ious[best] < iou_threshold:
                    no_mask.append((tid, frame_idx))
                    continue

                binary = (seg_masks[best] > 0.5).astype(np.uint8) * 255
                cv2.imwrite(
                    str(mask_path(out_dir, tid, frame_idx)),
                    binary[cy : cy + ch, cx : cx + cw],
                )
    finally:
        cap.release()

    return crop_boxes, n_crops, no_mask


def _build_result(
    tracks: dict[int, _RawTrack],
    selected: dict[int, list[int]],
    crop_boxes: dict[tuple[int, int], list[float]],
    meta: TrackMeta,
) -> TrackingResult:
    """中間表現から最終的な TrackingResult を組み立てる。"""
    result_tracks: dict[str, list[FramePoint]] = {}
    for tid, t in sorted(tracks.items()):
        chosen = set(selected.get(tid, []))
        points: list[FramePoint] = []
        for pos in range(len(t)):
            x, y, w, h = t.bbox[pos]
            saved = pos in chosen and (tid, pos) in crop_boxes
            points.append(
                FramePoint(
                    frame_idx=t.frame_idx[pos],
                    bbox=[x, y, w, h],
                    conf=t.conf[pos],
                    foot_point=[x + w / 2.0, y + h],
                    saved=saved,
                    keypoints=(
                        [[float(v) for v in kp] for kp in t.keypoints[pos]]
                        if saved
                        else None
                    ),
                    crop_box=crop_boxes[(tid, pos)] if saved else None,
                )
            )
        result_tracks[str(tid)] = points
    return TrackingResult(meta=meta, tracks=result_tracks)


def _print_stats(
    result: TrackingResult,
    n_crops: int,
    no_mask: list[tuple[int, int]],
    t_pass1: float,
    t_pass2: float,
) -> None:
    lengths = sorted(len(p) for p in result.tracks.values())
    print()
    print("=" * 56)
    print(f"総トラック数          : {len(result.tracks)}")
    if lengths:
        print(
            f"トラック長            : 最小 {lengths[0]} / "
            f"中央値 {int(statistics.median(lengths))} / 最大 {lengths[-1]}"
        )
    print(f"crop 保存枚数         : {n_crops}")
    print(f"mask 保存枚数         : {n_crops - len(no_mask)}")
    print(f"マスク取得失敗        : {len(no_mask)} フレーム")
    print("-" * 56)
    print(f"1パス目 (pose+track)  : {t_pass1/60:6.1f} 分")
    print(f"2パス目 (seg+save)    : {t_pass2/60:6.1f} 分")
    print(f"合計                  : {(t_pass1+t_pass2)/60:6.1f} 分")
    print("=" * 56)


def run_track(
    video: str | Path,
    date: str,
    camera: str,
    out_root: str | Path,
    params: dict[str, Any],
    max_frames: int | None = None,
) -> Path:
    """1動画を追跡し、tracks.json と crops/ masks/ を書き出してパスを返す。"""
    video = Path(video)
    width, height, fps, n_frames = _video_meta(video)
    out_dir = output_dir(out_root, date, camera)

    print(f"video   : {video}")
    print(f"size    : {width}x{height} @ {fps:.3f} fps, {n_frames} frames")
    print(f"device  : {params['device']}  pose: {params['model_pose']}  "
          f"seg: {params['model_seg']}  stride: {params['stride']}")
    print(f"output  : {out_dir}\n")

    print("[1/2] pose + BoT-SORT ...")
    t0 = time.perf_counter()
    tracks, n_processed = _pass1(video, params, max_frames)
    t_pass1 = time.perf_counter() - t0
    print(
        f"      {n_processed} フレーム処理, {len(tracks)} トラック検出 "
        f"({n_processed/t_pass1:.2f} fps)"
    )

    selected = {
        tid: select_frames(
            t.score,
            n_target=int(params["n_saved_per_track"]),
            n_bins=int(params["n_bins"]),
        )
        for tid, t in tracks.items()
    }
    n_selected_frames = len({tracks[tid].frame_idx[p] for tid, ps in selected.items() for p in ps})
    print(f"      選抜: {sum(len(v) for v in selected.values())} 件 "
          f"(ユニークフレーム {n_selected_frames})\n")

    print("[2/2] seg + crop/mask 保存 ...")
    t0 = time.perf_counter()
    crop_boxes, n_crops, no_mask = _pass2(
        video, tracks, selected, params, out_dir, width, height
    )
    t_pass2 = time.perf_counter() - t0

    for tid, frame_idx in no_mask:
        print(f"      [mask なし] track={tid} frame={frame_idx} (IoU < "
              f"{params['mask_iou_threshold']})")

    meta = TrackMeta(
        date=date,
        camera=camera,
        width=width,
        height=height,
        fps=fps,
        n_frames=n_frames,
        model_pose=str(params["model_pose"]),
        model_seg=str(params["model_seg"]),
        tracker=str(params["tracker"]),
        n_saved_per_track=int(params["n_saved_per_track"]),
        created_at=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    )
    result = _build_result(tracks, selected, crop_boxes, meta)
    path = save_tracking_result(result, result_path(out_dir))

    _print_stats(result, n_crops, no_mask, t_pass1, t_pass2)
    print(f"saved   : {path}")
    return path
