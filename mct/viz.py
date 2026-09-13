"""追跡結果の可視化。

`tracks.json` と動画だけを読み、追跡モデルは一切読み込まない。描画の調整を
するたびに追跡をやり直さずに済むよう、追跡本体とは分離している。
"""

from __future__ import annotations

import colorsys
from pathlib import Path

import cv2
import numpy as np

from mct.io_utils import load_tracking_result, output_dir, result_path
from mct.schema import FramePoint, TrackingResult

__all__ = ["viz_video", "viz_trajectory"]

# COCO 17点の骨格。描画用の接続関係。
SKELETON: tuple[tuple[int, int], ...] = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),   # 脚と腰
    (5, 11), (6, 12), (5, 6),                           # 胴
    (5, 7), (7, 9), (6, 8), (8, 10),                    # 腕
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),     # 頭部
)

KEYPOINT_CONF_THRESHOLD = 0.3
TRAJECTORY_MAX_GAP_SEC = 1.0


def _track_color(index: int) -> tuple[int, int, int]:
    """トラックごとに見分けやすい BGR 色を返す。

    黄金角で色相を回すと、何本あっても隣接する ID 同士の色が離れる。
    彩度・明度は高めに固定して、淡い背景の上でも沈まないようにする。
    """
    hue = (index * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.95, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def _put_label(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.6,
) -> None:
    """背景の明暗によらず読めるよう、黒で縁取りしてから本体を描く。"""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def _resolve(out_root: str | Path, date: str, camera: str) -> tuple[Path, TrackingResult]:
    out_dir = output_dir(out_root, date, camera)
    path = result_path(out_dir)
    if not path.is_file():
        raise FileNotFoundError(f"{path} がありません。先に track を実行してください。")
    return out_dir, load_tracking_result(path)


def _viz_dir(out_dir: Path) -> Path:
    d = out_dir / "viz"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _draw_skeleton(
    frame: np.ndarray, keypoints: list[list[float]], color: tuple[int, int, int]
) -> None:
    kps = np.asarray(keypoints, dtype=np.float32)
    for a, b in SKELETON:
        if kps[a, 2] < KEYPOINT_CONF_THRESHOLD or kps[b, 2] < KEYPOINT_CONF_THRESHOLD:
            continue
        cv2.line(
            frame,
            (int(kps[a, 0]), int(kps[a, 1])),
            (int(kps[b, 0]), int(kps[b, 1])),
            color, 2, cv2.LINE_AA,
        )
    for x, y, c in kps:
        if c >= KEYPOINT_CONF_THRESHOLD:
            cv2.circle(frame, (int(x), int(y)), 3, (255, 255, 255), -1, cv2.LINE_AA)


def viz_video(video: str | Path, date: str, camera: str, out_root: str | Path) -> Path:
    """bbox・track_id・骨格点を描画した動画を書き出す。

    骨格点は選抜フレームにしか保存していないので、bbox と ID は全フレームに、
    骨格はデータがあるフレームにのみ描かれる。
    """
    out_dir, result = _resolve(out_root, date, camera)

    # フレーム番号 -> そのフレームに写っている (track_id, 色, 点) の列
    by_frame: dict[int, list[tuple[str, tuple[int, int, int], FramePoint]]] = {}
    for order, (tid, points) in enumerate(sorted(result.tracks.items(), key=lambda kv: int(kv[0]))):
        color = _track_color(order)
        for p in points:
            by_frame.setdefault(p.frame_idx, []).append((tid, color, p))

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")

    dest = _viz_dir(out_dir) / f"{camera}_track.mp4"
    # mp4v は同じ内容で H.264 の30倍近いサイズになるので avc1 を使う
    writer = cv2.VideoWriter(
        str(dest),
        cv2.VideoWriter_fourcc(*"avc1"),
        result.meta.fps,
        (result.meta.width, result.meta.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer: {dest}")

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            for tid, color, p in by_frame.get(frame_idx, []):
                x, y, w, h = p.bbox
                cv2.rectangle(
                    frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2, cv2.LINE_AA
                )
                _put_label(frame, f"ID {tid}", (int(x), max(18, int(y) - 8)), color)
                cv2.circle(
                    frame,
                    (int(p.foot_point[0]), int(p.foot_point[1])),
                    4, color, -1, cv2.LINE_AA,
                )
                if p.keypoints is not None:
                    _draw_skeleton(frame, p.keypoints, color)

            _put_label(frame, f"frame {frame_idx}", (12, 30), (255, 255, 255))
            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    print(f"wrote {dest} ({frame_idx} frames)")
    return dest


def viz_trajectory(video: str | Path, date: str, camera: str, out_root: str | Path) -> Path:
    """全トラックの足元点の軌跡を、動画の代表フレームに重ねた静止画を書き出す。

    観測区間と遷移領域をこの図から目視で決めるので、見やすさを優先する。
    背景は淡くして線を浮かせ、トラックごとに色相を離した色を割り当て、
    開始点に丸印と ID を置く。
    """
    out_dir, result = _resolve(out_root, date, camera)

    # 代表フレームは動画中央付近。人が写っていて背景として分かりやすい。
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, result.meta.n_frames // 2))
    ok, background = cap.read()
    cap.release()
    if not ok:
        background = np.full((result.meta.height, result.meta.width, 3), 255, np.uint8)

    # 背景を白方向に寄せて、その上の彩度の高い線を目立たせる
    white = np.full_like(background, 255)
    canvas = cv2.addWeighted(background, 0.45, white, 0.55, 0)

    max_gap = max(1, int(round(result.meta.fps * TRAJECTORY_MAX_GAP_SEC)))
    tracks = [
        (tid, points, _track_color(order))
        for order, (tid, points) in enumerate(
            sorted(result.tracks.items(), key=lambda kv: int(kv[0]))
        )
        if points
    ]

    # 線をすべて描いてから開始点と ID を描く。トラックごとに完結させると、
    # 後から描かれた線が先に描いた ID ラベルを覆ってしまう。
    for _, points, color in tracks:
        # オクルージョンで途切れた区間を長い直線で結んでしまわないよう、
        # フレーム間隔が開いたところで折れ線を切る。
        segment: list[tuple[int, int]] = []
        prev_frame: int | None = None
        for p in points:
            if prev_frame is not None and p.frame_idx - prev_frame > max_gap:
                if len(segment) > 1:
                    cv2.polylines(canvas, [np.array(segment, np.int32)], False, color, 3, cv2.LINE_AA)
                segment = []
            segment.append((int(p.foot_point[0]), int(p.foot_point[1])))
            prev_frame = p.frame_idx
        if len(segment) > 1:
            cv2.polylines(canvas, [np.array(segment, np.int32)], False, color, 3, cv2.LINE_AA)

    # 同じ場所から始まるトラックが多いと ID が重なって読めなくなるので、
    # すでに置いたラベルと衝突する間は下にずらしてから描く。
    placed: list[tuple[int, int, int, int]] = []
    for tid, points, color in tracks:
        start = (int(points[0].foot_point[0]), int(points[0].foot_point[1]))
        cv2.circle(canvas, start, 10, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, start, 7, color, -1, cv2.LINE_AA)

        (tw, th), _ = cv2.getTextSize(tid, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        lx, ly = start[0] + 14, start[1] - 10
        for _ in range(24):
            rect = (lx, ly - th, lx + tw, ly)
            if not any(
                rect[0] < q[2] and q[0] < rect[2] and rect[1] < q[3] and q[1] < rect[3]
                for q in placed
            ):
                break
            ly += th + 6
        placed.append((lx, ly - th, lx + tw, ly))
        cv2.line(canvas, start, (lx, ly - th // 2), color, 1, cv2.LINE_AA)
        _put_label(canvas, tid, (lx, ly), color, scale=0.8)

    n_drawn = len(tracks)

    _put_label(
        canvas,
        f"{result.meta.date} {result.meta.camera}  tracks={n_drawn}  (circle = start)",
        (14, 36), (0, 0, 0), scale=0.8,
    )

    dest = _viz_dir(out_dir) / f"{camera}_trajectory.png"
    cv2.imwrite(str(dest), canvas)
    print(f"wrote {dest} ({n_drawn} tracks)")
    return dest
