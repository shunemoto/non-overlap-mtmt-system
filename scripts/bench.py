"""追跡の処理速度を実測する。

本実装に入る前に、M1 の MPS で 1パス目（pose + BoT-SORT）と 2パス目（seg）が
どれだけ出るかを測り、32本分の総所要時間を見積もるために使う。

    python scripts/bench.py --video data/videos/0720/camA.mp4 --frames 300
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2  # noqa: E402
from ultralytics import YOLO  # noqa: E402

N_VIDEOS = 32
POSE_MODELS = ["yolo11x-pose.pt", "yolo11l-pose.pt"]
SEG_MODEL = "yolo11x-seg.pt"


def video_info(path: str) -> tuple[int, float]:
    cap = cv2.VideoCapture(path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return n_frames, fps


def bench_pose(model_name: str, video: str, n_frames: int, device: str, imgsz: int) -> float:
    """1パス目を n_frames 分回し、ウォームアップを除いた fps を返す。"""
    model = YOLO(model_name)
    stream = model.track(
        source=video,
        stream=True,
        persist=True,
        tracker="botsort.yaml",
        conf=0.25,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )

    n_warmup = 10
    t_start = 0.0
    n_done = 0
    for i, r in enumerate(stream, start=1):
        # 実際の処理と同じだけ結果に触る（GPU 同期を含めるため）
        if r.boxes is not None and r.boxes.id is not None:
            r.boxes.xyxy.cpu().numpy()
            r.keypoints.data.cpu().numpy()
        if i == n_warmup:
            t_start = time.perf_counter()
        elif i > n_warmup:
            n_done += 1
        if i >= n_frames:
            break
    dt = time.perf_counter() - t_start
    return n_done / dt


def bench_seg(video: str, n_frames: int, device: str, imgsz: int) -> float:
    """2パス目の 1フレームあたり秒数を返す。"""
    model = YOLO(SEG_MODEL)
    cap = cv2.VideoCapture(video)
    frames = []
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    model(frames[0], device=device, imgsz=imgsz, retina_masks=True, verbose=False)

    t = time.perf_counter()
    for frame in frames:
        r = model(frame, device=device, imgsz=imgsz, retina_masks=True, verbose=False)[0]
        if r.masks is not None:
            r.masks.data.cpu().numpy()
    return (time.perf_counter() - t) / len(frames)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/videos/0720/camA.mp4")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--seg-frames", type=int, default=30)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    total_frames, fps = video_info(args.video)
    print(f"video      : {args.video}")
    print(f"frames     : {total_frames} @ {fps:.3f} fps")
    print(f"device     : {args.device}  imgsz: {args.imgsz}\n")

    pose_fps: dict[str, float] = {}
    for name in POSE_MODELS:
        t0 = time.perf_counter()
        pose_fps[name] = bench_pose(name, args.video, args.frames, args.device, args.imgsz)
        print(f"[pass1] {name:18s} {pose_fps[name]:6.2f} fps   (計測 {time.perf_counter()-t0:.0f}s)")

    sec_per_frame = bench_seg(args.video, args.seg_frames, args.device, args.imgsz)
    print(f"[pass2] {SEG_MODEL:18s} {sec_per_frame*1000:6.0f} ms/frame  ({1/sec_per_frame:.2f} fps)\n")

    # 2パス目の対象は「選抜されたユニークフレーム」。仮に 2,000 枚とする。
    n_selected = 2000
    print(f"{'model':18s} {'pass1/本':>10s} {'pass2/本':>10s} {'計/本':>10s} {'32本合計':>12s}")
    for name, f in pose_fps.items():
        p1 = total_frames / f
        p2 = n_selected * sec_per_frame
        print(
            f"{name:18s} {p1/60:8.1f}分 {p2/60:8.1f}分 {(p1+p2)/60:8.1f}分 "
            f"{(p1+p2)*N_VIDEOS/3600:9.1f}時間"
        )
    print(f"\n(pass2 は選抜ユニークフレーム {n_selected} 枚と仮定)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
