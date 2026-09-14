"""追跡結果の可視化。

`tracks.json` と動画だけを読み、追跡モデルは一切読み込まない。描画の調整を
するたびに追跡をやり直さずに済むよう、追跡本体とは分離している。

観測区間と遷移領域はこれらの図を見て決めるので、目的ごとに図を分けている。

- `viz_video`      : 追跡が正しく追えているかをフレーム単位で確認する
- `viz_trajectory` : 動線の全体像をつかむ
- `viz_density`    : 検出が安定している範囲と、端で途切れる位置を見る
- `viz_endpoints`  : 遷移領域の矩形と、進入・退出方向の角度範囲を決める
- `viz_region`     : 定義した観測区間・遷移領域が意図どおりか確かめる

`viz_region` だけは `tracks.json` を読まず、動画と `cameras.yaml` だけで動く。
領域の座標を書いては確認する作業を、追跡の実行中にも並行してできるようにする。
"""

from __future__ import annotations

import colorsys
import time
from pathlib import Path

import cv2
import numpy as np

from mct.config import CameraConfig, Region, TransitionZone, load_cameras
from mct.geometry import region_center
from mct.io_utils import load_tracking_result, output_dir, result_path
from mct.schema import FramePoint, TrackingResult

__all__ = ["viz_video", "viz_trajectory", "viz_density", "viz_endpoints", "viz_region"]

# COCO 17点の骨格。描画用の接続関係。
SKELETON: tuple[tuple[int, int], ...] = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),   # 脚と腰
    (5, 11), (6, 12), (5, 6),                           # 胴
    (5, 7), (7, 9), (6, 8), (8, 10),                    # 腕
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),     # 頭部
)

KEYPOINT_CONF_THRESHOLD = 0.3
TRAJECTORY_MAX_GAP_SEC = 1.0

DEFAULT_JUMP_THRESHOLD = 150.0
JUMP_COLOR = (0, 0, 255)

DENSITY_BIN_PX = 20
DENSITY_SMOOTH_SIGMA = 1.2

ENDPOINT_SHORT_TRACK_FRAMES = 30
ENDPOINT_ARROW_SEC = 2.0
ENDPOINT_ARROW_PX = 55
ENDPOINT_GRID_PX = 200
START_COLOR = (255, 80, 0)    # BGR: 青
END_COLOR = (0, 40, 255)      # BGR: 赤
SHORT_TRACK_COLOR = (150, 150, 150)

EXCLUDE_COLOR = (60, 60, 235)     # BGR: 赤
EXCLUDE_ALPHA = 0.35
ZONE_COLOR = (80, 200, 80)        # BGR: 緑
ZONE_ALPHA = 0.25
EXIT_COLOR = (0, 150, 255)        # BGR: 橙
ENTRY_COLOR = (255, 220, 90)      # BGR: 水色
EXIT_RADIUS = 80
ENTRY_RADIUS = 55                 # exit と重ねると読めないので内側に描く
SECTOR_ALPHA = 0.45


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


def _place_label(
    placed: list[tuple[int, int, int, int]],
    anchor: tuple[int, int],
    text: str,
    scale: float,
    offset: tuple[int, int] = (14, -10),
) -> tuple[int, int]:
    """すでに置いたラベルと重ならない描画位置を探して返す。

    同じ場所から始まるトラックが多いと ID が重なって読めなくなるので、
    衝突する間は下にずらす。`placed` は呼び出し側が持ち回る。
    """
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    lx, ly = anchor[0] + offset[0], anchor[1] + offset[1]
    for _ in range(24):
        rect = (lx, ly - th, lx + tw, ly)
        if not any(
            rect[0] < q[2] and q[0] < rect[2] and rect[1] < q[3] and q[1] < rect[3]
            for q in placed
        ):
            break
        ly += th + 6
    placed.append((lx, ly - th, lx + tw, ly))
    return lx, ly


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


def _background_frame(video: str | Path, result: TrackingResult) -> np.ndarray:
    """静止画の背景に使う代表フレーム（動画の中央付近）を読む。"""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, result.meta.n_frames // 2))
    ok, background = cap.read()
    cap.release()
    if not ok:
        return np.full((result.meta.height, result.meta.width, 3), 255, np.uint8)
    return background


def _faded(background: np.ndarray) -> np.ndarray:
    """背景を白方向に寄せ、その上に描く彩度の高い線を目立たせる。"""
    return cv2.addWeighted(background, 0.45, np.full_like(background, 255), 0.55, 0)


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


def _ordered_tracks(
    result: TrackingResult,
) -> list[tuple[str, list[FramePoint], tuple[int, int, int]]]:
    return [
        (tid, points, _track_color(order))
        for order, (tid, points) in enumerate(
            sorted(result.tracks.items(), key=lambda kv: int(kv[0]))
        )
        if points
    ]


def _jump_frames(points: list[FramePoint], threshold: float) -> set[int]:
    """足元点が急に跳んだフレーム番号を返す。

    トラックには最大20フレーム程度の欠落があり、間が空けば足元点が離れるのは
    当然なので、真に連続する2フレームの間だけを見る。
    """
    jumps: set[int] = set()
    for a, b in zip(points, points[1:]):
        if b.frame_idx - a.frame_idx != 1:
            continue
        dx = b.foot_point[0] - a.foot_point[0]
        dy = b.foot_point[1] - a.foot_point[1]
        if (dx * dx + dy * dy) ** 0.5 >= threshold:
            jumps.add(b.frame_idx)
    return jumps


def viz_video(
    video: str | Path,
    date: str,
    camera: str,
    out_root: str | Path,
    jump_threshold: float = DEFAULT_JUMP_THRESHOLD,
) -> Path:
    """bbox・track_id・骨格点・足元点を描画した動画を書き出す。

    人物が1人も検出されていないフレームは出力しない。撮影時間の大半が
    無人だと、そこを流して見るだけで時間を取られるため。フレームが飛ぶので、
    元動画のフレーム番号を画面に焼き込む。

    足元点が `jump_threshold` 以上跳んだフレームは bbox を赤枠にして "JUMP" と
    表示する。遮蔽で bbox の下辺が切り上がった箇所を見つけるための目印。
    """
    out_dir, result = _resolve(out_root, date, camera)

    # フレーム番号 -> そのフレームに写っている (track_id, 色, 点, JUMPか) の列
    by_frame: dict[int, list[tuple[str, tuple[int, int, int], FramePoint, bool]]] = {}
    n_jumps = 0
    for tid, points, color in _ordered_tracks(result):
        jumps = _jump_frames(points, jump_threshold)
        n_jumps += len(jumps)
        for p in points:
            by_frame.setdefault(p.frame_idx, []).append(
                (tid, color, p, p.frame_idx in jumps)
            )

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

    t0 = time.perf_counter()
    frame_idx = 0
    n_written = 0
    last_wanted = max(by_frame) if by_frame else 0
    try:
        while frame_idx < last_wanted:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            targets = by_frame.get(frame_idx)
            if not targets:
                continue  # 無人フレームは出力しない

            for tid, color, p, is_jump in targets:
                box_color = JUMP_COLOR if is_jump else color
                x, y, w, h = p.bbox
                cv2.rectangle(
                    frame, (int(x), int(y)), (int(x + w), int(y + h)),
                    box_color, 3 if is_jump else 2, cv2.LINE_AA,
                )
                _put_label(frame, f"ID {tid}", (int(x), max(18, int(y) - 8)), box_color)
                if is_jump:
                    _put_label(
                        frame, "JUMP", (int(x), min(result.meta.height - 6, int(y + h) + 24)),
                        JUMP_COLOR,
                    )
                cv2.circle(
                    frame,
                    (int(p.foot_point[0]), int(p.foot_point[1])),
                    4, box_color, -1, cv2.LINE_AA,
                )
                if p.keypoints is not None:
                    _draw_skeleton(frame, p.keypoints, color)

            # フレームが飛ぶので、元動画での番号を必ず焼き込む
            _put_label(
                frame,
                f"frame {frame_idx} / {result.meta.n_frames}",
                (12, 30), (255, 255, 255), scale=0.8,
            )
            writer.write(frame)
            n_written += 1
    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - t0
    total = result.meta.n_frames
    skipped = total - n_written
    print(f"wrote {dest}")
    print(f"  出力フレーム : {n_written} / {total} ({100*n_written/total:.1f}%)")
    print(f"  除外         : {skipped} フレーム ({100*skipped/total:.1f}% 削減)")
    print(f"  JUMP 箇所    : {n_jumps} (閾値 {jump_threshold:.0f}px)")
    print(f"  所要時間     : {elapsed:.1f} 秒")
    return dest


def viz_trajectory(video: str | Path, date: str, camera: str, out_root: str | Path) -> Path:
    """全トラックの足元点の軌跡を、動画の代表フレームに重ねた静止画を書き出す。

    観測区間と遷移領域をこの図から目視で決めるので、見やすさを優先する。
    背景は淡くして線を浮かせ、トラックごとに色相を離した色を割り当て、
    開始点に丸印と ID を置く。
    """
    out_dir, result = _resolve(out_root, date, camera)
    canvas = _faded(_background_frame(video, result))

    max_gap = max(1, int(round(result.meta.fps * TRAJECTORY_MAX_GAP_SEC)))
    tracks = _ordered_tracks(result)

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

        lx, ly = _place_label(placed, start, tid, 0.8)
        (_, th), _ = cv2.getTextSize(tid, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
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


def viz_density(
    video: str | Path,
    date: str,
    camera: str,
    out_root: str | Path,
    bin_px: int = DENSITY_BIN_PX,
) -> Path:
    """全トラックの足元点の density を代表フレームに重ねた静止画を書き出す。

    軌跡図は線が交錯すると読めなくなるが、密度なら本数に関係なく
    「人がよく通る範囲」と「端で急に検出が途切れる位置」が読み取れる。
    観測区間の外縁を決めるための図。
    """
    out_dir, result = _resolve(out_root, date, camera)
    background = _background_frame(video, result)
    height, width = result.meta.height, result.meta.width

    n_rows = max(1, -(-height // bin_px))
    n_cols = max(1, -(-width // bin_px))
    hist = np.zeros((n_rows, n_cols), dtype=np.float32)
    n_points = 0
    for points in result.tracks.values():
        for p in points:
            col = min(n_cols - 1, max(0, int(p.foot_point[0] // bin_px)))
            row = min(n_rows - 1, max(0, int(p.foot_point[1] // bin_px)))
            hist[row, col] += 1.0
            n_points += 1

    peak = float(hist.max())
    smoothed = cv2.GaussianBlur(hist, (0, 0), DENSITY_SMOOTH_SIGMA)
    density = cv2.resize(smoothed, (width, height), interpolation=cv2.INTER_LINEAR)

    top = float(density.max())
    # 密度は一部の動線に極端に偏るため、平方根を取って低密度側も見えるようにする
    norm = np.sqrt(np.clip(density, 0.0, None) / top) if top > 0 else density
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)

    alpha = norm * 0.80
    alpha[density <= 0.0] = 0.0        # 密度0の領域は完全に透明にする
    alpha3 = alpha[:, :, None]
    canvas = (background * (1.0 - alpha3) + heat * alpha3).astype(np.uint8)

    _draw_colorbar(canvas, peak)
    _put_label(
        canvas,
        f"{result.meta.date} {result.meta.camera}  foot points={n_points}  bin={bin_px}px",
        (14, 36), (255, 255, 255), scale=0.8,
    )

    dest = _viz_dir(out_dir) / f"{camera}_density.png"
    cv2.imwrite(str(dest), canvas)
    print(f"wrote {dest} ({n_points} points, peak {peak:.0f}/bin)")
    return dest


def _draw_colorbar(canvas: np.ndarray, peak: float) -> None:
    """密度の目盛りを右下に描く。表示は平方根スケールである点も添える。"""
    height, width = canvas.shape[:2]
    bar_w, bar_h = 260, 16
    x0, y0 = width - bar_w - 30, height - 60

    ramp = np.linspace(0, 255, bar_w, dtype=np.uint8)[None, :].repeat(bar_h, axis=0)
    canvas[y0 : y0 + bar_h, x0 : x0 + bar_w] = cv2.applyColorMap(ramp, cv2.COLORMAP_TURBO)
    cv2.rectangle(canvas, (x0, y0), (x0 + bar_w, y0 + bar_h), (255, 255, 255), 1, cv2.LINE_AA)
    _put_label(canvas, "0", (x0 - 4, y0 + bar_h + 22), (255, 255, 255), scale=0.5)
    _put_label(canvas, f"{peak:.0f} pts/bin", (x0 + bar_w - 90, y0 + bar_h + 22), (255, 255, 255), scale=0.5)
    _put_label(canvas, "density (sqrt scale)", (x0, y0 - 8), (255, 255, 255), scale=0.5)


def _draw_grid(canvas: np.ndarray, step: int = ENDPOINT_GRID_PX) -> None:
    """矩形の座標を読み取れるよう、薄いグリッドと目盛りを重ねる。"""
    height, width = canvas.shape[:2]
    overlay = canvas.copy()
    for x in range(0, width, step):
        cv2.line(overlay, (x, 0), (x, height), (110, 110, 110), 1)
    for y in range(0, height, step):
        cv2.line(overlay, (0, y), (width, y), (110, 110, 110), 1)
    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)

    for x in range(step, width, step):
        _put_label(canvas, str(x), (x + 4, 18), (60, 60, 60), scale=0.45)
    for y in range(step, height, step):
        _put_label(canvas, str(y), (4, y - 5), (60, 60, 60), scale=0.45)


def _direction_arrow(
    points: list[FramePoint], fps: float, entering: bool
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """進入（または退出）方向を示す短い矢印の始点と終点を返す。

    実際の位置まで線を引くと図が軌跡で埋まるので、方向だけを取り出して
    一定の長さの矢印にする。動きが無ければ None。
    """
    offset = ENDPOINT_ARROW_SEC * fps
    if entering:
        anchor = points[0]
        target = next(
            (p for p in points if p.frame_idx - anchor.frame_idx >= offset), points[-1]
        )
        dx = target.foot_point[0] - anchor.foot_point[0]
        dy = target.foot_point[1] - anchor.foot_point[1]
    else:
        anchor = points[-1]
        target = next(
            (p for p in reversed(points) if anchor.frame_idx - p.frame_idx >= offset),
            points[0],
        )
        dx = anchor.foot_point[0] - target.foot_point[0]
        dy = anchor.foot_point[1] - target.foot_point[1]

    norm = (dx * dx + dy * dy) ** 0.5
    if norm < 1e-6:
        return None
    ux, uy = dx / norm * ENDPOINT_ARROW_PX, dy / norm * ENDPOINT_ARROW_PX
    px, py = anchor.foot_point[0], anchor.foot_point[1]
    if entering:
        # 開始点から出ていく向き
        return (int(px), int(py)), (int(px + ux), int(py + uy))
    # 終了点へ入ってくる向き
    return (int(px - ux), int(py - uy)), (int(px), int(py))


def viz_endpoints(video: str | Path, date: str, camera: str, out_root: str | Path) -> Path:
    """各トラックの開始点と終了点だけをプロットした静止画を書き出す。

    遷移領域の矩形と、進入・退出方向の角度範囲をこの図から決める。軌跡を
    描かないので点が重ならず、座標を読み取るためのグリッドを重ねられる。
    """
    out_dir, result = _resolve(out_root, date, camera)
    canvas = _faded(_background_frame(video, result))
    _draw_grid(canvas)

    tracks = _ordered_tracks(result)
    placed: list[tuple[int, int, int, int]] = []
    n_short = 0

    # 短いトラックは誤検出や一瞬の写り込みが多いので、灰色の小さい点にして
    # 遷移領域の判断材料から外せるようにする。
    for tid, points, _ in tracks:
        if len(points) >= ENDPOINT_SHORT_TRACK_FRAMES:
            continue
        n_short += 1
        for p in (points[0], points[-1]):
            c = (int(p.foot_point[0]), int(p.foot_point[1]))
            cv2.circle(canvas, c, 4, SHORT_TRACK_COLOR, -1, cv2.LINE_AA)
        lx, ly = _place_label(placed, (int(points[0].foot_point[0]), int(points[0].foot_point[1])), tid, 0.45)
        _put_label(canvas, tid, (lx, ly), SHORT_TRACK_COLOR, scale=0.45)

    for tid, points, _ in tracks:
        if len(points) < ENDPOINT_SHORT_TRACK_FRAMES:
            continue
        for entering, color in ((True, START_COLOR), (False, END_COLOR)):
            arrow = _direction_arrow(points, result.meta.fps, entering)
            if arrow is not None:
                cv2.arrowedLine(canvas, arrow[0], arrow[1], (255, 255, 255), 5, cv2.LINE_AA, tipLength=0.35)
                cv2.arrowedLine(canvas, arrow[0], arrow[1], color, 2, cv2.LINE_AA, tipLength=0.35)

            p = points[0] if entering else points[-1]
            c = (int(p.foot_point[0]), int(p.foot_point[1]))
            cv2.circle(canvas, c, 10, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, c, 7, color, -1, cv2.LINE_AA)
            lx, ly = _place_label(placed, c, tid, 0.5)
            _put_label(canvas, tid, (lx, ly), color, scale=0.5)

    _put_label(
        canvas,
        f"{result.meta.date} {result.meta.camera}  tracks={len(tracks)} "
        f"(short={n_short})  grid={ENDPOINT_GRID_PX}px",
        (14, 36), (0, 0, 0), scale=0.8,
    )
    _put_label(canvas, "blue = start / entry direction", (14, 66), START_COLOR, scale=0.6)
    _put_label(canvas, "red  = end / exit direction", (14, 90), END_COLOR, scale=0.6)
    _put_label(
        canvas, f"gray = shorter than {ENDPOINT_SHORT_TRACK_FRAMES} frames",
        (14, 114), SHORT_TRACK_COLOR, scale=0.6,
    )

    dest = _viz_dir(out_dir) / f"{camera}_endpoints.png"
    cv2.imwrite(str(dest), canvas)
    print(f"wrote {dest} ({len(tracks)} tracks, {n_short} short)")
    return dest


def _region_polygon(region: Region) -> np.ndarray:
    """rect も含めて、塗りと輪郭に使える頂点列にそろえる。"""
    if region.type == "rect":
        (x0, y0), (x1, y1) = region.points
        return np.array(
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.int32
        )
    return np.asarray(region.points, dtype=np.int32)


def _fill_regions(
    canvas: np.ndarray,
    regions: list[Region],
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    """領域を半透明で塗り、輪郭は実線で描く。

    塗りだけ別バッファで合成するのは、輪郭まで薄くすると境界が読めなくなるため。
    """
    if not regions:
        return
    polygons = [_region_polygon(r) for r in regions]
    overlay = canvas.copy()
    cv2.fillPoly(overlay, polygons, color)
    cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)
    cv2.polylines(canvas, polygons, True, color, 2, cv2.LINE_AA)


def _draw_sector(
    canvas: np.ndarray,
    center: tuple[int, int],
    lo: float,
    hi: float,
    radius: int,
    color: tuple[int, int, int],
) -> None:
    """角度範囲を扇形で描く。0 をまたぐ区間は2つの扇に分けて描く。

    cv2.ellipse の角度は画像座標系で右向き0度・時計回りなので、
    cameras.yaml の角度をそのまま渡せる。
    """
    spans = [(lo, hi)] if lo <= hi else [(lo, 360.0), (0.0, hi)]

    overlay = canvas.copy()
    for start, end in spans:
        cv2.ellipse(overlay, center, (radius, radius), 0, start, end, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, SECTOR_ALPHA, canvas, 1.0 - SECTOR_ALPHA, 0, canvas)

    for start, end in spans:
        cv2.ellipse(canvas, center, (radius, radius), 0, start, end, color, 2, cv2.LINE_AA)
    # 範囲の両端を半径線で示し、扇のどこが境界か分かるようにする
    for angle in (lo, hi):
        rad = np.radians(angle)
        tip = (int(center[0] + radius * np.cos(rad)), int(center[1] + radius * np.sin(rad)))
        cv2.line(canvas, center, tip, color, 2, cv2.LINE_AA)


def _zone_caption(zone: TransitionZone) -> str:
    """遷移先を1行にまとめる。room 指定があれば cam:room と書く。"""
    if not zone.to:
        return "to: (未設定)"
    parts = [t.cam if t.room is None else f"{t.cam}:{t.room}" for t in zone.to]
    return "to " + ", ".join(parts)


def _draw_legend(canvas: np.ndarray) -> None:
    height = canvas.shape[0]
    rows = [
        ("exclude_regions (観測区間の外側)", EXCLUDE_COLOR),
        ("transition_zones", ZONE_COLOR),
        ("exit_dir (退出方向)", EXIT_COLOR),
        ("entry_dir (進入方向)", ENTRY_COLOR),
    ]
    x0, y0 = 14, height - 24 - 26 * len(rows)

    box = canvas.copy()
    cv2.rectangle(box, (x0 - 10, y0 - 26), (x0 + 400, height - 14), (255, 255, 255), -1)
    cv2.addWeighted(box, 0.65, canvas, 0.35, 0, canvas)

    for i, (text, color) in enumerate(rows):
        y = y0 + 26 * i
        cv2.rectangle(canvas, (x0, y - 12), (x0 + 22, y + 2), color, -1)
        _put_label(canvas, text, (x0 + 32, y + 2), (30, 30, 30), scale=0.55)


def viz_region(
    video: str | Path,
    date: str,
    camera: str,
    out_root: str | Path,
    config_path: str | Path = "configs/cameras.yaml",
) -> Path:
    """cameras.yaml に定義した領域を代表フレームに重ねた静止画を書き出す。

    座標を書いては確かめる作業のための図なので、`tracks.json` は読まない。
    まだ定義していないカメラでも、背景だけの画像を出して例外にしない。
    """
    out_dir = output_dir(out_root, date, camera)
    cameras = load_cameras(config_path)
    config = cameras.get(camera)
    if config is None:
        print(f"  [注意] {config_path} に {camera} の定義がありません。背景だけ出力します。")
        config = CameraConfig(name=camera)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n_frames // 2))
    ok, canvas = cap.read()
    cap.release()
    if not ok:
        width, height = config.image_size or (1920, 1080)
        canvas = np.full((height, width, 3), 255, np.uint8)

    _draw_grid(canvas)
    _fill_regions(canvas, config.exclude_regions, EXCLUDE_COLOR, EXCLUDE_ALPHA)
    _fill_regions(
        canvas, [z.region for z in config.transition_zones], ZONE_COLOR, ZONE_ALPHA
    )

    for zone in config.transition_zones:
        center = region_center(zone.region)
        if zone.exit_dir is not None:
            _draw_sector(canvas, center, *zone.exit_dir, EXIT_RADIUS, EXIT_COLOR)
        if zone.entry_dir is not None:
            _draw_sector(canvas, center, *zone.entry_dir, ENTRY_RADIUS, ENTRY_COLOR)

        cv2.circle(canvas, center, 5, (255, 255, 255), -1, cv2.LINE_AA)
        _put_label(canvas, zone.id, (center[0] + 12, center[1] - 8), (255, 255, 255), scale=0.75)
        _put_label(
            canvas, _zone_caption(zone), (center[0] + 12, center[1] + 16),
            (235, 235, 235), scale=0.5,
        )

    _put_label(
        canvas,
        f"{date} {camera}  exclude={len(config.exclude_regions)}  "
        f"zones={len(config.transition_zones)}  grid={ENDPOINT_GRID_PX}px",
        (14, 36), (255, 255, 255), scale=0.8,
    )
    _draw_legend(canvas)

    dest = _viz_dir(out_dir) / f"{camera}_region.png"
    cv2.imwrite(str(dest), canvas)
    print(
        f"wrote {dest} (exclude {len(config.exclude_regions)}, "
        f"zones {len(config.transition_zones)})"
    )
    return dest
