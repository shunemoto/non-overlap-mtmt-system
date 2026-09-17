"""トラック1本ごとの外観記述子を作る。

crop・マスク・骨格点はフェーズ1と4.5ですでにファイルになっているので、
ここでモデルは動かさない（**torch / ultralytics は import しない**）。

記述子はトラック単位で1組だけ作る。候補ペアの数はトラック数の二乗に近い
オーダーになるので、ペアごとに画像を読み直すと掛かる時間が桁で変わる。
パラメータを振り直したときに再計算する範囲を抑える意味もある。

色は **H×S だけ**を使い、明度 V は捨てる。カメラごとに露出が違うので、
V を入れると同じ服が別人に見える。低彩度の画素は色相がノイズになるため、
**無彩色ビン1つ**にまとめて H×S の格子から外す。
"""

from __future__ import annotations

import collections
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from mct.io_utils import crop_path, mask_path
from mct.schema import FramePoint, TrackingResult

__all__ = [
    "JOINT_NAMES",
    "N_JOINTS",
    "AppearanceDescriptor",
    "AppearanceStats",
    "FrameFeature",
    "split_y",
    "color_histogram",
    "joint_points",
    "patch_window",
    "sift_features",
    "aggregate",
    "build_descriptors",
    "write_descriptors",
    "load_descriptors",
    "print_stats",
    "appearance_params",
    "DESCRIPTOR_FILENAME",
]

DESCRIPTOR_FILENAME = "appearance.json"

# COCO 17点のうち使うもの
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12

# 胴体中心（肩と腰の中点）は単独 AUC が 0.58〜0.60 とほぼ偶然で、肩と腰の
# パッチに大きく重なるため独自の情報を持たない。5点の平均を薄めるだけなので外した。
JOINT_NAMES = ("L肩", "R肩", "L腰", "R腰", "首元")
N_JOINTS = len(JOINT_NAMES)

# 肩から bbox 下辺までのどこで上下を分けるか（腰が取れなかったときの代用）
SHOULDER_TO_HIP_RATIO = 0.45

DEFAULT_APPEARANCE: dict[str, Any] = {
    "hist_bins": [8, 8],
    "achromatic_threshold": 0.10,
    "sift_patch_ratio": 0.09,
    "sift_patch_size": 32,
    "sift_min_frames": 5,
    "lambda_missing": 0.9,
}

_SIFT = None


def _sift():
    """SIFT は生成が重いので使い回す。"""
    global _SIFT
    if _SIFT is None:
        _SIFT = cv2.SIFT_create()
    return _SIFT


def appearance_params(path: str | Path | None, **overrides: Any) -> dict[str, Any]:
    """既定値 → YAML の `appearance:` → 明示的な上書き、の順にマージする。"""
    from mct.config import load_params

    params = dict(DEFAULT_APPEARANCE)
    params.update(load_params(path).get("appearance") or {})
    params.update({k: v for k, v in overrides.items() if v is not None})
    return params


# --------------------------------------------------------------------------
# 上下分割
# --------------------------------------------------------------------------


def _mean_y(keypoints: list[list[float]] | None, indices: Iterable[int]) -> float | None:
    """指定した骨格点の y の平均。1つも検出できていなければ None。"""
    if keypoints is None:
        return None
    ys = [
        keypoints[i][1]
        for i in indices
        if i < len(keypoints) and keypoints[i][2] > 0
    ]
    return float(np.mean(ys)) if ys else None


def split_y(
    keypoints: list[list[float]] | None, bbox: list[float]
) -> tuple[float, str]:
    """上下を分ける y 座標（元画像座標系）と、その根拠。

    腰が最も素直な境目だが、後ろ向きや遮蔽で取れないことがある。肩が取れて
    いれば肩から下の 45% を腰の位置とみなし、それも無ければ bbox の中央で
    割る。根拠を返すのは、どの経路がどれだけ使われたかを診断で出すため。
    """
    hip = _mean_y(keypoints, (L_HIP, R_HIP))
    if hip is not None:
        return hip, "hip"

    shoulder = _mean_y(keypoints, (L_SHOULDER, R_SHOULDER))
    bottom = bbox[1] + bbox[3]
    if shoulder is not None and bottom > shoulder:
        return shoulder + (bottom - shoulder) * SHOULDER_TO_HIP_RATIO, "shoulder"

    return bbox[1] + bbox[3] * 0.5, "bbox"


# --------------------------------------------------------------------------
# 色ヒストグラム
# --------------------------------------------------------------------------


def color_histogram(
    crop_bgr: np.ndarray,
    mask: np.ndarray | None,
    split_row: int,
    params: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """上半身・下半身それぞれの H×S ヒストグラム（L1 正規化済み）。

    低彩度の画素は色相が定まらないので、`achromatic_threshold` 未満の彩度を
    まとめて末尾の1ビンに入れる。これをやらないと、白っぽい服や影の画素が
    色相全体にばらまかれてヒストグラムを汚す。

    `mask` が None なら crop 全体を使う。人物以外が混ざるが、色が何も
    取れないよりはましという判断。
    """
    n_h, n_s = (int(v) for v in params["hist_bins"])
    threshold = params.get("achromatic_threshold")

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    height = hsv.shape[0]
    row = int(np.clip(split_row, 0, height))

    valid = (
        np.ones(hsv.shape[:2], dtype=bool) if mask is None else mask > 127
    )

    upper = _histogram(hsv[:row], valid[:row], n_h, n_s, threshold)
    lower = _histogram(hsv[row:], valid[row:], n_h, n_s, threshold)
    return upper, lower


def _histogram(
    hsv: np.ndarray, valid: np.ndarray, n_h: int, n_s: int, threshold: float | None
) -> np.ndarray | None:
    """H×S の格子 + 無彩色ビン1つ。画素が無ければ None。"""
    size = n_h * n_s + 1
    if hsv.size == 0 or not valid.any():
        return None

    h = hsv[..., 0][valid].astype(np.float32)          # OpenCV の H は 0..179
    s = hsv[..., 1][valid].astype(np.float32)          # S は 0..255

    hist = np.zeros(size, dtype=np.float64)
    if threshold is None:
        chromatic = np.ones(len(s), dtype=bool)
    else:
        chromatic = s >= float(threshold) * 255.0
        hist[-1] = float((~chromatic).sum())

    if chromatic.any():
        # 上端がちょうど最大値のとき最後のビンからこぼれないよう clip する
        hi = np.clip((h[chromatic] / 180.0 * n_h).astype(int), 0, n_h - 1)
        si = np.clip((s[chromatic] / 256.0 * n_s).astype(int), 0, n_s - 1)
        np.add.at(hist, hi * n_s + si, 1.0)

    total = hist.sum()
    return hist / total if total > 0 else None


# --------------------------------------------------------------------------
# SIFT
# --------------------------------------------------------------------------


def joint_points(keypoints: list[list[float]] | None) -> list[tuple[float, float] | None]:
    """記述する5点（元画像座標系）。取れない点は None。

    肩・腰の4点に加えて、首元（肩の中点）を足す。首元は胸元の柄や襟の形が
    出るうえ、肩の2点とはパッチが重なりにくい位置にある。
    """
    def get(i: int) -> tuple[float, float] | None:
        if keypoints is None or i >= len(keypoints) or keypoints[i][2] <= 0:
            return None
        return (float(keypoints[i][0]), float(keypoints[i][1]))

    ls, rs = get(L_SHOULDER), get(R_SHOULDER)
    lh, rh = get(L_HIP), get(R_HIP)

    def mid(*points):
        usable = [p for p in points if p is not None]
        if len(usable) != len(points):
            return None
        return (
            float(np.mean([p[0] for p in usable])),
            float(np.mean([p[1] for p in usable])),
        )

    return [ls, rs, lh, rh, mid(ls, rs)]


def patch_window(
    cx: float, cy: float, radius: float, width: int, height: int
) -> tuple[int, int, int, int] | None:
    """(x, y, w, h)。crop からはみ出す場合は内側へずらす。

    足りない分をパディングで埋めると、そこに人工的なエッジが立って SIFT が
    それを拾ってしまう。窓ごと内側へ動かして実画素だけで埋める方が、
    中心が多少ずれても記述子は素直になる。
    """
    size = int(round(radius * 2))
    if size < 2 or size > width or size > height:
        return None

    x = int(round(cx - size / 2))
    y = int(round(cy - size / 2))
    x = max(0, min(x, width - size))
    y = max(0, min(y, height - size))
    return x, y, size, size


def sift_features(
    crop_gray: np.ndarray,
    keypoints: list[list[float]] | None,
    crop_box: list[float],
    bbox: list[float],
    params: dict[str, Any],
) -> list[np.ndarray | None]:
    """5点分の SIFT 記述子（L2 正規化済み）。取れない点は None。

    骨格点は元画像座標系なので crop の原点を引いて crop 座標系に直す。
    主方向は推定せず `angle=0` に固定する（`compute` を使い
    `detectAndCompute` は使わない）。人物は直立しているので回転不変性は
    要らず、無地の服では主方向がノイズで決まってしまう。

    マスクは使わない。マスクで背景を落とすと輪郭に沿った人工的なエッジが
    立ち、記述子がそれを拾う。
    """
    size = int(params["sift_patch_size"])
    descriptors: list[np.ndarray | None] = [None] * N_JOINTS

    radius = float(params["sift_patch_ratio"]) * float(bbox[3])
    height, width = crop_gray.shape[:2]
    ox, oy = float(crop_box[0]), float(crop_box[1])

    kp = cv2.KeyPoint(size / 2.0, size / 2.0, float(size))
    kp.angle = 0.0

    for j, point in enumerate(joint_points(keypoints)):
        if point is None:
            continue
        window = patch_window(point[0] - ox, point[1] - oy, radius, width, height)
        if window is None:
            continue
        x, y, w, h = window
        patch = cv2.resize(
            crop_gray[y : y + h, x : x + w], (size, size), interpolation=cv2.INTER_AREA
        )
        _, desc = _sift().compute(patch, [kp])
        if desc is None or len(desc) == 0:
            continue
        vector = desc[0].astype(np.float64)
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            continue                       # 完全に平坦なパッチ
        descriptors[j] = vector / norm

    return descriptors


# --------------------------------------------------------------------------
# 集約
# --------------------------------------------------------------------------


@dataclass
class FrameFeature:
    """1フレーム分の途中結果。集約前にだけ存在する。"""

    hist_upper: np.ndarray | None
    hist_lower: np.ndarray | None
    sift: list[np.ndarray | None]
    split_source: str
    has_mask: bool


@dataclass
class AppearanceDescriptor:
    """トラック1本ぶんの外観記述子。"""

    camera: str
    track_id: int
    hist_upper: list[float] | None = None
    hist_lower: list[float] | None = None
    sift: list[list[float]] | None = None
    sift_valid: list[bool] = field(default_factory=lambda: [False] * N_JOINTS)
    n_frames_used: int = 0

    @property
    def key(self) -> tuple[str, int]:
        return (self.camera, self.track_id)

    @property
    def n_valid_joints(self) -> int:
        return sum(self.sift_valid)

    def to_dict(self) -> dict[str, Any]:
        def r(values):
            return None if values is None else [round(float(v), 6) for v in values]

        return {
            "camera": self.camera,
            "track_id": self.track_id,
            "hist_upper": r(self.hist_upper),
            "hist_lower": r(self.hist_lower),
            "sift": None if self.sift is None else [r(d) for d in self.sift],
            "sift_valid": [bool(v) for v in self.sift_valid],
            "n_frames_used": self.n_frames_used,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AppearanceDescriptor:
        sift = d.get("sift")
        return cls(
            camera=str(d["camera"]),
            track_id=int(d["track_id"]),
            hist_upper=d.get("hist_upper"),
            hist_lower=d.get("hist_lower"),
            sift=None if sift is None else [list(v) for v in sift],
            sift_valid=[bool(v) for v in d.get("sift_valid", [])] or [False] * N_JOINTS,
            n_frames_used=int(d.get("n_frames_used", 0)),
        )


def _median_hist(values: list[np.ndarray]) -> list[float] | None:
    """ビンごとの中央値を取って再正規化する。

    平均だと、他人の映り込みやマスクの破綻が起きた数フレームに引きずられる。
    中央値なら少数の異常フレームは効かない。ビンごとに独立に中央値を取ると
    合計が1から外れるので、取り直した後に正規化する。
    """
    if not values:
        return None
    median = np.median(np.vstack(values), axis=0)
    total = float(median.sum())
    return list(median / total) if total > 0 else None


def aggregate(
    camera: str, track_id: int, frames: list[FrameFeature], params: dict[str, Any]
) -> AppearanceDescriptor:
    """フレーム列を1組の記述子にまとめる。"""
    min_frames = int(params["sift_min_frames"])

    hist_upper = _median_hist([f.hist_upper for f in frames if f.hist_upper is not None])
    hist_lower = _median_hist([f.hist_lower for f in frames if f.hist_lower is not None])

    sift: list[list[float]] = []
    valid: list[bool] = []
    for j in range(N_JOINTS):
        usable = [f.sift[j] for f in frames if f.sift[j] is not None]
        if len(usable) < min_frames:
            sift.append([0.0] * 128)
            valid.append(False)
            continue
        mean = np.mean(np.vstack(usable), axis=0)
        norm = float(np.linalg.norm(mean))
        if norm <= 0:
            sift.append([0.0] * 128)
            valid.append(False)
            continue
        sift.append(list(mean / norm))
        valid.append(True)

    return AppearanceDescriptor(
        camera=camera,
        track_id=track_id,
        hist_upper=hist_upper,
        hist_lower=hist_lower,
        sift=sift if any(valid) else None,
        sift_valid=valid,
        n_frames_used=len(frames),
    )


# --------------------------------------------------------------------------
# 入出力
# --------------------------------------------------------------------------


@dataclass
class AppearanceStats:
    """診断用の内訳。"""

    n_tracks: int = 0
    n_frames: int = 0
    n_no_mask: int = 0
    n_no_keypoints: int = 0
    split_sources: collections.Counter = field(default_factory=collections.Counter)
    n_valid_joints: collections.Counter = field(default_factory=collections.Counter)
    n_no_color: int = 0
    n_no_sift: int = 0
    joint_valid: collections.Counter = field(default_factory=collections.Counter)


def read_frame(
    out_dir: Path, track_id: str | int, point: FramePoint, params: dict[str, Any]
) -> FrameFeature | None:
    """1フレーム分の crop とマスクを読んで特徴を作る。"""
    crop = cv2.imread(str(crop_path(out_dir, track_id, point.frame_idx)))
    if crop is None:
        return None
    # マスクは取れないフレームがあるので、存在を確かめてから読む
    # （無いパスを imread に渡すと OpenCV が毎回警告を吐く）
    source = mask_path(out_dir, track_id, point.frame_idx)
    mask = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE) if source.is_file() else None
    if mask is not None and mask.shape[:2] != crop.shape[:2]:
        mask = None                        # 寸法が合わないマスクは信用しない

    line, source = split_y(point.keypoints, point.bbox)
    row = int(round(line - float(point.crop_box[1])))
    upper, lower = color_histogram(crop, mask, row, params)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sift = sift_features(gray, point.keypoints, point.crop_box, point.bbox, params)
    return FrameFeature(
        hist_upper=upper, hist_lower=lower, sift=sift,
        split_source=source, has_mask=mask is not None,
    )


def build_descriptors(
    result: TrackingResult, out_dir: str | Path, params: dict[str, Any]
) -> tuple[list[AppearanceDescriptor], AppearanceStats]:
    """1カメラ・1日分のトラックすべてについて記述子を作る。"""
    directory = Path(out_dir)
    stats = AppearanceStats()
    descriptors = []

    for track_id, points in sorted(result.tracks.items(), key=lambda kv: int(kv[0])):
        frames = []
        for point in points:
            if not point.saved or point.crop_box is None:
                continue
            feature = read_frame(directory, track_id, point, params)
            if feature is None:
                continue
            frames.append(feature)
            stats.n_frames += 1
            stats.n_no_mask += not feature.has_mask
            stats.n_no_keypoints += point.keypoints is None
            stats.split_sources[feature.split_source] += 1

        descriptor = aggregate(result.meta.camera, int(track_id), frames, params)
        descriptors.append(descriptor)
        stats.n_tracks += 1
        stats.n_valid_joints[descriptor.n_valid_joints] += 1
        stats.n_no_color += descriptor.hist_upper is None and descriptor.hist_lower is None
        stats.n_no_sift += descriptor.n_valid_joints < 2
        for j, ok in enumerate(descriptor.sift_valid):
            stats.joint_valid[j] += int(ok)

    return descriptors, stats


def write_descriptors(
    descriptors: list[AppearanceDescriptor],
    date: str,
    out_root: str | Path,
    params: dict[str, Any],
) -> Path:
    from mct.cache import cache_key

    dest = Path(out_root) / date / DESCRIPTOR_FILENAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "date": date,
            "n_tracks": len(descriptors),
            "n_joints": N_JOINTS,
            "joint_names": list(JOINT_NAMES),
            "config_hash": cache_key({"appearance": dict(sorted(params.items()))}),
        },
        "descriptors": [d.to_dict() for d in descriptors],
    }
    with dest.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    return dest


def load_descriptors(path: str | Path) -> dict[tuple[str, int], AppearanceDescriptor]:
    """`(camera, track_id)` で引ける形で読み戻す。"""
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    out = {}
    for d in payload.get("descriptors", []):
        descriptor = AppearanceDescriptor.from_dict(d)
        out[descriptor.key] = descriptor
    return out


def print_stats(date: str, camera: str, stats: AppearanceStats) -> None:
    frames = max(1, stats.n_frames)
    print(
        f"{date}/{camera}: トラック {stats.n_tracks} / フレーム {stats.n_frames}"
        f" / マスクなし {stats.n_no_mask} ({100*stats.n_no_mask/frames:.2f}%)"
        f" / 骨格点なし {stats.n_no_keypoints} ({100*stats.n_no_keypoints/frames:.2f}%)"
    )
    sources = " ".join(
        f"{k}:{v}({100*v/frames:.0f}%)" for k, v in stats.split_sources.most_common()
    )
    print(f"  上下分割の根拠: {sources}")
    joints = " ".join(f"{n}点:{c}" for n, c in sorted(stats.n_valid_joints.items()))
    print(f"  有効関節点数の分布: {joints}"
          f"   色なし {stats.n_no_color} / SIFT 不足(<2点) {stats.n_no_sift}")
