"""schema の往復シリアライズと、出力パス規約のテスト。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mct.io_utils import (
    crop_path,
    load_tracking_result,
    mask_path,
    output_dir,
    result_path,
    save_tracking_result,
)
from mct.schema import FramePoint, TrackingResult, TrackMeta


def _keypoints() -> list[list[float]]:
    """COCO 17点のダミー。"""
    return [[float(i) * 2.0, float(i) * 3.0, 0.5 + i * 0.01] for i in range(17)]


def _meta(camera: str = "camA") -> TrackMeta:
    return TrackMeta(
        date="0720",
        camera=camera,
        width=1920,
        height=1080,
        fps=30.0,
        n_frames=1800,
        model_pose="yolo11x-pose.pt",
        model_seg="yolo11x-seg.pt",
        tracker="botsort.yaml",
        n_saved_per_track=90,
        created_at="2026-07-20T13:45:00+09:00",
    )


def _result(camera: str = "camA") -> TrackingResult:
    """saved=True（keypoints あり）と saved=False（None）の両方を含む結果。"""
    return TrackingResult(
        meta=_meta(camera),
        tracks={
            "1": [
                FramePoint(
                    frame_idx=1,
                    bbox=[100.0, 200.0, 50.0, 120.0],
                    conf=0.91,
                    foot_point=[125.0, 320.0],
                    saved=True,
                    keypoints=_keypoints(),
                    crop_box=[95.0, 188.0, 60.0, 144.0],
                ),
                FramePoint(
                    frame_idx=2,
                    bbox=[102.5, 201.0, 50.0, 120.0],
                    conf=0.88,
                    foot_point=[127.5, 321.0],
                    saved=False,
                    keypoints=None,
                ),
            ],
            "7": [
                FramePoint(
                    frame_idx=120,
                    bbox=[0.0, 0.0, 10.0, 20.0],
                    conf=1.0,
                    foot_point=[5.0, 20.0],
                    saved=False,
                    keypoints=None,
                ),
            ],
        },
    )


def test_save_load_roundtrip(tmp_path: Path) -> None:
    original = _result()
    path = save_tracking_result(original, tmp_path / "0720" / "camA" / "r.json")

    assert path.exists()
    assert load_tracking_result(path) == original


def test_empty_tracks_roundtrip(tmp_path: Path) -> None:
    original = TrackingResult(meta=_meta(), tracks={})
    path = save_tracking_result(original, tmp_path / "r.json")

    assert load_tracking_result(path) == original


def test_dict_roundtrip_through_json() -> None:
    original = _result()
    restored = TrackingResult.from_dict(json.loads(json.dumps(original.to_dict())))

    assert restored == original


def test_frame_point_roundtrip_both_keypoint_cases() -> None:
    with_kps = FramePoint(
        frame_idx=10,
        bbox=[1.0, 2.0, 3.0, 4.0],
        conf=0.5,
        foot_point=[2.5, 6.0],
        saved=True,
        keypoints=_keypoints(),
        crop_box=[0.5, 1.5, 4.0, 5.0],
    )
    without_kps = FramePoint(
        frame_idx=11,
        bbox=[1.0, 2.0, 3.0, 4.0],
        conf=0.5,
        foot_point=[2.5, 6.0],
        saved=False,
        keypoints=None,
    )

    assert FramePoint.from_dict(with_kps.to_dict()) == with_kps
    assert FramePoint.from_dict(without_kps.to_dict()) == without_kps
    assert with_kps.to_dict()["keypoints"] is not None
    assert with_kps.to_dict()["crop_box"] == [0.5, 1.5, 4.0, 5.0]
    assert without_kps.to_dict()["keypoints"] is None
    assert without_kps.to_dict()["crop_box"] is None


def test_int_valued_floats_restore_as_float() -> None:
    """JSON 上で 1 と書かれた conf が 1.0 として戻ること。"""
    d = {
        "frame_idx": 1,
        "bbox": [1, 2, 3, 4],
        "conf": 1,
        "foot_point": [2, 5],
        "saved": False,
        "keypoints": None,
    }
    p = FramePoint.from_dict(d)

    assert isinstance(p.conf, float)
    assert all(isinstance(v, float) for v in p.bbox)
    assert all(isinstance(v, float) for v in p.foot_point)


def test_to_dict_does_not_alias_lists() -> None:
    p = _result().tracks["1"][0]
    d = p.to_dict()
    d["bbox"][0] = -1.0
    assert d["keypoints"] is not None
    d["keypoints"][0][0] = -1.0

    assert p.bbox[0] != -1.0
    assert p.keypoints is not None
    assert p.keypoints[0][0] != -1.0


def test_keypoints_rejected_when_not_saved() -> None:
    with pytest.raises(ValueError):
        FramePoint(
            frame_idx=1,
            bbox=[1.0, 2.0, 3.0, 4.0],
            conf=0.5,
            foot_point=[2.5, 6.0],
            saved=False,
            keypoints=_keypoints(),
        )


def test_crop_box_rejected_when_not_saved() -> None:
    with pytest.raises(ValueError):
        FramePoint(
            frame_idx=1,
            bbox=[1.0, 2.0, 3.0, 4.0],
            conf=0.5,
            foot_point=[2.5, 6.0],
            saved=False,
            crop_box=[0.0, 0.0, 5.0, 6.0],
        )


def test_crop_box_is_optional_for_old_json() -> None:
    """フェーズ0 が書いた crop_box 無しの JSON も読み戻せる。"""
    p = FramePoint.from_dict(
        {
            "frame_idx": 1,
            "bbox": [1.0, 2.0, 3.0, 4.0],
            "conf": 0.5,
            "foot_point": [2.5, 6.0],
            "saved": False,
            "keypoints": None,
        }
    )
    assert p.crop_box is None


def test_json_is_compact_and_not_ascii_escaped(tmp_path: Path) -> None:
    result = _result(camera="正門カメラ")
    path = save_tracking_result(result, tmp_path / "r.json")
    text = path.read_text(encoding="utf-8")

    assert "正門カメラ" in text  # ensure_ascii=False
    assert "\n" not in text  # インデントなし
    assert ", " not in text and ": " not in text  # 区切りのスペースもなし


def test_output_dir_creates_directory(tmp_path: Path) -> None:
    d = output_dir(tmp_path / "outputs", "0720", "camA")

    assert d == tmp_path / "outputs" / "0720" / "camA"
    assert d.is_dir()
    assert output_dir(tmp_path / "outputs", "0720", "camA") == d  # 再実行しても壊れない


def test_artifact_paths(tmp_path: Path) -> None:
    d = output_dir(tmp_path, "0720", "camA")

    assert crop_path(d, 7, 123) == d / "crops" / "t007_f000123.jpg"
    assert mask_path(d, 7, 123) == d / "masks" / "t007_f000123.png"
    assert result_path(d) == d / "tracks.json"

    # tracks のキーは文字列なので、そのまま渡しても同じパスになること。
    assert crop_path(d, "7", 123) == crop_path(d, 7, 123)

    # パス組み立てだけで、ディレクトリは作らない。
    assert not (d / "crops").exists()
    assert not (d / "masks").exists()
