"""Ground Truth の読み込みのテスト。

個人名を持ち込まないことと、ID の表記ゆれを吸収できることを確かめる。
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from mct.gt import gt_transitions, load_ground_truth

REPO_ROOT = Path(__file__).resolve().parent.parent
GT_ROOT = REPO_ROOT / "data" / "gt"
has_gt = pytest.mark.skipif(
    not (GT_ROOT / "globalID.csv").is_file(),
    reason="data/gt がありません（scripts/import_gt.py を実行してください）",
)


def _write_gt(root: Path, camera: str, lines: list[str]) -> None:
    d = root / "0720" / camera
    d.mkdir(parents=True, exist_ok=True)
    (d / "gt.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_global_ids(root: Path, rows: list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "globalID.csv").write_text(
        "sequence_id,global_id,camera_id,local_track_id\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )


def test_reads_mot_columns(tmp_path: Path) -> None:
    _write_gt(tmp_path, "camA", [
        "10,1,100.0,200.0,50.0,120.0,1,1,1.0",
        "11,1,101.0,200.0,50.0,120.0,1,1,0.8",
    ])
    _write_global_ids(tmp_path, ["720,P001,A,1"])

    gt = load_ground_truth(tmp_path, "0720", cameras=("camA",))
    boxes = gt.boxes["camA"][1]

    assert len(boxes) == 2
    assert boxes[0].frame_idx == 10
    assert boxes[0].bbox == (100.0, 200.0, 50.0, 120.0)
    assert boxes[0].not_ignored is True
    assert boxes[1].visibility == pytest.approx(0.8)


def test_sequence_and_camera_ids_are_normalised(tmp_path: Path) -> None:
    """"720" -> "0720"、"A" -> "camA" の変換。"""
    _write_gt(tmp_path, "camA", ["10,1,100,200,50,120,1,1,1.0"])
    _write_global_ids(tmp_path, ["720,P001,A,1"])

    gt = load_ground_truth(tmp_path, "0720", cameras=("camA",))

    assert gt.global_ids == {("camA", 1): "P001"}


def test_other_dates_are_not_loaded(tmp_path: Path) -> None:
    _write_gt(tmp_path, "camA", ["10,1,100,200,50,120,1,1,1.0"])
    _write_global_ids(tmp_path, ["720,P001,A,1", "803,P002,A,1"])

    gt = load_ground_truth(tmp_path, "0720", cameras=("camA",))

    assert list(gt.global_ids.values()) == ["P001"]


def test_not_ignored_zero_is_ignored(tmp_path: Path) -> None:
    _write_gt(tmp_path, "camA", [
        "10,1,100,200,50,120,1,1,1.0",
        "10,2,900,200,50,120,0,1,1.0",
    ])
    _write_global_ids(tmp_path, ["720,P001,A,1"])

    gt = load_ground_truth(tmp_path, "0720", cameras=("camA",))

    assert ("camA", 2) in gt.ignored
    assert ("camA", 1) not in gt.ignored
    assert gt.evaluable("camA") == [1]


def test_track_without_global_id_is_ignored(tmp_path: Path) -> None:
    """Global ID が無いトラックは正誤を判定できないので評価から外す。"""
    _write_gt(tmp_path, "camA", [
        "10,1,100,200,50,120,1,1,1.0",
        "10,3,500,200,50,120,1,1,1.0",     # 対応表に無い
    ])
    _write_global_ids(tmp_path, ["720,P001,A,1"])

    gt = load_ground_truth(tmp_path, "0720", cameras=("camA",))

    assert ("camA", 3) in gt.ignored
    assert gt.evaluable("camA") == [1]


def test_keeping_tracks_without_global_id(tmp_path: Path) -> None:
    _write_gt(tmp_path, "camA", [
        "10,1,100,200,50,120,1,1,1.0",
        "10,3,500,200,50,120,1,1,1.0",
    ])
    _write_global_ids(tmp_path, ["720,P001,A,1"])

    gt = load_ground_truth(
        tmp_path, "0720", cameras=("camA",), ignore_without_global_id=False
    )

    assert ("camA", 3) not in gt.ignored


def test_missing_gt_file_warns_and_yields_empty(tmp_path: Path) -> None:
    """camH のように GT が無いカメラでも落ちない。"""
    _write_gt(tmp_path, "camA", ["10,1,100,200,50,120,1,1,1.0"])
    _write_global_ids(tmp_path, ["720,P001,A,1"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gt = load_ground_truth(tmp_path, "0720", cameras=("camA", "camH"))

    assert gt.boxes["camH"] == {}
    assert gt.evaluable("camH") == []
    assert any("camH" in str(w.message) for w in caught)


def test_global_id_referring_to_missing_track_warns(tmp_path: Path) -> None:
    _write_gt(tmp_path, "camA", ["10,1,100,200,50,120,1,1,1.0"])
    _write_global_ids(tmp_path, ["720,P001,A,1", "720,P002,A,16"])   # 16 は gt.txt に無い

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gt = load_ground_truth(tmp_path, "0720", cameras=("camA",))

    assert ("camA", 16) not in gt.global_ids
    assert any("16" in str(w.message) for w in caught)


def test_frames_with_gt(tmp_path: Path) -> None:
    _write_gt(tmp_path, "camA", [
        "10,1,100,200,50,120,1,1,1.0",
        "12,1,100,200,50,120,1,1,1.0",
    ])
    _write_global_ids(tmp_path, ["720,P001,A,1"])

    gt = load_ground_truth(tmp_path, "0720", cameras=("camA",))

    assert gt.frames_with_gt("camA") == {10, 12}


def test_transitions_are_consecutive_pairs_in_time(tmp_path: Path) -> None:
    """同一 Global ID のトラックを出現順に並べ、隣り合う対を遷移とする。"""
    _write_gt(tmp_path, "camA", ["10,1,100,200,50,120,1,1,1.0"])
    _write_gt(tmp_path, "camB", ["50,2,100,200,50,120,1,1,1.0"])
    _write_gt(tmp_path, "camC", ["90,3,100,200,50,120,1,1,1.0"])
    _write_global_ids(tmp_path, ["720,P001,A,1", "720,P001,B,2", "720,P001,C,3"])

    transitions = gt_transitions(
        load_ground_truth(tmp_path, "0720", cameras=("camA", "camB", "camC"))
    )

    # 3トラックなら遷移は2件（トラック数 − 1）
    assert len(transitions) == 2
    assert (transitions[0].from_cam, transitions[0].to_cam) == ("camA", "camB")
    assert (transitions[1].from_cam, transitions[1].to_cam) == ("camB", "camC")


def test_self_loop_transition(tmp_path: Path) -> None:
    _write_gt(tmp_path, "camA", [
        "10,1,100,200,50,120,1,1,1.0",
        "500,4,100,200,50,120,1,1,1.0",
    ])
    _write_global_ids(tmp_path, ["720,P001,A,1", "720,P001,A,4"])

    transitions = gt_transitions(load_ground_truth(tmp_path, "0720", cameras=("camA",)))

    assert len(transitions) == 1
    assert transitions[0].is_self_loop


# --- 実データ -------------------------------------------------------------


@has_gt
def test_real_gt_counts() -> None:
    """取り込んだ GT が想定の件数であること。"""
    n_gid, n_tracks, n_transitions = set(), 0, 0
    for date in ("0720", "0727", "0730", "0803"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gt = load_ground_truth(GT_ROOT, date)
        n_gid |= {(date, g) for g in gt.global_ids.values()}
        n_tracks += len(gt.global_ids)
        n_transitions += len(gt_transitions(gt))

    assert len(n_gid) == 168
    # 0720/camE の local_track_id 16 が gt.txt に無いので1件落ちる
    assert n_tracks == 422
    assert n_transitions == 254


@has_gt
def test_real_gt_has_no_name_column() -> None:
    """取り込んだ対応表に個人名の列が残っていないこと。"""
    header = (GT_ROOT / "globalID.csv").read_text(encoding="utf-8").splitlines()[0]

    assert header.split(",") == ["sequence_id", "global_id", "camera_id", "local_track_id"]
    assert "name" not in header and "anonymize" not in header
