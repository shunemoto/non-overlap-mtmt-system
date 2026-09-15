"""Ground Truth を data/gt/ に取り込む。

元データは日付ごとの ZIP に入った MOT 形式のアノテーションと、Global ID の
対応表。対応表には個人名の列が含まれるので、**取り込みの時点で落とす**。
こうしておけば、以降のコードが誤って個人名に触れることがない。

    python scripts/import_gt.py [SRC_ROOT] [GID_CSV]

出力:
    data/gt/{date}/{camera}/gt.txt
    data/gt/globalID.csv        （sequence_id, global_id, camera_id, local_track_id）
"""

from __future__ import annotations

import collections
import csv
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = Path.home() / "Downloads" / "情報シンポ2026" / "corrected_annotation"
DEFAULT_GID = (
    Path.home() / "Downloads" / "情報シンポ2026" / "global_ids"
    / "globalID 18.16.34 18.57.24.csv"
)
DEST_ROOT = REPO_ROOT / "data" / "gt"

DATES = ("0720", "0727", "0730", "0803")
CAMERA_LETTERS = "ABCDEFGH"

# 取り込む列。name と anonymize は個人情報なので持ち込まない。
KEEP_COLUMNS = ["sequence_id", "global_id", "camera_id", "local_track_id"]

EXPECTED = {"global_ids": 168, "tracks": 423, "transitions": 255}


def import_boxes(src_root: Path) -> int:
    """32個の ZIP から gt.txt を取り出す。"""
    n = 0
    for date in DATES:
        for letter in CAMERA_LETTERS:
            src = src_root / date / f"{date}_camera{letter}_displayid.zip"
            if not src.is_file():
                print(f"  !! 見つかりません: {src}")
                continue
            dest = DEST_ROOT / date / f"cam{letter}" / "gt.txt"
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(src) as zf:
                dest.write_bytes(zf.read("gt/gt.txt"))
            n += 1
    return n


def import_global_ids(gid_csv: Path) -> list[dict[str, str]]:
    """Global ID の対応表を、必要な4列だけにして書き出す。"""
    with gid_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = [{k: r[k] for k in KEEP_COLUMNS} for r in csv.DictReader(f)]

    dest = DEST_ROOT / "globalID.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=KEEP_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def report(rows: list[dict[str, str]], n_boxes: int) -> bool:
    """件数が想定どおりかを確かめる。ずれていれば元データを疑う。"""
    by_gid: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for r in rows:
        by_gid[(r["sequence_id"], r["global_id"])].append(r["camera_id"])

    got = {
        "global_ids": len(by_gid),
        "tracks": len(rows),
        # 遷移は「同一 Global ID のトラックを時間順に並べたときの隣接対」
        "transitions": sum(len(v) - 1 for v in by_gid.values() if len(v) > 1),
    }

    print(f"\n  gt.txt         : {n_boxes} / 32 ファイル")
    ok = n_boxes == 32
    for key, label in (
        ("global_ids", "Global ID  "),
        ("tracks", "トラック   "),
        ("transitions", "カメラ間遷移"),
    ):
        mark = "" if got[key] == EXPECTED[key] else f"  ← 想定 {EXPECTED[key]}"
        ok &= got[key] == EXPECTED[key]
        print(f"  {label}   : {got[key]}{mark}")

    per_day = collections.Counter(s for s, _ in by_gid)
    print(f"  日付別 GID     : {dict(sorted(per_day.items()))}")
    return ok


def main(argv: list[str]) -> int:
    src_root = Path(argv[1]).expanduser() if len(argv) > 1 else DEFAULT_SRC
    gid_csv = Path(argv[2]).expanduser() if len(argv) > 2 else DEFAULT_GID

    for path, label in ((src_root, "アノテーション"), (gid_csv, "Global ID 対応表")):
        if not path.exists():
            print(f"{label}が見つかりません: {path}", file=sys.stderr)
            return 1

    print(f"取り込み元: {src_root}")
    n_boxes = import_boxes(src_root)
    rows = import_global_ids(gid_csv)

    print(f"\n書き出し先: {DEST_ROOT}")
    print(f"  globalID.csv の列: {KEEP_COLUMNS}（name と anonymize は取り込まない）")
    ok = report(rows, n_boxes)

    print("\n件数は想定どおりです。" if ok else "\n!! 件数が想定と違います。元データを確認してください。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
