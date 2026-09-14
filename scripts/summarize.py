"""日付ごとの処理結果を一覧表にまとめる。

`tracks.json` と `logs/{date}_summary.tsv` を突き合わせ、カメラ横断で
トラック数・トラック長・マスク失敗数・所要時間・人物が写っていた割合を出す。

    python scripts/summarize.py 0720
    python scripts/summarize.py 0720 --remaining 24
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from mct.io_utils import load_tracking_result, result_path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument(
        "--remaining", type=int, default=0,
        help="この本数分の推定所要時間を末尾に出す",
    )
    args = ap.parse_args()

    # 所要時間はログ側にしかないので TSV から拾う
    timings: dict[str, tuple[str, float, float]] = {}
    tsv = REPO_ROOT / args.log_dir / f"{args.date}_summary.tsv"
    if tsv.is_file():
        for line in tsv.read_text().splitlines()[1:]:
            cells = line.split("\t")
            if len(cells) >= 4:
                camera, status = cells[0], cells[1]
                track_s = float(cells[2]) if cells[2] else 0.0
                viz_s = float(cells[3]) if cells[3] else 0.0
                timings[camera] = (status, track_s, viz_s)

    rows: list[tuple] = []
    for camera in sorted({*timings} | _cameras_on_disk(args.out, args.date)):
        # output_dir() はディレクトリを作ってしまうので、集計では使わない
        path = result_path(Path(args.out) / args.date / camera)
        if not path.is_file():
            rows.append((camera, None))
            continue

        result = load_tracking_result(path)
        lengths = sorted(len(p) for p in result.tracks.values())
        saved = sum(1 for ps in result.tracks.values() for p in ps if p.saved)
        out_dir = path.parent
        n_masks = len(list((out_dir / "masks").glob("*.png")))
        occupied = len({p.frame_idx for ps in result.tracks.values() for p in ps})

        status, track_s, viz_s = timings.get(camera, ("?", 0.0, 0.0))
        rows.append(
            (
                camera, status, len(result.tracks), lengths, saved, saved - n_masks,
                occupied, result.meta.n_frames, result.meta.fps, track_s, viz_s,
            )
        )

    _print_table(rows)
    _print_totals(rows, args.remaining)
    return 0


def _cameras_on_disk(out: str, date: str) -> set[str]:
    d = REPO_ROOT / out / date
    return {p.name for p in d.iterdir() if p.is_dir()} if d.is_dir() else set()


def _print_table(rows: list[tuple]) -> None:
    head = (
        f"{'camera':<8}{'tracks':>7}{'長さ min/中央/max':>20}{'crop':>7}"
        f"{'mask失敗':>9}{'人物あり':>12}{'track':>9}{'viz':>8}{'計':>8}"
    )
    print(head)
    print("-" * len(head))
    for row in rows:
        if len(row) == 2:
            print(f"{row[0]:<8}{'(未処理)':>7}")
            continue
        (camera, _status, n_tracks, lengths, saved, n_no_mask,
         occupied, n_frames, fps, track_s, viz_s) = row
        span = f"{lengths[0]}/{int(statistics.median(lengths))}/{lengths[-1]}" if lengths else "-"
        ratio = 100.0 * occupied / n_frames if n_frames else 0.0
        print(
            f"{camera:<8}{n_tracks:>7}{span:>20}{saved:>7}{n_no_mask:>9}"
            f"{occupied:>7} ({ratio:4.1f}%){track_s/60:>8.1f}分{viz_s/60:>7.1f}分"
            f"{(track_s+viz_s)/60:>7.1f}分"
        )


def _print_totals(rows: list[tuple], remaining: int) -> None:
    done = [r for r in rows if len(r) > 2]
    if not done:
        return

    total_s = sum(r[9] + r[10] for r in done)
    total_frames = sum(r[7] for r in done)
    total_occupied = sum(r[6] for r in done)
    print("-" * 88)
    print(
        f"{'合計':<8}{sum(r[2] for r in done):>7}{'':>20}"
        f"{sum(r[4] for r in done):>7}{sum(r[5] for r in done):>9}"
        f"{total_occupied:>7} ({100*total_occupied/total_frames:4.1f}%)"
        f"{'':>9}{'':>8}{total_s/3600:>7.2f}時間"
    )
    if remaining > 0:
        per_video = total_s / len(done)
        print(
            f"\n残り {remaining} 本の推定: "
            f"{per_video/60:.1f} 分/本 × {remaining} = "
            f"{per_video*remaining/3600:.1f} 時間"
        )


if __name__ == "__main__":
    raise SystemExit(main())
