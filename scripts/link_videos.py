"""撮影済み動画への symlink を data/videos/ 以下に張る。

元データは Downloads 配下に日付ごとのフォルダで置かれており、1本あたり
約650 MB・全32本で26 GB ある。リポジトリに複製すると容量を食うので、
`data/videos/{date}/{camera}.mp4` という扱いやすい名前の symlink だけを作る。

    python scripts/link_videos.py [SRC_ROOT]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

DEFAULT_SRC_ROOT = Path.home() / "Downloads" / "情報シンポ2026" / "シナリオ"
DEST_ROOT = Path(__file__).resolve().parent.parent / "data" / "videos"

# 例: "0720_cameraA.mp4" -> date="0720", camera="camA"
NAME_RE = re.compile(r"^(?P<date>\d{4})_camera(?P<cam>[A-Z])\.mp4$")


def main(argv: list[str]) -> int:
    src_root = Path(argv[1]).expanduser() if len(argv) > 1 else DEFAULT_SRC_ROOT
    if not src_root.is_dir():
        print(f"source root not found: {src_root}", file=sys.stderr)
        return 1

    n_linked = 0
    for src in sorted(src_root.glob("*/*_時刻同期/*.mp4")):
        m = NAME_RE.match(src.name)
        if m is None:
            print(f"skip (unexpected name): {src.name}")
            continue

        dest = DEST_ROOT / m["date"] / f"cam{m['cam']}.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        dest.symlink_to(src)
        n_linked += 1
        print(f"{dest.relative_to(DEST_ROOT.parent.parent)} -> {src}")

    print(f"\nlinked {n_linked} videos into {DEST_ROOT}")
    return 0 if n_linked else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
