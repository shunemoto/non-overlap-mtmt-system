#!/usr/bin/env bash
# 指定した日付・カメラについて4種類の可視化をまとめて生成する。
#
# 観測区間と遷移領域を決める作業では4つの図を見比べるので、毎回4コマンドを
# 打つ手間を省く。追跡（track）はやり直さない。
#
#   scripts/run_viz_all.sh 0720 camA
#   scripts/run_viz_all.sh 0720 camA camB camC     # 複数カメラ
#   OUT=other_outputs scripts/run_viz_all.sh 0720 camA

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 DATE CAMERA [CAMERA ...]" >&2
    echo "  例: $0 0720 camA camB" >&2
    exit 1
fi

DATE="$1"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# .venv があればそれを使い、無ければ PATH 上の python にフォールバックする
if [ -x .venv/bin/python ]; then
    PYTHON=.venv/bin/python
else
    PYTHON="${PYTHON:-python3}"
fi

OUT="${OUT:-outputs}"

for CAMERA in "$@"; do
    echo "=============================================="
    echo " $DATE / $CAMERA"
    echo "=============================================="
    for CMD in viz-video viz-trajectory viz-density viz-endpoints; do
        echo "--- $CMD ---"
        "$PYTHON" -m mct.cli "$CMD" --date "$DATE" --camera "$CAMERA" --out "$OUT"
    done
    echo
    echo "出力: $OUT/$DATE/$CAMERA/viz/"
    ls -lh "$OUT/$DATE/$CAMERA/viz/"
    echo
done
