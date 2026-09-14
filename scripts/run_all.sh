#!/usr/bin/env bash
# 指定した日付の複数カメラについて、track と4種類の可視化を順に実行する。
#
# 1台あたり20分強かかるため、途中でスリープしないよう caffeinate で抑止し、
# 1台が失敗しても残りを止めずに続け、最後にまとめて報告する。
#
#   scripts/run_all.sh 0720 camB camC camD camE camF camG camH
#   scripts/run_all.sh 0727            # カメラ省略時は camA..camH
#
# ログ     : logs/{date}_{camera}.log
# 集計     : logs/{date}_summary.tsv（camera / status / 各所要秒数）

set -uo pipefail        # 1台の失敗で止めないため -e は付けない

if [ $# -lt 1 ]; then
    echo "usage: $0 DATE [CAMERA ...]" >&2
    exit 1
fi

DATE="$1"
shift
CAMERAS=("$@")
if [ ${#CAMERAS[@]} -eq 0 ]; then
    CAMERAS=(camA camB camC camD camE camF camG camH)
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -x .venv/bin/python ]; then
    PYTHON=.venv/bin/python
else
    PYTHON="${PYTHON:-python3}"
fi

OUT="${OUT:-outputs}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/${DATE}_summary.tsv"

# このスクリプトが生きている間だけシステムのスリープを抑止する
caffeinate -i -w $$ &

printf 'camera\tstatus\ttrack_sec\tviz_sec\n' > "$SUMMARY"

FAILED=()
TOTAL_START=$(date +%s)

for CAMERA in "${CAMERAS[@]}"; do
    LOG="$LOG_DIR/${DATE}_${CAMERA}.log"
    VIDEO="data/videos/$DATE/$CAMERA.mp4"

    echo "=============================================="
    echo " $DATE / $CAMERA   ($(date '+%H:%M:%S'))"
    echo "=============================================="

    if [ ! -e "$VIDEO" ]; then
        echo "!! 動画が見つかりません: $VIDEO"
        printf '%s\tno_video\t\t\n' "$CAMERA" >> "$SUMMARY"
        FAILED+=("$CAMERA (動画なし)")
        continue
    fi

    : > "$LOG"

    t0=$(date +%s)
    "$PYTHON" -m mct.cli track --date "$DATE" --camera "$CAMERA" --out "$OUT" >> "$LOG" 2>&1
    track_rc=$?
    t1=$(date +%s)

    if [ $track_rc -ne 0 ]; then
        echo "!! track が失敗しました (exit $track_rc) — $LOG の末尾:"
        tail -n 15 "$LOG"
        printf '%s\ttrack_failed\t%d\t\n' "$CAMERA" "$((t1 - t0))" >> "$SUMMARY"
        FAILED+=("$CAMERA (track exit $track_rc)")
        continue
    fi

    for CMD in viz-video viz-trajectory viz-density viz-endpoints; do
        echo "--- $CMD ---" >> "$LOG"
        "$PYTHON" -m mct.cli "$CMD" --date "$DATE" --camera "$CAMERA" --out "$OUT" >> "$LOG" 2>&1 \
            || FAILED+=("$CAMERA ($CMD)")
    done
    t2=$(date +%s)

    # 1台終わるごとの報告
    echo
    sed -n '/^====/,/^====/p' "$LOG" | head -n 20
    grep -E '出力フレーム|除外' "$LOG" || true
    echo "  track $((t1 - t0)) 秒 / viz $((t2 - t1)) 秒 / 計 $((t2 - t0)) 秒"
    echo

    printf '%s\tok\t%d\t%d\n' "$CAMERA" "$((t1 - t0))" "$((t2 - t1))" >> "$SUMMARY"
done

TOTAL=$(($(date +%s) - TOTAL_START))
echo "=============================================="
echo " 完了: ${#CAMERAS[@]} 台, 合計 $((TOTAL / 60)) 分 $((TOTAL % 60)) 秒"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo " 失敗なし"
else
    echo " 失敗 ${#FAILED[@]} 件:"
    printf '   - %s\n' "${FAILED[@]}"
fi
echo " 集計: $SUMMARY"
echo "=============================================="
