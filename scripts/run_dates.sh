#!/usr/bin/env bash
# 複数日付・全カメラを通しで処理し、進捗を1行ずつ logs/progress.txt に残す。
#
# 10時間を超える処理を目で見張ることはできないので、別端末から
#
#     tail -f logs/progress.txt
#
# で状況を追えるようにしてある。1台終わるごとに1行追記する。
# 1台が失敗しても FAILED 行を残して次に進み、最後にまとめて報告する。
#
#   scripts/run_dates.sh 0727 0730 0803
#
# 既存の出力は消さない。処理するのは引数に与えた日付の下だけなので、
# 別日付の可視化を見ながら並行して作業できる。

set -uo pipefail        # 1台の失敗で止めないため -e は付けない

if [ $# -lt 1 ]; then
    echo "usage: $0 DATE [DATE ...]" >&2
    exit 1
fi

DATES=("$@")
CAMERAS=(camA camB camC camD camE camF camG camH)

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
PROGRESS="$LOG_DIR/progress.txt"

# 実行中はスリープさせない（このスクリプトが生きている間だけ）
caffeinate -i -w $$ &

now() { date '+%Y-%m-%d %H:%M:%S'; }

# 先頭に処理対象の一覧を書いておく。どこまで進んだかが一目で分かる。
{
    echo "# 開始 $(now)"
    echo "# 対象 $(( ${#DATES[@]} * ${#CAMERAS[@]} )) 本"
    for d in "${DATES[@]}"; do
        for c in "${CAMERAS[@]}"; do
            echo "#   $d/$c"
        done
    done
    echo "#"
} > "$PROGRESS"

TOTAL_START=$(date +%s)
N_OK=0
FAILED=()

for DATE in "${DATES[@]}"; do
    SUMMARY="$LOG_DIR/${DATE}_summary.tsv"
    printf 'camera\tstatus\ttrack_sec\tviz_sec\n' > "$SUMMARY"

    for CAMERA in "${CAMERAS[@]}"; do
        LOG="$LOG_DIR/${DATE}_${CAMERA}.log"
        VIDEO="data/videos/$DATE/$CAMERA.mp4"
        t0=$(date +%s)

        if [ ! -e "$VIDEO" ]; then
            echo "$(now) $DATE/$CAMERA FAILED  reason=no_video" >> "$PROGRESS"
            printf '%s\tno_video\t\t\n' "$CAMERA" >> "$SUMMARY"
            FAILED+=("$DATE/$CAMERA 動画なし")
            continue
        fi

        : > "$LOG"
        "$PYTHON" -m mct.cli track --date "$DATE" --camera "$CAMERA" --out "$OUT" >> "$LOG" 2>&1
        track_rc=$?
        t1=$(date +%s)

        if [ $track_rc -ne 0 ]; then
            echo "$(now) $DATE/$CAMERA FAILED  reason=track_exit_$track_rc elapsed=$(( (t1-t0)/60 ))" >> "$PROGRESS"
            printf '%s\ttrack_failed\t%d\t\n' "$CAMERA" "$((t1 - t0))" >> "$SUMMARY"
            FAILED+=("$DATE/$CAMERA track exit $track_rc")
            continue
        fi

        viz_rc=0
        for CMD in viz-video viz-trajectory viz-density viz-endpoints; do
            echo "--- $CMD ---" >> "$LOG"
            "$PYTHON" -m mct.cli "$CMD" --date "$DATE" --camera "$CAMERA" --out "$OUT" >> "$LOG" 2>&1 || viz_rc=1
        done
        t2=$(date +%s)

        tracks=$(awk -F: '/総トラック数/{gsub(/[^0-9]/,"",$2); print $2}' "$LOG" | tail -1)
        masks=$(awk -F: '/マスク取得失敗/{gsub(/[^0-9]/,"",$2); print $2}' "$LOG" | tail -1)

        if [ $viz_rc -ne 0 ]; then
            echo "$(now) $DATE/$CAMERA FAILED  reason=viz tracks=${tracks:-?} masks_failed=${masks:-?} elapsed=$(( (t2-t0)/60 ))" >> "$PROGRESS"
            FAILED+=("$DATE/$CAMERA 可視化の一部")
        else
            echo "$(now) $DATE/$CAMERA done  tracks=${tracks:-?} masks_failed=${masks:-?} elapsed=$(( (t2-t0)/60 ))" >> "$PROGRESS"
            N_OK=$((N_OK + 1))
        fi
        printf '%s\tok\t%d\t%d\n' "$CAMERA" "$((t1 - t0))" "$((t2 - t1))" >> "$SUMMARY"
    done
done

TOTAL=$(($(date +%s) - TOTAL_START))
{
    echo "#"
    echo "# 終了 $(now)  成功 $N_OK / $(( ${#DATES[@]} * ${#CAMERAS[@]} )) 本  合計 $((TOTAL / 3600)) 時間 $(((TOTAL % 3600) / 60)) 分"
    if [ ${#FAILED[@]} -gt 0 ]; then
        printf '# 失敗: %s\n' "${FAILED[@]}"
    fi
} >> "$PROGRESS"

tail -n 5 "$PROGRESS"
