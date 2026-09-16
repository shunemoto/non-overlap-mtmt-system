#!/usr/bin/env bash
# GT の bbox から外観特徴の元データを作る（再同定単体評価用）。
#
# 4日×8台で1時間半ほどかかるので、途中でスリープしないよう caffeinate で
# 抑止し、1カメラ終わるごとに logs/progress_gt.txt へ追記する。別端末から
#
#     tail -f logs/progress_gt.txt
#
# で追える。1台が失敗しても止めず、最後にまとめて報告する。
#
#   scripts/run_gt_features.sh                 # 全4日
#   scripts/run_gt_features.sh 0720 0727       # 日付を指定

set -uo pipefail        # 1台の失敗で止めないため -e は付けない

DATES=("$@")
if [ ${#DATES[@]} -eq 0 ]; then
    DATES=(0720 0727 0730 0803)
fi
CAMERAS=(camA camB camC camD camE camF camG camH)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -x .venv/bin/python ]; then
    PYTHON=.venv/bin/python
else
    PYTHON="${PYTHON:-python3}"
fi

LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
PROGRESS="$LOG_DIR/progress_gt.txt"

# このスクリプトが生きている間だけスリープを抑止する
caffeinate -i -w $$ &

now() { date '+%Y-%m-%d %H:%M:%S'; }

{
    echo "# 開始 $(now)"
    echo "# 対象 $(( ${#DATES[@]} * ${#CAMERAS[@]} )) カメラ"
} > "$PROGRESS"

TOTAL_START=$(date +%s)
FAILED=()

for DATE in "${DATES[@]}"; do
    for CAMERA in "${CAMERAS[@]}"; do
        LOG="$LOG_DIR/gt_${DATE}_${CAMERA}.log"
        t0=$(date +%s)
        "$PYTHON" -m mct.cli gt-features --date "$DATE" --camera "$CAMERA" > "$LOG" 2>&1
        rc=$?
        t1=$(date +%s)

        if [ $rc -ne 0 ]; then
            echo "$(now) $DATE/$CAMERA FAILED  exit=$rc elapsed=$(( (t1-t0)/60 ))分" >> "$PROGRESS"
            FAILED+=("$DATE/$CAMERA (exit $rc)")
            tail -n 10 "$LOG"
            continue
        fi

        tracks=$(awk -F'->' '/GT トラック/{gsub(/[^0-9]/,"",$2); print $2}' "$LOG" | tail -1)
        crops=$(awk '/crop [0-9]+ 枚/{for(i=1;i<=NF;i++) if($i=="crop") print $(i+1)}' "$LOG" | tail -1)
        nokp=$(awk -F'骨格点なし' '/骨格点なし/{split($2,a," "); print a[1]}' "$LOG" | tail -1)
        nomask=$(awk -F'マスクなし' '/マスクなし/{split($2,a," "); print a[1]}' "$LOG" | tail -1)
        echo "$(now) $DATE/$CAMERA done  tracks=${tracks:-0} crops=${crops:-0}" \
             "no_kp=${nokp:-0} no_mask=${nomask:-0} elapsed=$(( (t1-t0)/60 ))分" >> "$PROGRESS"
    done
done

TOTAL=$(($(date +%s) - TOTAL_START))
{
    echo "#"
    echo "# 終了 $(now)  合計 $((TOTAL / 3600)) 時間 $(((TOTAL % 3600) / 60)) 分"
    if [ ${#FAILED[@]} -gt 0 ]; then
        printf '# 失敗: %s\n' "${FAILED[@]}"
    else
        echo "# 失敗なし"
    fi
} >> "$PROGRESS"

tail -n 5 "$PROGRESS"
