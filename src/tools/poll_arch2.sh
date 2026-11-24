#!/bin/bash
# 轮询 arch_v2 批量作业: 用法 ./poll_arch2.sh [间隔秒] [次数]
KEY=/Users/wangguan/Downloads/spotmaverick_wuzh02.hpccube.com_RsaKeyExpireTime_2027-02-21_21-34-03.txt
HOST=spotmaverick@wuzh02.hpccube.com
PORT=65091
GAP=${1:-900}; MAX=${2:-12}
i=0
while [ "$MAX" -eq 0 ] || [ "$i" -lt "$MAX" ]; do
  i=$((i+1))
  echo "===== poll $i $(date '+%H:%M:%S') ====="
  ssh -n -p $PORT -i "$KEY" -o BatchMode=yes -o ConnectTimeout=20 "$HOST" '
    echo "--- running jobs ---"
    squeue -u spotmaverick -o "%.10i %.8T %.22j %.10M" | head -15
    echo "--- arch_v2 progress ---"
    for f in /work/home/spotmaverick/nam_project/arch_v2/*.log; do
      [ -f "$f" ] && echo "$(basename $f .log): $(tail -1 "$f")"
    done
    echo "--- results ---"
    ls /work/home/spotmaverick/nam_project/arch_v2/*.json 2>/dev/null | wc -l
  '
  if [ "$MAX" -eq 0 ] || [ "$i" -lt "$MAX" ]; then sleep "$GAP"; fi
done
