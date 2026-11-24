#!/bin/bash
# 轮询 DCU 作业与诊断结果: 用法 ./poll_dcu.sh [间隔秒] [次数]; 0 次=无限
KEY=/Users/wangguan/Downloads/spotmaverick_wuzh02.hpccube.com_RsaKeyExpireTime_2027-02-21_21-34-03.txt
HOST=spotmaverick@wuzh02.hpccube.com
PORT=65091
GAP=${1:-120}; MAX=${2:-30}
i=0
while [ "$MAX" -eq 0 ] || [ "$i" -lt "$MAX" ]; do
  i=$((i+1))
  echo "===== poll $i $(date '+%H:%M:%S') ====="
  ssh -n -p $PORT -i "$KEY" -o BatchMode=yes -o ConnectTimeout=20 "$HOST" '
    echo "--- jobs ---"
    squeue -u spotmaverick -o "%.10i %.8T %.22j %.10M" | head -20
    echo "--- diag3 progress ---"
    for f in /work/home/spotmaverick/nam_project/diag3/*.log; do
      [ -f "$f" ] && echo "### $(basename $f): $(tail -1 "$f")"
    done
    echo "--- diag3 results ---"
    ls -lh /work/home/spotmaverick/nam_project/diag3/*.json 2>/dev/null
  '
  if [ "$MAX" -eq 0 ] || [ "$i" -lt "$MAX" ]; then sleep "$GAP"; fi
done
