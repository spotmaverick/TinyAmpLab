#!/bin/bash
# 三机 13 种子训练进度全景查询（本地运行，只读不写远端）
# 用法: bash scripts/p0a_status.sh
set -u
DCU=~/.workbuddy/skills/sugon-dcu-fleet/scripts/dcu.sh

echo "================ P0a 13 种子训练进度 ================"
date "+%Y-%m-%d %H:%M:%S"
echo

echo "--- bw（2 路，15 核独占）---"
$DCU -m bw ssh '
for s in 47 53; do
  L=/workspace/TinyAmpLab/p0a/logs/s${s}.log
  if [ -f "$L" ]; then
    e=$(tr "\r" "\n" < "$L" | grep "55/55" | tail -1 | grep -oE "Epoch [0-9]+" | awk "{print \$2}")
    r=$(tr "\r" "\n" < "$L" | grep "55/55" | tail -1 | grep -oE "[0-9.]+it/s" | tail -1)
    echo "  s${s}: epoch ${e:-?}/300  ${r:-?}"
  else echo "  s${s}: 无日志"; fi
done
echo -n "  nam: "; ls /workspace/TinyAmpLab/p0a/seeds/*.nam 2>/dev/null | xargs -n1 basename 2>/dev/null | tr "\n" " "; echo
echo -n "  监控: "; tail -1 /workspace/TinyAmpLab/p0a/logs/monitor.log 2>/dev/null
' 2>&1 | grep -v "^>>"

echo
echo "--- bw2（1 路，GrpJobs=1 配额）---"
$DCU -m bw2 ssh '
e=$(grep -oE "Epoch [0-9]+" /public/home/spotmaverick/p0a_logs/s43.log 2>/dev/null | tail -1 | awk "{print \$2}")
r=$(grep -oE "[0-9.]+it/s" /public/home/spotmaverick/p0a_logs/s43.log 2>/dev/null | tail -1)
echo "  s43: epoch ${e:-?}/300  ${r:-?}"
echo -n "  队列: "; squeue -h -u spotmaverick -o "%j(%T) " 2>/dev/null; echo
' 2>&1 | grep -v "^>>"

echo
echo "--- z100（4 路并行）---"
$DCU -m z100 ssh '
for s in 59 61 67 71; do
  L=/work/home/spotmaverick/nam_project/p0a_logs/s${s}.log
  e=$(grep -oE "Epoch [0-9]+" "$L" 2>/dev/null | tail -1 | awk "{print \$2}")
  r=$(grep -oE "[0-9.]+it/s" "$L" 2>/dev/null | tail -1)
  echo "  s${s}: epoch ${e:-?}/300  ${r:-?}"
done
echo -n "  nam: "; ls /work/home/spotmaverick/nam_project/p0a_seeds/*.nam 2>/dev/null | xargs -n1 basename 2>/dev/null | tr "\n" " "; echo
' 2>&1 | grep -v "^>>"

echo
echo "--- 本地已回收 ---"
ls reclaimed/ 2>/dev/null | tr "\n" " "; echo
