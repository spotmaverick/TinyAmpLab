#!/bin/bash
# 夜间总调度：一次性投递全部实验，Slurm 自动排队，早上统一回收。
# 用法（在 zbw 上执行）: bash night_dispatch.sh [max_concurrent]
#   max_concurrent 默认 8，即最多 8 个同时 RUNNING，其余排队。
#
# 设计原则：
#   1. 全部提交给 Slurm，不自己写并发控制 —— Slurm 的调度比手工可靠
#   2. 每个作业独立日志，失败不影响其他
#   3. 作业名带 dataset 便于 squeue 一眼看出跑的是哪个箱
set -u
cd /work/home/spotmaverick/nam_project

MAXC=${1:-8}
N=0
FAILED=0

submit() {
  local V=$1 E=$2 S=$3 D=$4
  local JID
  JID=$(sbatch --parsable --job-name="${V:0:4}_${D}" sbatch_arch2.sh "$V" "$E" "$S" "$D" 2>&1)
  if [[ "$JID" =~ ^[0-9]+$ ]]; then
    echo "  ok   $JID  $V / $D"
    N=$((N + 1))
  else
    echo "  FAIL      $V / $D -> $JID"
    FAILED=$((FAILED + 1))
  fi
}

echo "=== 夜间批次投递开始 $(date '+%F %H:%M:%S') ==="
echo "并发上限由 Slurm 控制；本脚本一次性投完，自动排队"
echo

echo "[1/3] 多管箱泛化 lite × 11 箱（跳过 blackstar 坏数据）"
for d in bigmuff_diy bsht5m englrt50 markv_clean markvx \
         mesa550_burn mesa550_clean mesa550_crunch ts9_s1 ts9_s2 ua1176; do
  submit lite 300 7 "$d"
done

echo
echo "[2/3] Marshall 四档规模补齐"
submit feather 300 7 marshall_jvm410h_chod1
submit nano    300 7 marshall_jvm410h_chod1
submit feather 300 7 marshall_jvm410h_chod1_spk
submit nano    300 7 marshall_jvm410h_chod1_spk

echo
echo "[3/3] 多管箱 standard 补充（头部 4 个箱，验证规模效应可迁移）"
for d in markv_clean ts9_s1 bigmuff_diy ua1176; do
  submit standard 300 7 "$d"
done

echo
echo "=== 投递完成：成功 $N，失败 $FAILED ==="
echo "当前队列:"
squeue -u spotmaverick -o '%.10i %.22j %.9T %.10M' | head -25
echo
echo "监控: python night_watch.py   回收: python night_reclaim.py"
