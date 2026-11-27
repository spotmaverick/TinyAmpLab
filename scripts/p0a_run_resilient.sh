#!/bin/bash
# p0a 单种子训练启动器（带 checkpoint 自愈重试）
#
# 背景：官方 nam 的 full.main 本来就在写 checkpoint（checkpoint_epoch_*.ckpt 每 epoch 覆盖一份），
#       含 optimizer_states / lr_schedulers / epoch / global_step，不是空壳。
#       缺的只是"续训入口"，train_seeded.py 已补上 --resume-from。
#
# 本脚本的作用：进程/节点异常退出时，不从头重跑，而是从最近 checkpoint 续训。
# 这是 bw2 坏节点（a02r3n12 缺 libhydmi.so 静默 CPU 降级）那次教训的直接产物——
# 当时是整颗种子作废，有了本脚本就只损失异常发生前一个 epoch 的进度。
#
# 用法：
#   bash p0a_run_resilient.sh <seed> [learn_cfg] [max_retries]
#
# 环境变量：
#   P0A_ROOT   项目根（默认 /workspace/TinyAmpLab）
#   P0A_CFGDIR 配置目录（默认 $P0A_ROOT/p0a/cfg）
set -u

# `set -u` 下不能直接展开未定义变量（`${2:-$P0A_ROOT/...}` 会在 P0A_ROOT 未设时直接退出）。
# 必须先给默认值，再引用。
ROOT=${P0A_ROOT:-/workspace/TinyAmpLab}
CFG=${P0A_CFGDIR:-$ROOT/p0a/cfg}

S=${1:?seed required}
LC=${2:-$CFG/learn_300ep.json}
MAXRET=${3:-3}

cd "$ROOT" || exit 1

export LD_LIBRARY_PATH=/opt/dtk-26.04/.hyhal/rocm_smi/lib:${LD_LIBRARY_PATH:-}

LOG="p0a/logs/s${S}.log"
RUN="p0a/runs/p0a_s${S}"

# 加速卡硬断言：宁可快速失败，也不要静默降级到 CPU 跑几小时
assert_accel() {
  python3 -c 'import torch; assert torch.cuda.is_available(), "NO DCU"'
}

# 找最近一次 checkpoint（按 epoch 号排序，而不是 mtime——续训会产生新 version 目录）
latest_ckpt() {
  find "$RUN" -name 'checkpoint_epoch_*.ckpt' 2>/dev/null \
    | sed 's/.*epoch=0*\([0-9]*\)\.ckpt/\1\t&/' \
    | sort -rn | head -1 | cut -f2-
}

launch_fresh() {
  python3 -u "$CFG/train_seeded.py" \
    "$CFG/data_tsa15_bw.json" "$CFG/model_L10-ch32-bigRF.json" "$LC" \
    "$RUN" "$S" --no-show --no-plots
}

launch_resume() {
  local ck=$1
  python3 -u "$CFG/train_seeded.py" \
    "$CFG/data_tsa15_bw.json" "$CFG/model_L10-ch32-bigRF.json" "$LC" \
    --resume-from "$ck" "$S" --max-epochs 300 --no-show --no-plots
}

echo "=== [$(date +%F_%T)] seed=$S start (max_retries=$MAXRET) ===" >>"$LOG"

# 单例保护：同一 run 目录只允许一个训练主进程。
# 判据不能用 `pgrep -f train_seeded`——dataloader worker（num_workers=4）的 cmdline
# 与主进程逐字相同，会被误判成"多开"（这个坑我踩过：把 worker 当成重复训练进程）。
# 正确判据：cmdline 匹配本 run 目录的进程中，ppid 不在该集合里的那个才是主进程。
running_owner() {
  python3 - "$S" <<'PY'
import os, sys, subprocess
seed = sys.argv[1]
tag = f"runs/p0a_s{seed}"
out = subprocess.run(["ps", "-eo", "pid,ppid,args"], capture_output=True, text=True).stdout
rows = []
for line in out.splitlines()[1:]:
    parts = line.split(None, 2)
    if len(parts) != 3:
        continue
    pid, ppid, args = parts
    if tag in args and "train_seeded" in args:
        rows.append((int(pid), int(ppid), args))
pids = {r[0] for r in rows}
for pid, ppid, args in rows:
    if ppid not in pids:      # 父进程不在本组 -> 这是主进程（worker 的 ppid 一定在组内）
        print(pid)
PY
}

OWNER=$(running_owner)
if [ -n "$OWNER" ]; then
  echo "ALREADY_RUNNING_s${S} pid=$OWNER -> skip（单例保护）" >>"$LOG"
  exit 0
fi

if ! assert_accel >>"$LOG" 2>&1; then
  echo "ACCEL_FAIL_NO_DCU$(date +%F_%T)" >>"$LOG"
  exit 9
fi

attempt=0
while : ; do
  attempt=$((attempt + 1))
  CK=$(latest_ckpt)

  if [ -n "${CK:-}" ]; then
    EPOCH=$(echo "$CK" | sed 's/.*epoch=0*\([0-9]*\)\.ckpt/\1/')
    echo "=== [$(date +%F_%T)] attempt#$attempt RESUME from ep$EPOCH ===" >>"$LOG"
    launch_resume "$CK" >>"$LOG" 2>&1
  else
    echo "=== [$(date +%F_%T)] attempt#$attempt FRESH ===" >>"$LOG"
    launch_fresh >>"$LOG" 2>&1
  fi
  rc=$?

  if [ $rc -eq 0 ]; then
    NAM=$(find "$RUN" -name 'model.nam' | head -1)
    if [ -n "$NAM" ]; then
      cp "$NAM" "p0a/seeds/s${S}.nam"
      echo "DONE_s${S}_attempt${attempt}" >>"$LOG"
      exit 0
    fi
    echo "NO_NAM_AFTER_SUCCESS_s${S}" >>"$LOG"
    rc=1
  fi

  if [ $attempt -ge $MAXRET ]; then
    echo "FAIL_s${S}_rc${rc}_attempts${attempt}" >>"$LOG"
    exit $rc
  fi

  echo "=== [$(date +%F_%T)] attempt#$attempt failed rc=$rc, retry in 20s ===" >>"$LOG"
  sleep 20
done
