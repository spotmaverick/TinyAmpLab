#!/bin/bash
# bw 单卡批量：补跑 mesa550_clean(lite) + 4 个 standard（2026-11-26）
# 用途：补齐 12 箱 lite 矩阵、验证 L1/L2/L3 档是否也需要大模型
# 协议与 z100 夜间批次完全一致：300 ep, seed 7, 同数据集
set -u
cd "$(dirname "$0")" || exit 1
export LD_LIBRARY_PATH=/opt/dtk-26.04/.hyhal/rocm_smi/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=8
mkdir -p arch_v2
LOG=batch_bw.log
echo "[$(date '+%m-%d %H:%M:%S')] ===== BATCH START (5 jobs) =====" >> "$LOG"

run() {
  local variant=$1 ds=$2 seed=7 stem="${1}_${2}_s7"
  local t0
  t0=$(date +%s)
  echo "[$(date '+%m-%d %H:%M:%S')] START $stem" >> "$LOG"
  python3 arch_search_v2.py "$variant" 300 "$seed" "$ds" > "arch_v2/$stem.log" 2>&1
  local rc=$?
  echo "[$(date '+%m-%d %H:%M:%S')] END   $stem rc=$rc $(( $(date +%s) - t0 ))s" >> "$LOG"
}

run lite mesa550_clean &
run standard markv_clean &
run standard ts9_s1 &
run standard bigmuff_diy &
run standard ua1176 &
wait
echo "[$(date '+%m-%d %H:%M:%S')] ===== ALL DONE =====" >> "$LOG"
