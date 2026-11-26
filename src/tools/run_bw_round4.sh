#!/bin/bash
# bw 第四轮：batch 轴向扫描（2026-11-26 14:00）
#
# 前一轮因并发超限（8 路）导致 24k/96k 全崩，本轮严格限制 4 并发。
#
# 目的：确认 batch 是否存在最优点。
# 已知（rf100，300ep）：
#   mesa550_burn : 192k=0.1218  48k=0.0959
#   markv_clean  : 192k=0.0245  48k=0.0173
# 单调向小 batch 改善。但「更小是否继续改善」未知：
#   - 若 24k 继续改善 -> 仍是优化步数不足，可再压
#   - 若 24k 变差   -> 存在最优 batch（太小则梯度噪声主导）
# 96k 作为内插点，用来判断是否为单调曲线。
set -u
cd "$(dirname "$0")" || exit 1
export LD_LIBRARY_PATH=/opt/dtk-26.04/.hyhal/rocm_smi/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=8
mkdir -p arch_v2
LOG=batch_bw_round4.log
echo "[$(date '+%m-%d %H:%M:%S')] ===== ROUND4 START (4 jobs, 限并发) =====" >> "$LOG"

run() {
  local variant=$1 ds=$2 nb=$3 tag=$4 seed=7
  local t0
  t0=$(date +%s)
  echo "[$(date '+%m-%d %H:%M:%S')] START ${tag} (${variant}/${ds}/${nb})" >> "$LOG"
  NAMBATCH=$nb python3 arch_search_v2.py "$variant" 300 "$seed" "$ds" > "arch_v2/$tag.log" 2>&1
  echo "[$(date '+%m-%d %H:%M:%S')] END   ${tag} rc=$? $(( $(date +%s) - t0 ))s" >> "$LOG"
}

run lite mesa550_burn 96000 lite_b96k_burn     &
run lite mesa550_burn 24000 lite_b24k_burn     &
run lite markv_clean  96000 lite_b96k_markvc   &
run lite markv_clean  24000 lite_b24k_markvc   &

wait
echo "[$(date '+%m-%d %H:%M:%S')] ===== ROUND4 ALL DONE =====" >> "$LOG"
