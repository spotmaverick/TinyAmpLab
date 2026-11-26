#!/bin/bash
# bw 第二轮：补 7 箱 standard + mesa550_burn 的 RF 加倍实验（2026-11-26）
#
# 目的：
#   A. 补齐 standard 矩阵单点空白（当前 13 箱里 7 箱只有 lite）
#   B. 验证 burn "记忆长度不足" 假说 —— 该箱 lite val 0.1218 是全库最难，
#      若 RF 加倍后显著改善，则说明瓶颈是感受野而非参数量
#
# 协议与夜间批次完全一致：300 ep / seed 7 / MSE+batch192000+lr3e-3+cosine
# RF 说明：variant 里 _rf<N> 表示 dilation 缩放到 N%。rf100=原设计，rf200=加倍。
#          lite 原 RF=4093(85.3ms) -> rf200 约 8186(170.6ms)
set -u
cd "$(dirname "$0")" || exit 1
export LD_LIBRARY_PATH=/opt/dtk-26.04/.hyhal/rocm_smi/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=8
mkdir -p arch_v2
LOG=batch_bw_round2.log
echo "[$(date '+%m-%d %H:%M:%S')] ===== ROUND2 START (9 jobs) =====" >> "$LOG"

run() {
  local variant=$1 ds=$2 seed=7 stem="${1}_${2}_s7"
  local t0
  t0=$(date +%s)
  echo "[$(date '+%m-%d %H:%M:%S')] START $stem" >> "$LOG"
  python3 arch_search_v2.py "$variant" 300 "$seed" "$ds" > "arch_v2/$stem.log" 2>&1
  local rc=$?
  echo "[$(date '+%m-%d %H:%M:%S')] END   $stem rc=$rc $(( $(date +%s) - t0 ))s" >> "$LOG"
}

# A 组：补 standard 单点空白（7 个箱）
run standard ts9_s2 &
run standard bsht5m &
run standard markvx &
run standard mesa550_crunch &
run standard englrt50 &
run standard mesa550_clean &
run standard mesa550_burn &

# B 组：burn 的 RF 加倍实验（最难箱，验证记忆长度假说）
run lite_rf200 mesa550_burn &
run lite_rf150 mesa550_burn &

wait
echo "[$(date '+%m-%d %H:%M:%S')] ===== ROUND2 ALL DONE =====" >> "$LOG"
