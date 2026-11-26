#!/bin/bash
# bw 第七轮：给关键结论补种子（2026-11-26 13:30）
#
# 起因：第六轮用 seed 11 复现 englrt50，把 seed 7 的"lite 拐点"推翻。
# 这暴露出大部分结论都只有单种子（seed 7）。本轮给最关键的几条补 seed 11/23。
#
# 优先级（按结论重要性）：
#   P1  规模收益 vs 难度（r=-0.753，基于单种子）—— 补 2 个代表箱的 standard
#       ts9_s2（易，lite 0.0177 / std 0.0173，差值仅 2.2% 在噪声内）
#       mesa550_burn（最难，lite 0.1218 / std 0.0943，差 22.6% 应稳健，补种子做对照）
#   P2  RF 收益随难度递减（5 点表，全 seed 7）—— 补最短的 ts9_s1 的 rf200
#       ts9_s1 rf200 seed 7 = 0.0115 vs rf100 0.0142（-18.5%，幅度大应稳健，但仍需验证）
#
# 并发：严格 2 路（bw 目前负载高，宁可慢也不要崩）
# 全部 48k batch，与已有 seed 7 结果可直接对比
set -u
cd "$(dirname "$0")" || exit 1
export LD_LIBRARY_PATH=/opt/dtk-26.04/.hyhal/rocm_smi/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=8
export NAMBATCH=48000
mkdir -p arch_v2
LOG=batch_bw_round7.log
echo "[$(date '+%m-%d %H:%M:%S')] ===== ROUND7 START (2-way, seed 11/23) =====" >> "$LOG"

run() {
  local variant=$1 ds=$2 seed=$3 tag=$4 rc t0
  t0=$(date +%s)
  echo "[$(date '+%m-%d %H:%M:%S')] START ${tag}" >> "$LOG"
  python3 arch_search_v2.py "$variant" 300 "$seed" "$ds" > "arch_v2/$tag.log" 2>&1
  rc=$?
  echo "[$(date '+%m-%d %H:%M:%S')] END   ${tag} rc=$rc $(( $(date +%s) - t0 ))s" >> "$LOG"
}

# P1: 规模收益两端各补一个种子
run standard ts9_s2        11 r7_std_ts9s2_s11   &
run standard mesa550_burn  11 r7_std_burn_s11    &
wait

# P2: RF 规律最短点补种子
run lite_rf200 ts9_s1 11 r7_rf200_ts9s1_s11 &
run lite       ts9_s1 11 r7_lite_ts9s1_s11  &
wait

echo "[$(date '+%m-%d %H:%M:%S')] ===== ROUND7 ALL DONE =====" >> "$LOG"
