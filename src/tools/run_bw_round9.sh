#!/bin/bash
# bw 第九轮：用 192k 重跑被 48k "降级" 的关键实验（2026-11-26 14:10）
#
# 依据（实测，非推测）：
#   1. 192k 精度显著优于 48k：
#        englrt50 standard  s7: 192k 0.0650 vs 48k 0.0825  (-21%)
#        mesa550_burn std   s7: 192k 0.0801 vs 48k 0.0939  (-15%)
#      → 前几轮为规避 VMFault 把 batch 降到 48k，代价是精度和速度双输。
#   2. 192k 速度远快于 48k（真实数据实测）：
#        B=48k  6611 ms/ep (152 step)
#        B=192k 1627 ms/ep ( 38 step)   → 4.1x
#        B=768k  936 ms/ep            → 7.1x
#      显存占用极小：192k 仅 492MB / 64GB，768k 1.7GB。
#   3. 坑18的 VMFault 已在后续推翻：batch 本身无辜，真因是并发×组合的 HIP 间歇故障。
#      本轮严格串行，规避并发风险。
#
# 本机是 1 卡容器，串行 = 独占，不会浪费。
set -u
cd "$(dirname "$0")" || exit 1
export LD_LIBRARY_PATH=/opt/dtk-26.04/.hyhal/rocm_smi/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=8
export NAMBATCH=192000
mkdir -p arch_v2
LOG=batch_bw_round9.log
echo "[$(date '+%m-%d %H:%M:%S')] ===== ROUND9 START (192k 重跑) =====" >> "$LOG"

run() {
  local variant=$1 ds=$2 seed=$3 tag=$4 rc t0
  t0=$(date +%s)
  echo "[$(date '+%m-%d %H:%M:%S')] START ${tag} (${variant}/${ds}/s${seed})" >> "$LOG"
  python3 arch_search_v2.py "$variant" 300 "$seed" "$ds" > "arch_v2/$tag.log" 2>&1
  rc=$?
  echo "[$(date '+%m-%d %H:%M:%S')] END   ${tag} rc=$rc $(( $(date +%s) - t0 ))s" >> "$LOG"
}

# P1: 把 s11 从 48k 提升到 192k，与已有 192k s7 配对 → 真正干净的种子对比
run lite        englrt50  11 r9_lite_engl_s11      # s7 有 192k (0.0774)
run standard    englrt50  11 r9_std_engl_s11       # s7 有 192k (0.0650)
run standard    mesa550_burn 11 r9_std_burn_s11    # s7 有 192k (0.0801)
run standard    ts9_s2    11 r9_std_ts9s2_s11      # s7 有 192k (0.0169)

# P2: 补齐 192k 下缺失的 lite/standard 配对（规模结论的核心证据，目前多为单点）
run lite        mesa550_burn 7 r9_lite_burn_s7      # 已有 0.0921 但需确认 batch
run lite        englrt50  7 r9_lite_engl_s7        # 已有 0.0774 但需确认 batch
run lite        ts9_s2    7 r9_lite_ts9s2_s7       # 缺：ts9_s2 无 lite
run standard    markv_clean 11 r9_std_markvcl_s11  # 最易箱补种子
run lite        mesa550_clean 7 r9_lite_mesacl_s7  # 缺 lite

echo "[$(date '+%m-%d %H:%M:%S')] ===== ROUND9 ALL DONE =====" >> "$LOG"
