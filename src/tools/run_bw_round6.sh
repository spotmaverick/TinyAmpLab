#!/bin/bash
# bw 第六轮：验证拐点 + 探 burn 的真瓶颈（2026-11-26 12:05）
#
# 依据第五轮结论：
#   6.5  englrt50 在 lite(6553) 见底，standard 反而 +3.4% -> 是噪声还是真拐点？
#   6.6  容量+视野叠加无增益 -> burn 的瓶颈不是结构维度
#
# 方向 A — 拐点是否真实（种子复现）
#   englrt50 的 lite(0.0715) vs standard(0.0739) 只差 3.4%，接近噪声地板。
#   用 seed 11 复跑这两档：若 lite 仍胜 -> 拐点真实；若反转 -> 属噪声。
#
# 方向 B — burn 的真瓶颈：训练策略而非结构
#   burn 在 48k batch 下 0.0959，容量/视野都推不动。换两个非结构杠杆：
#   B1 更长训练：600 ep（当前 300ep 可能未收敛到底）
#   B2 更强正则/不同 lr：lr 1e-3（当前 3e-3，噪声大时更稳）
#   注意：B1 用 600ep，耗时约 2 倍。
#
# 并发：严格 2 路
set -u
cd "$(dirname "$0")" || exit 1
export LD_LIBRARY_PATH=/opt/dtk-26.04/.hyhal/rocm_smi/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=8
mkdir -p arch_v2
LOG=batch_bw_round6.log
echo "[$(date '+%m-%d %H:%M:%S')] ===== ROUND6 START (2-way) =====" >> "$LOG"

run() {
  local variant=$1 ds=$2 seed=$3 ep=$4 tag=$5 rc t0
  t0=$(date +%s)
  echo "[$(date '+%m-%d %H:%M:%S')] START ${tag} (ep=${ep} seed=${seed})" >> "$LOG"
  python3 arch_search_v2.py "$variant" "$ep" "$seed" "$ds" > "arch_v2/$tag.log" 2>&1
  rc=$?
  echo "[$(date '+%m-%d %H:%M:%S')] END   ${tag} rc=$rc $(( $(date +%s) - t0 ))s" >> "$LOG"
}

export NAMBATCH=48000

# 方向 A：englrt50 拐点复现（seed 11）
run lite     englrt50 11 300 r6_lite_engl_s11     &
run standard englrt50 11 300 r6_std_engl_s11      &
wait

# 方向 B：burn 的可训练性（600 ep，看是否只是没训够）
run lite mesa550_burn 7 600 r6_lite_burn_600ep   &
run lite markv_clean  7 600 r6_lite_markvc_600ep &
wait

echo "[$(date '+%m-%d %H:%M:%S')] ===== ROUND6 ALL DONE =====" >> "$LOG"
