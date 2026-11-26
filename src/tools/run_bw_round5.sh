#!/bin/bash
# bw 第五轮：两个高价值方向（2026-11-26 12:20）
# batch 轴向已按用户决定放弃（边际价值低）。改做：
#
# 方向 A — 规模收益的下限在哪（nano/feather 在中等难度箱）
#   现有规模数据几乎都在「易箱(Marshall/tsa15)」或「standard 上限」。
#   未知：中等难度箱(englrt50 0.0715 / markvx 0.0830)用极小模型会差多少？
#   若 nano(841) 就已可用 → 边缘设备可用更小模型；若差距大 → 容量是硬门槛。
#   设计：nano + feather × englrt50 + markvx，配 lite/standard 已有值构成完整阶梯
#
# 方向 B — 容量与视野能否叠加（standard_rf200）
#   已确立：困难箱吃参数(容量)不吃 RF，容易箱吃 RF 不吃参数。
#   未知：对最难的 burn，把「大容量 + 大视野」同时给，能否突破 0.094 的平台？
#   设计：standard_rf200 × mesa550_burn / markvx（一个最难一个中难）
#
# 并发：严格 2 路（实测并发越高越易 VMFault，2 路最稳）
set -u
cd "$(dirname "$0")" || exit 1
export LD_LIBRARY_PATH=/opt/dtk-26.04/.hyhal/rocm_smi/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=8
mkdir -p arch_v2
LOG=batch_bw_round5.log
echo "[$(date '+%m-%d %H:%M:%S')] ===== ROUND5 START (2-way) =====" >> "$LOG"

run() {
  local variant=$1 ds=$2 tag=$3 seed=7 rc t0
  t0=$(date +%s)
  echo "[$(date '+%m-%d %H:%M:%S')] START ${tag}" >> "$LOG"
  python3 arch_search_v2.py "$variant" 300 "$seed" "$ds" > "arch_v2/$tag.log" 2>&1
  rc=$?
  echo "[$(date '+%m-%d %H:%M:%S')] END   ${tag} rc=$rc $(( $(date +%s) - t0 ))s" >> "$LOG"
}

# 方向 A：规模下限（nano 841 / feather 3025）
run nano    englrt50 r5_nano_engl    &
run feather englrt50 r5_feather_engl &
wait
run nano    markvx   r5_nano_markvx    &
run feather markvx   r5_feather_markvx &
wait

# 方向 B：容量+视野叠加
run standard_rf200 mesa550_burn r5_std_rf200_burn  &
run standard_rf200 markvx       r5_std_rf200_markvx &
wait

echo "[$(date '+%m-%d %H:%M:%S')] ===== ROUND5 ALL DONE =====" >> "$LOG"
