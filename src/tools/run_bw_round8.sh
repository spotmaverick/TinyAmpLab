#!/bin/bash
# bw 第八轮：补 48k 下的 seed 7，让种子对比成立（2026-11-26 13:40）
#
# 问题：第七轮补的 seed 11 全部在 48k，而 seed 7 的 standard 是 192k，
#       两者不能对比。本轮补齐「48k × seed 7 的 standard」，
#       使每个箱在 48k 下同时有 s7 与 s11，种子方差才可量化。
#
# 同时补 lite 的 48k × s7 缺项（englrt50 lite_s7 已是 48k，无需补）。
#
# 串行 2 路，48k
set -u
cd "$(dirname "$0")" || exit 1
export LD_LIBRARY_PATH=/opt/dtk-26.04/.hyhal/rocm_smi/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=8
export NAMBATCH=48000
mkdir -p arch_v2
LOG=batch_bw_round8.log
echo "[$(date '+%m-%d %H:%M:%S')] ===== ROUND8 START (48k seed7) =====" >> "$LOG"

run() {
  local variant=$1 ds=$2 seed=$3 tag=$4 rc t0
  t0=$(date +%s)
  echo "[$(date '+%m-%d %H:%M:%S')] START ${tag}" >> "$LOG"
  python3 arch_search_v2.py "$variant" 300 "$seed" "$ds" > "arch_v2/$tag.log" 2>&1
  rc=$?
  echo "[$(date '+%m-%d %H:%M:%S')] END   ${tag} rc=$rc $(( $(date +%s) - t0 ))s" >> "$LOG"
}

# 48k × seed 7 的 standard（与已有 s11 配对）
run standard mesa550_burn 7 r8_std_burn_s7   &
run standard englrt50     7 r8_std_engl_s7   &
wait

run standard ts9_s2 7 r8_std_ts9s2_s7 &
wait

echo "[$(date '+%m-%d %H:%M:%S')] ===== ROUND8 ALL DONE =====" >> "$LOG"
