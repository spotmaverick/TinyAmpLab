#!/bin/bash
# bw 第三轮：新轴向实验（2026-11-26 11:00）
#
# 不再复跑已覆盖的配置。两个新方向：
#
# 方向 A — batch/步数轴向扫描（本轮最有价值的发现来自这里）
#   已测：burn 192k=0.1218 / 48k=0.0959；markv_clean 192k=0.0245 / 48k=0.0173
#   小 batch = 每 epoch 更多梯度步，是比 RF 强得多的杠杆（11~30%）
#   未知：继续减小是否还有收益？是否存在最优点？还是要更多「总步数」而非更小 batch？
#   设计：补 96k（内插点）与 24k（继续减半），在 burn（最难，余量最大）与 markv_clean（中等）各跑
#   若 24k 继续改善 -> 单调，可再压；若 24k 变差 -> 存在最优 batch
#
# 方向 B — 扩展「RF 收益随难度递减」规律的样本点
#   已测 3 点：ts9_s1(易) -18.5% / markv_clean(中) -11.3% / mesa550_burn(难) -0.3%
#   只有 3 点不足以定论，补 2 个中高难度箱：englrt50(0.0942) / mesa550_crunch(0.0941)
#   每个箱需 rf100@48k 与 rf200@48k 一对（基线必须同 batch，否则不可比）
#
# 协议：300 ep / seed 7 / 与之前完全一致，仅变 batch 与 RF
set -u
cd "$(dirname "$0")" || exit 1
export LD_LIBRARY_PATH=/opt/dtk-26.04/.hyhal/rocm_smi/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=8
mkdir -p arch_v2
LOG=batch_bw_round3.log
echo "[$(date '+%m-%d %H:%M:%S')] ===== ROUND3 START =====" >> "$LOG"

# run <variant> <dataset> <batch> <logtag>
run() {
  local variant=$1 ds=$2 nb=$3 tag=$4 seed=7
  local t0
  t0=$(date +%s)
  echo "[$(date '+%m-%d %H:%M:%S')] START ${variant}/${ds}/${nb} $tag" >> "$LOG"
  NAMBATCH=$nb python3 arch_search_v2.py "$variant" 300 "$seed" "$ds" > "arch_v2/$tag.log" 2>&1
  local rc=$?
  echo "[$(date '+%m-%d %H:%M:%S')] END   ${variant}/${ds}/${nb} $tag rc=$rc $(( $(date +%s) - t0 ))s" >> "$LOG"
}

# ---- 方向 A：batch 轴向 ----
run lite mesa550_burn  96000 lite_b96k_burn     &
run lite mesa550_burn  24000 lite_b24k_burn     &
run lite markv_clean   96000 lite_b96k_markvc   &
run lite markv_clean   24000 lite_b24k_markvc   &

# ---- 方向 B：RF 规律扩样（每箱一对，同 48k）----
run lite englrt50       48000 lite_b48k_engl     &
run lite_rf200 englrt50 48000 lite_rf200_b48k_engl &
run lite mesa550_crunch 48000 lite_b48k_crunch   &
run lite_rf200 mesa550_crunch 48000 lite_rf200_b48k_crunch &

wait
echo "[$(date '+%m-%d %H:%M:%S')] ===== ROUND3 ALL DONE =====" >> "$LOG"
