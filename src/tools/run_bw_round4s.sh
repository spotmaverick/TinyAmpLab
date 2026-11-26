#!/bin/bash
# bw 第四轮（重做）：batch 轴向扫描 —— 完全串行
#
# 背景：并发会触发间歇性 KERNEL VMFault，且与「batch × 数据集」组合有关，
#       难以预测。单进程实测 24k/32k/48k/64k/96k 全部 PASS。
#       为保证产出，本脚本**完全串行**，一次只跑一个。
# 代价：每作业约 10~25 min（96k 约 10 min，24k 约 40 min），共 4 个约 1.5~2 h。
#       但完全避免崩掉重跑，总体更划算。
set -u
cd "$(dirname "$0")" || exit 1
export LD_LIBRARY_PATH=/opt/dtk-26.04/.hyhal/rocm_smi/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=8
mkdir -p arch_v2
LOG=batch_bw_round4s.log
echo "[$(date '+%m-%d %H:%M:%S')] ===== ROUND4S START (fully serial) =====" >> "$LOG"

run() {
  local variant=$1 ds=$2 nb=$3 tag=$4 seed=7 rc t0
  t0=$(date +%s)
  echo "[$(date '+%m-%d %H:%M:%S')] START ${tag} (${ds}/${nb})" >> "$LOG"
  NAMBATCH=$nb python3 arch_search_v2.py "$variant" 300 "$seed" "$ds" > "arch_v2/$tag.log" 2>&1
  rc=$?
  echo "[$(date '+%m-%d %H:%M:%S')] END   ${tag} rc=$rc $(( $(date +%s) - t0 ))s" >> "$LOG"
}

# 先跑 96k（快，用来确认内插点趋势）
run lite mesa550_burn 96000 lite_b96k_burn
run lite markv_clean  96000 lite_b96k_markvc
# 再跑 24k（慢，判断是否继续改善）
run lite mesa550_burn 24000 lite_b24k_burn
run lite markv_clean  24000 lite_b24k_markvc

echo "[$(date '+%m-%d %H:%M:%S')] ===== ROUND4S ALL DONE =====" >> "$LOG"
