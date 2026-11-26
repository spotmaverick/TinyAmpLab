#!/bin/bash
# z100 批量投递：小计算量 / 多路并行（2026-11-26 14:20）
#
# 分配原则（用户指定）：计算相对小、ep 少、需并行多个 → z100（8 卡）
#   bw 用于大计算量、ep 多、串行独占
#
# z100 环境事实：
#   DTK 22.04 + python 3.8.10 + torch 1.10.0a0+git450cdd1-dtk22.04
#   必须 source /work/home/spotmaverick/nam_probe/run_env.sh
#   Slurm 分区 wzhdtest，每节点 4 卡，--gres=dcu:1 即单卡作业
#   已实测：nano 3ep/tsa15 = 97s，全链路通
#
# 本轮任务：补齐 bw 上因按开机时间计费而不划算的「多点小补测」
#   —— 全部是 300 ep 但模型小 / 数据短的组合，单独占 bw 不经济
set -u
cd /work/home/spotmaverick/nam_project || exit 1
LOG=dispatch_z100_r1.log
echo "[$(date '+%m-%d %H:%M:%S')] ===== Z100 ROUND1 DISPATCH =====" >> "$LOG"

submit() {
  local V=$1 S=$2 D=$3 EP=${4:-300}
  local JID
  JID=$(sbatch --parsable --job-name="${V:0:6}_${D:0:8}" sbatch_arch2.sh "$V" "$EP" "$S" "$D" 2>&1)
  if [[ "$JID" =~ ^[0-9]+$ ]]; then
    echo "[$(date '+%m-%d %H:%M:%S')] ok   $JID  ${V}/${D}/s${S} ep${EP}" >> "$LOG"
  else
    echo "[$(date '+%m-%d %H:%M:%S')] FAIL      ${V}/${D}/s${S} -> $JID" >> "$LOG"
  fi
}

# nano / feather 是轻量架构，bw 上跑纯属浪费卡时；z100 单卡完全够
submit nano    7  englrt50
submit nano    11 englrt50
submit feather 7  englrt50
submit feather 11 englrt50

# 中等难度箱的轻量架构补点（构成架构-难度曲线）
submit nano    7  mesa550_burn
submit feather 7  mesa550_burn
submit nano    7  markvx
submit feather 7  markvx

echo "[$(date '+%m-%d %H:%M:%S')] ===== DISPATCH DONE =====" >> "$LOG"
tail -20 "$LOG"
