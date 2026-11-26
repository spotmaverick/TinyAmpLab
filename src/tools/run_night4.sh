#!/usr/bin/env bash
# 第四批（修正基线）：在发现 A2 slimmable 死锁后，重建公平基线
#
# 核心修正：
#   官方 wavenet_slimmable.json 用 init_strategy=smallest_and_zeros，
#   该策略把大通道 weight+bias 全清零，而 input_mixer 无 bias，
#   于是大通道输入恒为 0 -> 梯度恒为 0 -> 12ch 分支永久死锁。
#   实测：ratio>=0.5 时总梯度严格为 0，导出权重 15686 个里 14598 个是 0。
#   => 官方 100ep 的 "A2" 实际只训练了 3 个通道，Full/Lite 输出完全一致。
#
#   改用官方另一策略 init_strategy=channel_causal 后，ratio=1.0 梯度 3.7e-01，正常。
#
# 本批任务：
#   1. A2-causal 300ep（Full 12ch 真正训练过的基线）—— 最高优先级
#   2. A2-causal 100ep（与 Round1 的 100ep 协议对齐，验证死锁影响）
#   3. A3ch 300ep（族A 3通道，与 A2 Lite 同容量对比）
set -u
cd /root/autodl-tmp || exit 1

DATA=cfg/data_tsa15_4090.json
L300=cfg/learn_a2_300ep.json
L100=cfg/learn_a2_100ep.json
LOGD=logs
mkdir -p "$LOGD" runs

MAXJOB=${MAXJOB:-4}
running() { jobs -rp | wc -l; }
wait_slot() { while [ "$(running)" -ge "$MAXJOB" ]; do sleep 5; done; }

launch() {  # launch <tag> <model_cfg> <seed> <learn_cfg>
  local tag=$1 mcfg=$2 seed=$3 lcfg=${4:-$L300}
  local out="runs/n4_${tag}_s${seed}"
  if compgen -G "${out}/*/model.nam" > /dev/null 2>&1; then
    echo "[skip] $tag s$seed 已完成"; return
  fi
  rm -rf "$out"; wait_slot
  echo "[$(date +%H:%M:%S)] START $tag s$seed"
  nohup python train_seeded.py "$DATA" "$mcfg" "$lcfg" "$out" "$seed" \
      --no-show --no-plots > "$LOGD/n4_${tag}_s${seed}.log" 2>&1 &
}

echo "===== 第四批（修正基线）开始 $(date +%H:%M:%S) MAXJOB=$MAXJOB ====="

# 最高优先级：A2 真正训练过的 Full 分支
launch a2causal-300 cfg/model_a2_causal.json 7  "$L300"
launch a2causal-300 cfg/model_a2_causal.json 11 "$L300"
# 对齐 Round1 的 100ep 协议，量化死锁造成多大偏差
launch a2causal-100 cfg/model_a2_causal.json 7  "$L100"
# 族A 3 通道（与 A2 Lite 同容量）
launch a3ch-300     cfg/model_a3ch.json      7  "$L300"

wait
echo "===== 第四批结束 $(date +%H:%M:%S) ====="
