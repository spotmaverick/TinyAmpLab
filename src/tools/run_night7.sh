#!/usr/bin/env bash
# 第七批：残差缩放重训（模拟原生架构的最后一环）
#
# 依据（见 ANALOG_SIM_FINDINGS.md）：
#   器件仿真显示 a-grp4 数字 NRMS 0.065，器件下退化到 0.415（44.7×）。
#   消融证明 100% 来自 Gilbert cell 三次失真，根因是残差流幅度 0.047->22.6 RMS。
#   试过减层/加 bottleneck/前端 AGC 全部无效（幅度反而更大）。
#   直接缩放已训练权重会崩（NRMS 0.064->0.81），必须重训。
#
# 本批：ScaledWaveNet，每层 layer1x1 输出乘固定 alpha。
#   物理意义：残差支路上的电阻分压比，模拟电路零成本。
set -u
cd /root/autodl-tmp || exit 1

DATA=cfg/data_tsa15_4090.json
LEARN=cfg/learn_a2_300ep.json
LOGD=logs
mkdir -p "$LOGD" runs

MAXJOB=${MAXJOB:-4}
running() { jobs -rp | wc -l; }
wait_slot() { while [ "$(running)" -ge "$MAXJOB" ]; do sleep 5; done; }

launch() {
  local tag=$1 mcfg=$2 seed=$3
  local out="runs/n7_${tag}_s${seed}"
  if compgen -G "${out}/*/model.nam" > /dev/null 2>&1; then
    echo "[skip] $tag s$seed"; return
  fi
  rm -rf "$out"; wait_slot
  echo "[$(date +%H:%M:%S)] START $tag s$seed"
  nohup python train_seeded.py "$DATA" "$mcfg" "$LEARN" "$out" "$seed" \
      --no-show --no-plots > "$LOGD/n7_${tag}_s${seed}.log" 2>&1 &
}

echo "===== 第七批（残差缩放）开始 $(date +%H:%M:%S) MAXJOB=$MAXJOB ====="

launch scaled-grp4-a50 cfg2/model_scaled-grp4-a50.json 7
launch scaled-grp4-a50 cfg2/model_scaled-grp4-a50.json 11
launch scaled-grp4-a30 cfg2/model_scaled-grp4-a30.json 7
launch scaled-grp4-a30 cfg2/model_scaled-grp4-a30.json 11

wait
echo "===== 第七批结束 $(date +%H:%M:%S) ====="
