#!/usr/bin/env bash
# 第六批：从"模拟电路摆幅"这一物理约束反推架构
#
# 动机（器件仿真实测）：
#   a-grp4 的数字 NRMS=0.065，但放进器件模型后退化到 0.415（41×）。
#   消融显示退化 100% 来自 Gilbert cell 三次失真（关闭后回到 0.073）。
#   根因：残差流幅度从 0.20 单调增长到 89.8（450×），乘法器必然饱和。
#   V_SAT 扫描：要压到 2× 退化需 V_SAT≈64~128，意味着极高电源摆幅。
#
# 既然残差幅度增长是主因，就试两条抑制路线：
#   bn6   —— bottleneck=6 压缩 1x1 宽度，降低每层注入能量
#   L10   —— 层数 15->10，直接减少累加次数
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
  local out="runs/n6_${tag}_s${seed}"
  if compgen -G "${out}/*/model.nam" > /dev/null 2>&1; then
    echo "[skip] $tag s$seed"; return
  fi
  rm -rf "$out"; wait_slot
  echo "[$(date +%H:%M:%S)] START $tag s$seed"
  nohup python train_seeded.py "$DATA" "$mcfg" "$LEARN" "$out" "$seed" \
      --no-show --no-plots > "$LOGD/n6_${tag}_s${seed}.log" 2>&1 &
}

echo "===== 第六批（摆幅约束反推）开始 $(date +%H:%M:%S) MAXJOB=$MAXJOB ====="

launch a-grp6-ch18-bn6 cfg2/model_a-grp6-ch18-bn6.json 7
launch a-grp6-ch18-bn6 cfg2/model_a-grp6-ch18-bn6.json 11
launch a-grp6-ch18-L10 cfg2/model_a-grp6-ch18-L10.json 7
launch a-grp6-ch18-L10 cfg2/model_a-grp6-ch18-L10.json 11

wait
echo "===== 第六批结束 $(date +%H:%M:%S) ====="
