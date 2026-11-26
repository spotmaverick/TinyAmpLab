#!/usr/bin/env bash
# 第五批：depthwise 极限路线 + 族 A 完整横扫
#
# 动机：模拟电路里交叉开关（乘法器阵列）是第二大成本，仅次于延迟线。
# depthwise（groups_input = channels）把交叉开关降到理论最低，
# 代价是通道间完全不混合 -> Round1 实测 a-dw = 0.1117（差于 grp4 的 0.0775）。
# 但 a-dw-*-l1grpN 在 1x1 层恢复部分混合，可能是"零交叉开关 + 最小混合"的更优折中。
#
# 本批全部用 300ep（与修正后 A2 同协议）。
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
  local out="runs/n5_${tag}_s${seed}"
  if compgen -G "${out}/*/model.nam" > /dev/null 2>&1; then
    echo "[skip] $tag s$seed"; return
  fi
  rm -rf "$out"; wait_slot
  echo "[$(date +%H:%M:%S)] START $tag s$seed"
  nohup python train_seeded.py "$DATA" "$mcfg" "$LEARN" "$out" "$seed" \
      --no-show --no-plots > "$LOGD/n5_${tag}_s${seed}.log" 2>&1 &
}

echo "===== 第五批开始 $(date +%H:%M:%S) MAXJOB=$MAXJOB ====="

# 零交叉开关 + 最小混合（最有模拟价值的方向）
launch a-dw-l1grp3        cfg/model_a-dw-l1grp3.json        7
launch a-dw-l1grp4        cfg/model_a-dw-l1grp4.json        7
launch a-dw-ch16-l1grp4   cfg/model_a-dw-ch16-l1grp4.json   7
# 纯 depthwise 扩宽（交叉开关仍为 0）
launch a-dw-ch16          cfg/model_a-dw-ch16.json          7
launch a-dw-ch24          cfg/model_a-dw-ch24.json          7
# 1x1 分组但主卷积全连接（隔离"混合发生在哪"的影响）
launch a-l1grp3           cfg/model_a-l1grp3.json           7

wait
echo "===== 第五批结束 $(date +%H:%M:%S) ====="
