#!/usr/bin/env bash
# 第三批（收尾夜）：统一 300ep 协议下的公平对比 + 族 B2 收尾
#
# 关键动机：Round1 用 100ep 官方协议，A2 只有 test NRMS 0.1786，
# 而自写 trainer 跑 300ep 能到 0.0710 —— 100ep 下 A2 是欠训练的。
# 拿欠训练的 A2 当基准会高估族 A 的收益，因此必须先补齐同协议 300ep 基线。
#
# 并发度：4090 显存只用 1.7GB/24GB，128 核 CPU，可安全跑 8 路。
set -u
cd /root/autodl-tmp || exit 1

DATA=cfg/data_tsa15_4090.json
LEARN=cfg/learn_a2_300ep.json
LOGD=logs
mkdir -p "$LOGD" runs

MAXJOB=${MAXJOB:-8}

running() { jobs -rp | wc -l; }
wait_slot() {
  while [ "$(running)" -ge "$MAXJOB" ]; do sleep 5; done
}

launch() {  # launch <tag> <model_cfg> <seed> <learn_cfg>
  local tag=$1 mcfg=$2 seed=$3 lcfg=${4:-$LEARN}
  local out="runs/n3_${tag}_s${seed}"
  if compgen -G "${out}/*/model.nam" > /dev/null 2>&1; then
    echo "[skip] $tag s$seed 已完成"
    return
  fi
  rm -rf "$out"
  wait_slot
  echo "[$(date +%H:%M:%S)] START $tag s$seed"
  nohup python train_seeded.py "$DATA" "$mcfg" "$lcfg" "$out" "$seed" \
      --no-show --no-plots > "$LOGD/n3_${tag}_s${seed}.log" 2>&1 &
}

echo "===== 第三批开始 $(date +%H:%M:%S)  MAXJOB=$MAXJOB ====="

# ---------------- 优先级 1：同协议基线（决定全部结论是否成立）----------------
# A2 Full（12ch）：100ep 的 0.1786 需要同协议 300ep 才能作公平基准
launch a2         cfg/model_a2.json         7
launch a2         cfg/model_a2.json        11

# ---------------- 优先级 2：族 A 最优，300ep + 多种子 ----------------
# 若 300ep 下族 A 仍优于 A2，则结论稳固；否则 Round1 的 −61% 是欠训练假象
launch a-grp6-ch18 cfg2/model_a-grp6-ch18.json  7
launch a-grp6-ch18 cfg2/model_a-grp6-ch18.json 11
launch a-grp4       cfg/model_a-grp4.json       7
launch a-grp4       cfg/model_a-grp4.json      11
launch a-grp4-rf50  cfg2/model_a-grp4-rf50.json  7

# ---------------- 优先级 3：族 B2 收尾（延迟线归零路线）----------------
launch recurrent_16b4g4 cfg/model_recurrent_16b4g4.json 7
launch recurrent_32b6g4 cfg/model_recurrent_32b6g4.json 7

wait
echo "===== 第三批全部结束 $(date +%H:%M:%S) ====="
for d in runs/n3_*/; do
  ck=$(find "$d" -name "*.ckpt" 2>/dev/null | sort | tail -1)
  echo "$d -> ${ck:-NO_CKPT}"
done
