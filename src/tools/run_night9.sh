#!/usr/bin/env bash
# 第九批：逐层增益渐变（Layer Gain Taper）筛选
#
# 依据（2026-11-26 晚 V_SAT 扫描 + 逐层幅度诊断）：
#   官方 _Layer.forward 里
#       residual    = x + layer1x1(post_activation)   <- 过 layer1x1
#       head_output = post_activation                 <- **不过 layer1x1**
#   所以把 alpha 乘在 layer1x1 上只压残差链，head 累加路径完全不受影响。
#   实测：a-grp4(a=1.0) 与 scaled-grp4(a=0.5) 都需要 V_SAT≈64 才到 2×，
#   逐层诊断显示 head_in 累积幅度 30.8（preact 13.2），从 0.24 涨上来 55×。
#
#   正确作用点 = 每层 activation 之后，一个点同时覆盖两条路径。
#   gamma_i 随层号递减，是结构性约束（后期层要补偿需连续放大 1/gamma，
#   15 层累计天文数字 -> 梯度爆炸 -> 训练自然不走这条路）。
#
# 物理意义：每层激活后串一个固定电阻分压比，模拟电路零成本。
#
# 本批 6 个配置 × 1 种子 × 150ep，用于筛选；胜出者再到 4090 跑 300ep。
set -u
cd /root/autodl-tmp || exit 1

DATA=cfg/data_tsa15_4090.json
LEARN=${LEARN:-cfg/learn_a2_150ep.json}
LOGD=logs
mkdir -p "$LOGD" runs

MAXJOB=${MAXJOB:-6}
running() { jobs -rp | wc -l; }
wait_slot() { while [ "$(running)" -ge "$MAXJOB" ]; do sleep 5; done; }

launch() {
  local tag=$1 mcfg=$2 seed=$3
  local out="runs/n9_${tag}_s${seed}"
  if compgen -G "${out}/*/model.nam" > /dev/null 2>&1; then
    echo "[skip] $tag s$seed"; return
  fi
  rm -rf "$out"; wait_slot
  nohup python train_seeded.py "$DATA" "$mcfg" "$LEARN" "$out" "$seed" \
      --no-show --no-plots > "$LOGD/n9_${tag}_s${seed}.log" 2>&1 &
  echo "[launch] $tag s$seed -> $out"
}

echo "=== n9 gain-taper 筛选 启动 $(date +%H:%M:%S) ==="
for t in taper-e50 taper-e30 taper-e15 taper-e08 taper-late30 taper-const50; do
  launch "$t" "cfg2/model_${t}.json" 7
done

wait
echo "=== n9 全部完成 $(date +%H:%M:%S) ==="
