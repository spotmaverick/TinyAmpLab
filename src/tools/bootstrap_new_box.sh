#!/usr/bin/env bash
# 一键把「官方 nam + 数据 + 代码 + 已有 run」迁到新实例（3090 / 任意 AutoDL 机）
#
# 用法（在本地 Mac 执行，参数从 AutoDL 控制台复制）：
#   bash src/tools/bootstrap_new_box.sh <HOST> <PORT> <PASS> [TAG]
#
#   HOST  例如 connect.westc.seetacloud.com
#   PORT  例如 12345
#   PASS  root 密码
#   TAG   可选，默认 box2
#
# 做完后新机器路径：/root/autodl-tmp
# 预计耗时：安装 6~10 min，数据 78MB 约 30s，代码 5s
set -uo pipefail

HOST=${1:?需要 HOST}
PORT=${2:?需要 PORT}
PASS=${3:?需要 PASS}
TAG=${4:-box2}

BASE="/Users/wangguan/WorkBuddy/2026-11-23-15-15-52/analog-nam-gated-block-sim"
SRC_BOX="connect.nmb2.seetacloud.com"
SRC_PORT="24865"
SRC_PASS=$(grep AUTODL_PASS "$HOME/.workbuddy/.autodl_env" | cut -d'"' -f2)

ENVF="$HOME/.workbuddy/.autodl_${TAG}.env"
cat > "$ENVF" <<EOF
AUTODL_HOST="$HOST"
AUTODL_PORT="$PORT"
AUTODL_PASS="$PASS"
EOF
chmod 600 "$ENVF"
echo "[env] $ENVF 已写入"

SK="$HOME/.workbuddy/skills/autodl-4090-train/scripts/autodl.sh"

# 用临时 env 指向新机
run_new() { AUTODL_HOST=$HOST AUTODL_PORT=$PORT AUTODL_PASS=$PASS bash "$SK" "$@"; }
run_src() { AUTODL_HOST=$SRC_BOX AUTODL_PORT=$SRC_PORT AUTODL_PASS=$SRC_PASS bash "$SK" "$@"; }

echo "===== 1/5 目标机自检 ====="
run_new sh 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nproc; df -h /root/autodl-tmp | tail -1; /root/miniconda3/bin/python -c "import torch;print(\"torch\",torch.__version__,torch.cuda.is_available())"' || {
  echo "!! 目标机自检失败，检查 HOST/PORT/PASS"; exit 1; }

echo "===== 2/5 安装 nam 0.13.0 ====="
run_new sh 'export PATH=/root/miniconda3/bin:$PATH; pip install -q nam==0.13.0 2>&1 | tail -3; python -c "import nam;print(\"nam\",nam.__version__)"' || {
  echo "!! nam 安装失败"; exit 1; }

echo "===== 3/5 迁移数据集 ====="
# 源机打包
run_src sh 'cd /root/autodl-tmp && tar czf /tmp/data_pack.tgz data cfg 2>/dev/null; ls -lh /tmp/data_pack.tgz'
run_src get /tmp/data_pack.tgz /tmp/data_pack.tgz
ls -lh /tmp/data_pack.tgz
run_new put /tmp/data_pack.tgz /root/autodl-tmp/data_pack.tgz
run_new sh 'cd /root/autodl-tmp && tar xzf data_pack.tgz && rm data_pack.tgz && du -sh data cfg && ls data | head -5'

echo "===== 4/5 迁移代码 ====="
cd "$BASE"
for f in train_seeded.py analog_scaled_wavenet.py analog_sim_generic.py vsat_sweep.py eval_v2.py amp_probe.py; do
  [ -f "$f" ] && run_new put "$f" "/root/autodl-tmp/$f"
done
mkdir -p experiments/a2_official/cfg
run_new sh 'cd /root/autodl-tmp && mkdir -p logs runs results && ls *.py'

echo "===== 5/5 迁移环境专用配置 ====="
# 把原来 cfg/ 里的配置文件也带过去（在 data_pack 里已含 cfg）
run_new sh 'cd /root/autodl-tmp && ls cfg/*.json | head -20'

echo
echo "================================================"
echo " 完成。新机器启动命令示例："
echo "   AUTODL_HOST=$HOST AUTODL_PORT=$PORT AUTODL_PASS=*** \\"
echo "     bash $SK sh 'cd /root/autodl-tmp && bash src_tools_run_batch.sh'"
echo "================================================"
