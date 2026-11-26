"""A2 基线批次：跑 A2 Full / A2 Lite 的独立训练锚点

在 4090 上串行跑。每任务 300ep，约 10 分钟。
用法：bash a2_baseline.sh
"""
set -u
cd /root/autodl-tmp
export PATH=/root/miniconda3/bin:$PATH
PY=/root/miniconda3/bin/python
mkdir -p logs

run() {
  local tag="$1"; shift
  echo "[$(date +%H:%M:%S)] START $tag : $*" | tee -a logs/baseline.log
  $PY train_a2.py "$@" >> logs/${tag}.log 2>&1
  echo "[$(date +%H:%M:%S)] END   $tag rc=$?" | tee -a logs/baseline.log
}

# A2 Full（12 通道，15697 参数）——Full 锚点
for s in 7 11; do
  run "a2_full_s${s}" a2 300 $s tsa15
done

# A2 Lite（3 通道，1090 参数，独立训练）——独立训练下限参考
for s in 7 11; do
  run "a2_lite_s${s}" a2_ch3 300 $s tsa15
done

echo "[$(date +%H:%M:%S)] ===== BASELINE ALL DONE =====" | tee -a logs/baseline.log
