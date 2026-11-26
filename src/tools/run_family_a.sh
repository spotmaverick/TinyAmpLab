#!/usr/bin/env bash
# 4090 批量跑族 A 候选（官方配置表达，官方训练器）
# 用法: bash run_family_a.sh [epochs] [并发数]
set -u
EPOCHS="${1:-100}"
PAR="${2:-4}"
export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp || exit 1

DATA=cfg/data_tsa15_4090.json
mkdir -p runs cfg/logs

run_one() {
  local name="$1"
  local model="cfg/model_${name}.json"
  [ -f "$model" ] || { echo "skip $name (no cfg)"; return; }
  local out="runs/cand_${name}"
  local log="cfg/logs/${name}.log"
  echo "[start] $name"
  nam-full "$DATA" "$model" cfg/learn_a2_100ep.json "$out" --no-show --no-plots \
    > "$log" 2>&1
  local rc=$?
  echo "[done] $name rc=$rc"
  # 取最新 checkpoint 做 test 评测
  local ck
  ck=$(ls -t "$out"/*/lightning_logs/version_*/checkpoints/*.ckpt 2>/dev/null | head -1)
  if [ -n "$ck" ] && [ "$rc" = "0" ]; then
    python eval_nam_official.py "$ck" data/tsa15_test_input.wav \
      data/tsa15_test_target.wav > "cfg/logs/${name}.test.json" 2>>"$log"
    echo "[eval] $name -> $(cat cfg/logs/${name}.test.json 2>/dev/null | head -c 200)"
  fi
}
export -f run_one
export DATA

NAMES="a2 a-dw a-grp4 a-grp6 a-bneck4 a-bneck6 a-k3 a-l1grp4 a-dw-l1grp4 a-dw-l1grp3 a-dw-ch16 a-dw-ch16-l1grp4"
printf '%s\n' $NAMES | xargs -P "$PAR" -I{} bash -c 'run_one {}'
echo "ALL DONE"
