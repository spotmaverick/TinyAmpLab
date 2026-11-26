#!/usr/bin/env bash
# 多种子复现：验证候选不是种子运气
# 用法: bash run_seeds.sh <par> <name> <seed1> [seed2 ...]
#   例: bash run_seeds.sh 3 a-grp4 7 11 23
set -u
PAR="${1:-3}"; shift
NAME="$1"; shift
SEEDS="$*"
export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp || exit 1
mkdir -p runs cfg/logs results

run_seed() {
  local name="$1" seed="$2"
  local model="cfg/model_${name}.json"
  local out="runs/seed_${name}_s${seed}"
  local log="cfg/logs/${name}_s${seed}.log"

  [ -f "$model" ] || { echo "[skip] $name 无配置"; return; }
  rm -rf "$out"
  echo "[start] $name seed=$seed"

  # 用 train_seeded.py（官方 main + 可控种子；族 A 配置同样适用）
  python train_seeded.py cfg/data_tsa15_4090.json "$model" cfg/learn_a2_100ep.json \
      "$out" "$seed" --no-show --no-plots > "$log" 2>&1

  local ck cfgf
  ck=$(ls -t "$out"/*/lightning_logs/version_*/checkpoints/*step*.ckpt 2>/dev/null | head -1)
  cfgf=$(find "$out" -maxdepth 2 -name config_model.json 2>/dev/null | head -1)
  if [ -n "$ck" ] && [ -n "$cfgf" ]; then
    python eval_nam_official.py "$ck" data/tsa15_val_input.wav \
        data/tsa15_val_target.wav "$cfgf" > "results/${name}_s${seed}_val.json" 2>/dev/null
    python eval_nam_official.py "$ck" data/tsa15_test_input.wav \
        data/tsa15_test_target.wav "$cfgf" > "results/${name}_s${seed}_test.json" 2>/dev/null
    python - "$name" "$seed" "results/${name}_s${seed}_test.json" <<'PY'
import json, sys
n, s, p = sys.argv[1:4]
try:
    t = json.load(open(p))
    print(f"[done] {n} s{s}: test={t['nrms']:.4f} corr={t['corr']:.4f}")
except Exception as e:
    print(f"[err] {n} s{s}: {e}")
PY
  else
    echo "[warn] $name s$seed 无产物"
  fi
}
export -f run_seed

for s in $SEEDS; do echo "$s"; done | xargs -P "$PAR" -I{} bash -c "run_seed '$NAME' {}"
echo "=== SEEDS DONE: $NAME ==="
