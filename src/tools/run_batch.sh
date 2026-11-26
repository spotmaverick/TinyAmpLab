#!/usr/bin/env bash
# 只跑指定变体（跳过已有完成 checkpoint 的），用于增量实验
# 用法: bash run_batch.sh <par> <name1> [name2 ...]
set -u
PAR="${1:-4}"; shift
export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp || exit 1
mkdir -p runs cfg/logs results

run_one() {
  local name="$1"
  local model="cfg/model_${name}.json"
  local out="runs/cand_${name}"
  local log="cfg/logs/${name}.log"

  [ -f "$model" ] || { echo "[skip] $name 无配置"; return; }

  # 若已有 100ep 完成标记则跳过
  if ls "$out"/*/lightning_logs/version_*/checkpoints/checkpoint_epoch_epoch=0099.ckpt \
      >/dev/null 2>&1; then
    echo "[skip] $name 已完成"
    return
  fi
  rm -rf "$out"

  echo "[start] $name"
  nam-full cfg/data_tsa15_4090.json "$model" cfg/learn_a2_100ep.json \
      "$out" --no-show --no-plots > "$log" 2>&1
  local rc=$?

  local ck cfgf
  ck=$(ls -t "$out"/*/lightning_logs/version_*/checkpoints/*step*.ckpt 2>/dev/null | head -1)
  cfgf=$(find "$out" -maxdepth 2 -name config_model.json 2>/dev/null | head -1)
  if [ -n "$ck" ] && [ -n "$cfgf" ]; then
    python eval_nam_official.py "$ck" data/tsa15_val_input.wav \
        data/tsa15_val_target.wav "$cfgf" > "results/${name}_val.json" 2>/dev/null
    python eval_nam_official.py "$ck" data/tsa15_test_input.wav \
        data/tsa15_test_target.wav "$cfgf" > "results/${name}_test.json" 2>/dev/null
    python - "$name" "results/${name}_val.json" "results/${name}_test.json" <<'PY'
import json, sys
name, vp, tp = sys.argv[1:4]
try:
    v = json.load(open(vp)); t = json.load(open(tp))
    print(f"[done] {name:<20} val={v['nrms']:.4f} test={t['nrms']:.4f} "
          f"corr={t['corr']:.4f} rf={t['rf']}")
except Exception as e:
    print(f"[eval-err] {name}: {e}")
PY
  else
    echo "[warn] $name 无 ckpt/config (rc=$rc)"
  fi
}
export -f run_one

printf '%s\n' "$@" | xargs -P "$PAR" -I{} bash -c 'run_one {}'
echo "=== BATCH DONE ==="
