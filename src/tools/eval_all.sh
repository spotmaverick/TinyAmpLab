#!/usr/bin/env bash
# 批量评测所有已完成的 run，输出排行榜
set -u
export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp || exit 1
mkdir -p results

for d in runs/cand_* runs/a2lite_off; do
  [ -d "$d" ] || continue
  name=$(basename "$d")
  ck=$(ls -t "$d"/*/lightning_logs/version_*/checkpoints/*step*.ckpt 2>/dev/null | head -1)
  [ -n "$ck" ] || { echo "[skip] $name 无 ckpt"; continue; }
  cfgf=$(find "$d" -maxdepth 2 -name config_model.json 2>/dev/null | head -1)
  [ -n "$cfgf" ] || { echo "[skip] $name 无 config"; continue; }

  python eval_nam_official.py "$ck" data/tsa15_val_input.wav \
      data/tsa15_val_target.wav "$cfgf" > "results/${name}_val.json" 2>/dev/null
  python eval_nam_official.py "$ck" data/tsa15_test_input.wav \
      data/tsa15_test_target.wav "$cfgf" > "results/${name}_test.json" 2>/dev/null

  python - "$name" "results/${name}_val.json" "results/${name}_test.json" <<'PY'
import json, sys
name, vp, tp = sys.argv[1:4]
try:
    v = json.load(open(vp)); t = json.load(open(tp))
    print(f"{name:<20} val={v['nrms']:.4f}  test={t['nrms']:.4f}  "
          f"corr={t['corr']:.4f}  rf={t['rf']}")
except Exception as e:
    print(f"{name:<20} ERR {e}")
PY
done
echo "=== EVAL ALL DONE ==="
