#!/usr/bin/env bash
# 统一评测：扫全部 run 目录，取最后一个 ckpt，用官方 esr 评测
#
# 注意：整段推理需要较多显存（test 段 182 万采样）。若与训练并发会 OOM，
# 因此这里 (1) 每次只评一个，(2) 评之间强制释放并等待，(3) 失败就跳过。
set -u
cd /root/autodl-tmp || exit 1

OUT=results_final.jsonl
: > "$OUT"

echo "===== 评测开始 $(date +%H:%M:%S) ====="

free_mb() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '; }
wait_vram() {  # 等到显存占用低于 $1 MiB
  local lim=$1 n=0
  while [ "$(free_mb)" -gt "$lim" ] && [ $n -lt 40 ]; do sleep 5; n=$((n+1)); done
}

for d in runs/n3_*/ runs/n4_*/ runs/n5_*/ runs/n6_*/ runs/n7_*/ runs/n8_*/; do
  [ -d "$d" ] || continue
  case "$d" in *DEADLOCK*) echo "[skip] $d"; continue;; esac
  ck=$(find "$d" -name 'epoch=*.ckpt' 2>/dev/null | sort -t= -k2 -n | tail -1)
  [ -z "$ck" ] && { echo "[no-ckpt] $d"; continue; }
  tag=$(basename "$d")

  wait_vram 12000
  line=$(python eval_v2.py "$ck" --slim 1 2>/dev/null | tail -1)
  if echo "$line" | grep -q '"nrms"'; then
    echo "$line" >> "$OUT"
    nr=$(echo "$line" | python -c 'import sys,json;print(f"{json.load(sys.stdin)[\"nrms\"]:.4f}")')
    pr=$(echo "$line" | python -c 'import sys,json;print(json.load(sys.stdin)[\"params\"])')
    printf '[ok] %-30s NRMS=%s params=%s\n' "$tag" "$nr" "$pr"
  else
    printf '[fail] %-30s\n' "$tag"
  fi
  # slimmable 模型额外评 Lite 分支
  if echo "$line" | grep -q '"slimmed": true'; then
    wait_vram 12000
    l=$(python eval_v2.py "$ck" --slim 0 2>/dev/null | tail -1)
    echo "$l" | grep -q '"nrms"' && echo "$l" >> "$OUT"
  fi
  sleep 2
done

echo "===== 评测完成 $(date +%H:%M:%S) ====="
wc -l "$OUT"
