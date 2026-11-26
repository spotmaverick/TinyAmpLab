#!/usr/bin/env bash
# 自动等待全部训练完成，然后统一评测并回传
# 用法：setsid nohup bash watchdog.sh > logs/watchdog.log 2>&1 &
set -u
cd /root/autodl-tmp || exit 1

DEADLINE=${DEADLINE:-22:40}   # 用户计划 23:00 关机，留 20 分钟评测+回传

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "watchdog 启动，截止 $DEADLINE"

while true; do
  running=$(ps aux | grep '[t]rain_see' | grep -c python)
  now=$(date +%H%M)
  # 截止时间换算成数字比较
  dh=${DEADLINE%:*}; dm=${DEADLINE#*:}
  cutoff=$((10#$dh * 100 + 10#$dm))

  if [ "$running" -eq 0 ]; then
    log "全部训练结束（无活跃进程）"
    break
  fi
  if [ "$((10#$now))" -ge "$cutoff" ]; then
    log "到达截止时间，仍有 $running 个进程，强制收尾"
    break
  fi
  log "运行中 $running 个进程，等待…"
  sleep 120
done

log "===== 开始统一评测 ====="
bash eval_sweep.sh 2>&1 | tail -60

log "===== 打包产物 ====="
tar czf /root/autodl-tmp/results_$(date +%m%d_%H%M).tgz \
    results_v2.jsonl runs/*/*/model.nam logs/n*.log 2>/dev/null
ls -lh /root/autodl-tmp/results_*.tgz | tail -3
log "watchdog 完成"
