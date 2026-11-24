#!/bin/bash
# 智算服务器 → MacBook 本地 全量回传整理
# 用法: bash sync_back.sh [group]
#   group = data | weights | results | all (默认 all)
# 目录结构 (本地 analog-nam-gated-block-sim/sync/):
#   data/<amp>_<split>_<io>.wav       # 48kHz 训练数据
#   weights/e1_official/<arch>_tsa15/ # 官方包权重 (model.nam + result.json)
#   weights/e1_dcu/<arch>_tsa15_dcu/  # DCU 移植版官方架构权重
#   weights/e2_lgw/<variant>/         # LGW 官方协议权重
#   weights/legacy_dcu/<run>/         # 原架构搜索 22 run 权重
#   results/logs/                     # 训练日志
#   results/ledger/                   # runs.csv / 实验账本
set -e
KEY="/Users/wangguan/Downloads/spotmaverick_wuzh02.hpccube.com_RsaKeyExpireTime_2027-02-21_21-34-03.txt"
HOST="spotmaverick@wuzh02.hpccube.com"
PORT=65091
SRV=/work/home/spotmaverick/nam_project
BASE="/Users/wangguan/WorkBuddy/2026-11-23-15-15-52/analog-nam-gated-block-sim"
SYNC="$BASE/sync"
GROUP="${1:-all}"

SSH() { ssh -i "$KEY" -p $PORT -o BatchMode=yes $HOST "$@"; }

mkdir -p "$SYNC"/{data,weights/{e1_official,e1_dcu,e2_lgw,legacy_dcu},results/{logs,ledger}}

if [ "$GROUP" = "data" ] || [ "$GROUP" = "all" ]; then
  echo "=== 回传数据 wav ==="
  SSH "mkdir -p $SRV/tmp_sync && cd $SRV/data && tar czf $SRV/tmp_sync/data_sync.tar.gz *.wav" 
  scp -i "$KEY" -P $PORT -q $HOST:$SRV/tmp_sync/data_sync.tar.gz $SRV/tmp_sync/data_sync.tar.gz
  tar xzf $SRV/tmp_sync/data_sync.tar.gz -C "$SYNC/data" && rm $SRV/tmp_sync/data_sync.tar.gz
  SSH "rm -f $SRV/tmp_sync/data_sync.tar.gz"
  echo "数据: $(ls "$SYNC/data" | wc -l | tr -d ' ') 个文件"
fi

if [ "$GROUP" = "weights" ] || [ "$GROUP" = "all" ]; then
  echo "=== 回传权重 ==="
  # E1 官方包 (model.nam + result.json)
  SSH "cd $SRV/e1 && tar czf $SRV/tmp_sync/w1.tar.gz --exclude='lightning_logs' nano_tsa15 feather_tsa15 lite_tsa15 standard_tsa15 2>/dev/null"
  scp -i "$KEY" -P $PORT -q $HOST:$SRV/tmp_sync/w1.tar.gz $SRV/tmp_sync/w1.tar.gz 2>/dev/null && \
    tar xzf $SRV/tmp_sync/w1.tar.gz -C "$SYNC/weights/e1_official" && rm $SRV/tmp_sync/w1.tar.gz
  # E1 DCU 移植版
  SSH "cd $SRV/e1 && tar czf $SRV/tmp_sync/w2.tar.gz nano_tsa15_dcu feather_tsa15_dcu lite_tsa15_dcu standard_tsa15_dcu 2>/dev/null"
  scp -i "$KEY" -P $PORT -q $HOST:$SRV/tmp_sync/w2.tar.gz $SRV/tmp_sync/w2.tar.gz 2>/dev/null && \
    tar xzf $SRV/tmp_sync/w2.tar.gz -C "$SYNC/weights/e1_dcu" && rm $SRV/tmp_sync/w2.tar.gz
  # E2 LGW 官方协议
  SSH "cd $SRV && tar czf $SRV/tmp_sync/w3.tar.gz --exclude='*/train.log' e2_lgw_official 2>/dev/null"
  scp -i "$KEY" -P $PORT -q $HOST:$SRV/tmp_sync/w3.tar.gz $SRV/tmp_sync/w3.tar.gz 2>/dev/null && \
    tar xzf $SRV/tmp_sync/w3.tar.gz -C "$SYNC/weights" && rm $SRV/tmp_sync/w3.tar.gz
  # 旧架构搜索权重 (22 run)
  SSH "cd $SRV && du -sh dcu_runs && tar czf $SRV/tmp_sync/w4.tar.gz dcu_runs/*/weights.pt dcu_runs/runs.csv dcu_runs/runs.md 2>/dev/null"
  scp -i "$KEY" -P $PORT -q $HOST:$SRV/tmp_sync/w4.tar.gz $SRV/tmp_sync/w4.tar.gz 2>/dev/null && \
    tar xzf $SRV/tmp_sync/w4.tar.gz -C "$SYNC/weights/legacy_dcu" && rm $SRV/tmp_sync/w4.tar.gz
  SSH "rm -f $SRV/tmp_sync/w1.tar.gz $SRV/tmp_sync/w2.tar.gz $SRV/tmp_sync/w3.tar.gz $SRV/tmp_sync/w4.tar.gz"
  echo "权重目录:"; du -sh "$SYNC/weights"/* 2>/dev/null
fi

if [ "$GROUP" = "results" ] || [ "$GROUP" = "all" ]; then
  echo "=== 回传日志与账本 ==="
  SSH "cd $SRV/e1 && tar czf $SRV/tmp_sync/l1.tar.gz e1_*_tsa15.log e1_dcu_*_tsa15.log e2_*.log param_table.json 2>/dev/null"
  scp -i "$KEY" -P $PORT -q $HOST:$SRV/tmp_sync/l1.tar.gz $SRV/tmp_sync/l1.tar.gz 2>/dev/null && \
    tar xzf $SRV/tmp_sync/l1.tar.gz -C "$SYNC/results/logs" && rm $SRV/tmp_sync/l1.tar.gz
  # 所有 result.json 汇总
  SSH "cd $SRV && find e1 e1_dcu e2_lgw_official -name result.json 2>/dev/null | tar czf $SRV/tmp_sync/l2.tar.gz -T -"
  scp -i "$KEY" -P $PORT -q $HOST:$SRV/tmp_sync/l2.tar.gz $SRV/tmp_sync/l2.tar.gz 2>/dev/null && \
    tar xzf $SRV/tmp_sync/l2.tar.gz -C "$SYNC/results/ledger" && rm $SRV/tmp_sync/l2.tar.gz
  SSH "rm -f $SRV/tmp_sync/l1.tar.gz $SRV/tmp_sync/l2.tar.gz"
  echo "日志: $(ls "$SYNC/results/logs" | wc -l | tr -d ' ') 个"
fi

echo "=== sync_back 完成 ($GROUP) ==="
