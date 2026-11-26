#!/usr/bin/env bash
# 实验产物回传：远端 p1/out + 日志 → 本地 runs_remote/<TAG>/
# 用法: ./fetch_result.sh <TAG> [远端作业ID]
set -euo pipefail

TAG="${1:?用法: ./fetch_result.sh <TAG> [jobid]}"
JOBID="${2:-}"

DCU="$HOME/.workbuddy/skills/sugon-dcu-fleet/scripts/dcu.sh"
LOCAL_ROOT="$(cd "$(dirname "$0")" && pwd)"
DEST="$LOCAL_ROOT/runs_remote/$TAG"
REMOTE=/work/home/spotmaverick/nam_project
mkdir -p "$DEST"

echo "== 回传 $TAG =="

# 1. 结果 json / 日志
# dcu.sh pull 的语义是"远端目录 -> 本地父目录"，会自己建一层同名目录，
# 所以这里传父目录 DEST，落下来正好是 DEST/out/，不要传 DEST/out
mkdir -p "$DEST"
"$DCU" -m z100 pull "$REMOTE/p1/out" "$DEST" 2>&1 | grep -v "^rsync error" | tail -3 || true

# 2. 该作业的 slurm out/err（单独抓一份到 DEST 根，方便直接看）
if [ -n "$JOBID" ]; then
  for ext in out err; do
    "$DCU" -m z100 pull "$REMOTE/p1/out/p1*$JOBID.$ext" "$DEST/" 2>&1 | tail -1 || true
  done
fi

# 3. 环境指纹（记录本次跑在什么卡上）
"$DCU" -m z100 ssh 'bash -lc "module purge >/dev/null 2>&1; module load compiler/devtoolset/7.3.1 compiler/dtk/22.10.1 mpi/hpcx/gcc-7.3.1 apps/Pytorch/llama_py38 >/dev/null 2>&1; /public/software/apps/DeepLearning/PyTorch/llama_py38/bin/python -c \"import torch;print(torch.__version__, getattr(torch.version,\\\"hip\\\",None))\" 2>/dev/null; hy-smi 2>/dev/null | head -3"' \
  > "$DEST/env_fingerprint.txt" 2>&1 || true

echo "== 已回传到: $DEST =="
ls -la "$DEST"
