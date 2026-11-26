#!/bin/bash
#SBATCH -J p1smoke2
#SBATCH -p wzhdtest
#SBATCH -c 4
#SBATCH --mem=14000M
#SBATCH --gres=dcu:1
#SBATCH -t 00:20:00
#SBATCH -o /work/home/spotmaverick/nam_project/p1/out/p1smoke2-%j.out
#SBATCH -e /work/home/spotmaverick/nam_project/p1/out/p1smoke2-%j.err

# =============================================================================
# 海光 DCU Slurm 作业模板（由 dcu.sh submit 渲染）
#
# 两条硬规则（血泪教训，勿删）：
#   1. 输出路径必须是集群共享盘真实路径。zzeshell 上 /home 是登录节点的软链，
#      计算节点不存在 → 作业 1 秒内 Signal 53 死掉且不产生输出文件。
#   2. zzeshell 必须 source $DTKROOT/env.sh，只 module load 会缺 lib64 →
#      "load comgr library error" / HIP_DEVICES=0。
# =============================================================================

set -o pipefail

# 可选的环境准备（--env 注入；为空则跳过）
#
# ⚠ 必须在此之前不要开 `set -u`：
#   DTK 的 env.sh 内部直接引用 $LD_LIBRARY_PATH，若该变量未定义，
#   `set -u` 会让脚本在这一行就退出，表现为 ExitCode=1:0、无任何输出。
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PATH="${PATH:-/usr/local/bin:/usr/bin:/bin}"

module purge; module load compiler/devtoolset/7.3.1; module load compiler/dtk/24.04.3; module load mpi/hpcx/gcc-7.3.1; export TORCH_PY=/work/home/spotmaverick/torch21_env/bin/python; export P1_REQUIRE_ACCEL=1; export P1_ROOT=/work/home/spotmaverick/nam_project

cd /work/home/spotmaverick/nam_project || { echo "FATAL: 工作目录不存在: /work/home/spotmaverick/nam_project"; exit 1; }

echo "===== JOB INFO ====="
echo "job_id   : ${SLURM_JOB_ID:-NA}"
echo "job_name : ${SLURM_JOB_NAME:-NA}"
echo "node     : $(hostname)"
echo "partition: ${SLURM_JOB_PARTITION:-NA}"
echo "start    : $(date '+%F %T')"
echo "cwd      : $(pwd)"
echo "gres     : ${SLURM_GPUS_ON_NODE:-NA}"

echo "===== DCU 可见性 ====="
SMI=""
for c in "${DTKROOT:-/nonexistent}/bin/rocm-smi" rocm-smi hy-smi; do
  command -v "$c" >/dev/null 2>&1 && { SMI="$c"; break; }
done
if [ -n "$SMI" ]; then
  "$SMI" --showproductname --showmeminfo vram 2>/dev/null \
    | grep -E 'Card Series|Total Memory|DCU\[|HCU\[' | head -6
else
  # z100 的 dtk module 不设 DTKROOT，也没有 rocm-smi/hy-smi，只能靠 torch 反查
  echo "(no smi tool; querying via torch / hip runtime)"
fi
# 用 HIP runtime 直接数卡，不依赖 torch 是否装好
python3 -c "
try:
    import ctypes
    h = ctypes.CDLL('libamdhip64.so'); n = ctypes.c_int()
    rc = h.hipGetDeviceCount(ctypes.byref(n))
    print(f'hip_devices rc={rc} n={n.value}')
except Exception as e:
    print('hip probe skipped:', e)
" 2>/dev/null || true
# 有 torch 时补一条带型号/显存的记录
PYBIN=""
for c in "${TORCH_PY:-}" python3 python; do
  [ -n "$c" ] && command -v "$c" >/dev/null 2>&1 && { PYBIN="$c"; break; }
done
if [ -n "$PYBIN" ]; then
  "$PYBIN" -c "
import torch
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'torch {torch.__version__}  device={p.name}  mem={p.total_memory/1024**3:.1f}GB  count={torch.cuda.device_count()}')
else:
    print(f'torch {torch.__version__}  cuda_available=False')
" 2>/dev/null || true
fi

echo "===== RUN ====="
START=$(date +%s)

cd /work/home/spotmaverick/nam_project && $TORCH_PY -u p1_run.py --seeds s7 --mc 1 --seconds 0.5 --skip-sweep --tag sm2

RC=$?
END=$(date +%s)
echo "===== DONE ====="
echo "rc=${RC}  elapsed=$((END-START))s  end=$(date '+%F %T')"
exit $RC
