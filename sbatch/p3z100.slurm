#!/bin/bash
#SBATCH -J p3z100
#SBATCH -p wzhdtest
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=14G
#SBATCH --time=03:00:00
#SBATCH --gres=dcu:1
#SBATCH -o /work/home/spotmaverick/nam_project/runs/p3z100-%j.out
export LD_LIBRARY_PATH=""
module purge
module load compiler/devtoolset/7.3.1 compiler/dtk/24.04.3 mpi/hpcx/gcc-7.3.1
export TORCH_PY=/work/home/spotmaverick/torch21_env/bin/python
cd /work/home/spotmaverick/nam_project
export P3_REQUIRE_ACCEL=1
echo "== JOB $SLURM_JOB_ID on $(hostname) $(date +%F\ %T) =="
START=$(date +%s)
# 交叉验证：与 bw2 同脚本同参数，验证 P3 结论不依赖机器。
# call 模式在 z100 上慢，关掉；stats+layer 足够验证主线结论。
$TORCH_PY -u p3_sharp.py --mode stats --seeds s7 s11 s23 s31 s37 --mc 8 \
    --seconds 5.0 --tag z100stats 2>&1 | tail -40
$TORCH_PY -u p3_sharp.py --mode layer --seeds s7 s11 s23 s31 s37 --mc 8 \
    --seconds 5.0 --tag z100layer 2>&1 | tail -100
RC=$?
echo "== DONE rc=$RC elapsed=$(( $(date +%s)-START ))s =="
