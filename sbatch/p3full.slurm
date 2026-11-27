#!/bin/bash
#SBATCH -J p3full
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=03:00:00
#SBATCH --gres=dcu:1
#SBATCH -o /public/home/spotmaverick/p26/p3full-%j.out
source ~/p26/bw2env.sh
cd /public/home/spotmaverick/nam_proj
export PYTHONPATH=/public/home/spotmaverick/t29:/public/home/spotmaverick/nam_proj:$PYTHONPATH
export P3_REQUIRE_ACCEL=1
echo "== JOB $SLURM_JOB_ID on $(hostname) $(date +%F\ %T) =="
START=$(date +%s)
# 主作业：5 种子 × MC=8，三模式全跑。
# call 模式成本 = 83 次前向 × call_mc，用 2 次 MC 控制总时长。
$PY -u p3_sharp.py --mode all \
    --seeds s7 s11 s23 s31 s37 --mc 8 --call-mc 2 \
    --seconds 5.0 --tag bw2 2>&1 | tail -120
RC=${PIPESTATUS[0]}
echo "== DONE rc=$RC elapsed=$(( $(date +%s)-START ))s =="
