#!/bin/bash
#SBATCH -J p3smoke
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=00:30:00
#SBATCH --gres=dcu:1
#SBATCH -o /public/home/spotmaverick/p26/p3smoke-%j.out
source ~/p26/bw2env.sh
cd /public/home/spotmaverick/nam_proj
export PYTHONPATH=/public/home/spotmaverick/t29:/public/home/spotmaverick/nam_proj:$PYTHONPATH
export P3_REQUIRE_ACCEL=1
echo "== JOB $SLURM_JOB_ID on $(hostname) $(date +%F\ %T) =="
START=$(date +%s)
# 冒烟：单种子 1 秒音频，三模式全过一遍，验证布局自校验与统计口径
$PY -u p3_sharp.py --mode all --seeds s7 --mc 1 --call-mc 1 --seconds 1.0 --tag smoke 2>&1 | tail -60
RC=$?
echo "== DONE rc=$RC elapsed=$(( $(date +%s)-START ))s =="
