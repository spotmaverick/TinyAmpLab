#!/bin/bash
#SBATCH -J p4smoke
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=01:00:00
#SBATCH --gres=dcu:1
#SBATCH -o /public/home/spotmaverick/p26/p4smoke-%j.out
source ~/p26/bw2env.sh
cd /public/home/spotmaverick/nam_proj
export PYTHONPATH=/public/home/spotmaverick/t29:/public/home/spotmaverick/nam_proj:$PYTHONPATH
export P4_REQUIRE_ACCEL=1
echo "== SMOKE $SLURM_JOB_ID on $(hostname) $(date +%F\ %T) =="
$PY -u p4_proxy.py --mode all --seeds s7 --mc 1 --tag smoke2
echo "== DONE rc=$? =="
