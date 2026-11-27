#!/bin/bash
#SBATCH -J p4check
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --gres=dcu:1
#SBATCH -o /public/home/spotmaverick/p26/p4check-%j.out
source ~/p26/bw2env.sh
cd /public/home/spotmaverick/nam_proj
export PYTHONPATH=/public/home/spotmaverick/t29:/public/home/spotmaverick/nam_proj:$PYTHONPATH
export P4_REQUIRE_ACCEL=1
echo "== CHECK $SLURM_JOB_ID on $(hostname) $(date +%F\ %T) =="
$PY -u check_p3p4.py
echo "== DONE rc=$? =="
