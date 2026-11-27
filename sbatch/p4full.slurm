#!/bin/bash
#SBATCH -J p4full
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --gres=dcu:1
#SBATCH -o /public/home/spotmaverick/p26/p4full-%j.out
source ~/p26/bw2env.sh
cd /public/home/spotmaverick/nam_proj
export PYTHONPATH=/public/home/spotmaverick/t29:/public/home/spotmaverick/nam_proj:$PYTHONPATH
export P4_REQUIRE_ACCEL=1
echo "== P4 FULL $SLURM_JOB_ID on $(hostname) $(date +%F\ %T) =="
$PY -u p4_proxy.py --mode proxy --mc 8 --tag full_1127
echo "== PROXY rc=$? =="
$PY -u p4_proxy.py --mode causal --mc 8 --tag causal_1127
echo "== CAUSAL rc=$? =="
echo "== DONE $(date +%F\ %T) =="
