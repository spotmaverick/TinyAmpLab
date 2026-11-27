#!/bin/bash
#SBATCH -J p4z100
#SBATCH -p wzhdtest
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --gres=dcu:1
#SBATCH -o /work/home/spotmaverick/nam_project/runs/p4z100-%j.out
export LD_LIBRARY_PATH=""
module purge
module load compiler/devtoolset/7.3.1 compiler/dtk/24.04.3 mpi/hpcx/gcc-7.3.1
export TORCH_PY=/work/home/spotmaverick/torch21_env/bin/python
cd /work/home/spotmaverick/nam_project
export P4_REQUIRE_ACCEL=1
echo "== JOB $SLURM_JOB_ID on $(hostname) $(date +%F\ %T) =="
START=$(date +%s)
$TORCH_PY -u p4_proxy.py --mode causal --mc 4 --tag z100causal
echo "== causal rc=$? =="
$TORCH_PY -u p4_paircheck.py 2>&1 | tail -40
echo "== DONE rc=$? elapsed=$(( $(date +%s)-START ))s =="
