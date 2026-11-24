#!/bin/bash
#SBATCH -p wzhdtest
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --gres=dcu:1
#SBATCH -t 12:00:00
EXP=$1
EP=$2
source /work/home/spotmaverick/nam_probe/run_env.sh
cd /work/home/spotmaverick/nam_project
mkdir -p diag3
python diag_three.py $EXP $EP > diag3/${EXP}.log 2>&1
echo "exit=$? tail:"; tail -3 diag3/${EXP}.log
