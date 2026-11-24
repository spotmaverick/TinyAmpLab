#!/bin/bash
#SBATCH -p wzhdtest
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --gres=dcu:1
#SBATCH -t 1:00:00
source /work/home/spotmaverick/nam_probe/run_env.sh
cd /work/home/spotmaverick/nam_project
python analog_diag.py
