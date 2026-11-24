#!/bin/bash
#SBATCH -p wzhdtest
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --gres=dcu:1
#SBATCH -t 4:00:00
source /work/home/spotmaverick/nam_probe/run_env.sh
cd /work/home/spotmaverick/nam_project
mkdir -p analog_sim
echo "=== DCU 冒烟 (5s quick) ==="
python analog_sim.py --quick > analog_sim/quick.log 2>&1
tail -6 analog_sim/quick.log
echo "=== 全量 (38.1s MC×16) ==="
python analog_sim.py > analog_sim/full.log 2>&1
echo "exit=$?"
tail -14 analog_sim/full.log
