#!/bin/bash
#SBATCH -p wzhdtest
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --gres=dcu:1
#SBATCH -t 6:00:00
source /work/home/spotmaverick/nam_probe/run_env.sh
cd /work/home/spotmaverick/nam_project
mkdir -p analog_sim
echo "=== 1. 对拍 (关全部非理想性, 必须 PASS) ==="
python analog_sim_wavenet.py --sanity > analog_sim/lite_sanity.log 2>&1
echo "sanity exit=$?"; tail -6 analog_sim/lite_sanity.log
echo
echo "=== 2. 快速冒烟 (5s, MCx3) ==="
python analog_sim_wavenet.py --quick > analog_sim/lite_quick.log 2>&1
echo "quick exit=$?"; tail -10 analog_sim/lite_quick.log
echo
echo "=== 3. 全量 (38.1s, MCx16) ==="
python analog_sim_wavenet.py > analog_sim/lite_full.log 2>&1
echo "full exit=$?"; tail -14 analog_sim/lite_full.log
