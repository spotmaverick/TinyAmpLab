#!/bin/bash
#SBATCH -p wzhdtest
#SBATCH -c 4
#SBATCH --mem=12G
#SBATCH --gres=dcu:1
#SBATCH -t 36:00:00
VARIANT=$1
EP=${2:-300}
SEED=${3:-7}
DATASET=${4:-tsa15}
source /work/home/spotmaverick/nam_probe/run_env.sh
cd /work/home/spotmaverick/nam_project
mkdir -p arch_v2
if [ "$DATASET" = "tsa15" ]; then
  LOG="arch_v2/${VARIANT}_s${SEED}.log"
else
  LOG="arch_v2/${VARIANT}_${DATASET}_s${SEED}.log"
fi
python arch_search_v2.py "$VARIANT" "$EP" "$SEED" "$DATASET" > "$LOG" 2>&1
echo "exit=$? $(tail -1 "$LOG")"
