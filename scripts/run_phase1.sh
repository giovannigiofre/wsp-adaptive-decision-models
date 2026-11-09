#!/bin/bash
# Phase 1 overnight pipeline (1.1 then the 1.2 grid). Every stage is resumable: rerun the script
# after an interruption and finished rows are skipped. Logs go to ~/dllm-cache/logs/.
# Sizes chosen from the 1.0 speed measurement (see experiments/1.2/README.md).
set -u
cd "$(dirname "$0")/.."
PY=~/dllm-cache/venv/bin/python
export HF_HOME=~/dllm-cache/hf HF_HUB_OFFLINE=1
L=~/dllm-cache/logs
mkdir -p $L
stamp() { date "+%F %T"; }

if [ ! -f results/phase1/unmasking/runs.csv ]; then
  echo "$(stamp) start 1.1" >> $L/pipeline.log
  $PY -m src.phase1.unmasking --n_per_task 3 --gen_length 128 --steps 64 > $L/unmasking.log 2>&1
  echo "$(stamp) end 1.1 rc=$?" >> $L/pipeline.log
fi
for spec in "sudoku ${N_SUDOKU:-12} 64" "countdown ${N_COUNTDOWN:-12} 64" "gsm8k ${N_GSM8K:-5} 128"; do
  set -- $spec
  echo "$(stamp) start 1.2 $1 n=$2 gen=$3" >> $L/pipeline.log
  $PY -m src.phase1.run_grid --task $1 --n $2 --gen_length $3 >> $L/grid_$1.log 2>&1
  echo "$(stamp) end 1.2 $1 rc=$?" >> $L/pipeline.log
done
echo "$(stamp) pipeline done" >> $L/pipeline.log
