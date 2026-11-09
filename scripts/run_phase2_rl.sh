#!/usr/bin/env bash
# Runs all phase-2 RL runs sequentially (2.3a wd1 tau=1, 2.3b positive-only, 2.3c tau sweep), 3 seeds each.
# Each run is resumable (state in ~/dllm-cache/phase2/rl/<name>/state.pt); finished runs are skipped.
# Usage: bash scripts/run_phase2_rl.sh [python]   (from the repo root)
set -euo pipefail
PY=${1:-python}
export PYTHONPATH=src
run() {  # variant tau seed
  local name="$1_tau$2_seed$3"
  if [[ -f "results/phase2/rl/${name}.json" ]]; then echo "skip ${name}"; return; fi
  echo "=== ${name} $(date +%T)"
  "$PY" -m phase2.rl_wd1 --variant "$1" --tau "$2" --seed "$3"
}
for s in 0 1 2; do run wd1 1 "$s"; run positive_only 1 "$s"; done   # 2.3a, 2.3b
for s in 0 1 2; do run wd1 0.5 "$s"; run wd1 2 "$s"; done           # 2.3c (tau=1 shared with 2.3a)
echo "all done $(date +%T)"
