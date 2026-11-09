#!/usr/bin/env bash
# Runs every RL run of one phase 3 experiment, a few at a time. Finished runs are skipped and
# interrupted ones resume, so the script can simply be relaunched.
# Usage (from the repo root): bash scripts/run_phase3.sh 3.1a [python]      JOBS=3 by default
set -euo pipefail
EXP=$1
PY=${2:-python}
export PYTHONPATH=src

runs() {  # one line of phase3.rl arguments per run
  case "$EXP" in
    3.1a) for s in 0 1 2 3 4; do for b in 1 0; do
            echo "--seed $s --set rl.max_rl_steps=2000 rl.objective_params.beta=$b"; done; done ;;
    *) echo "unknown experiment $EXP" >&2; exit 1 ;;
  esac
}

runs | xargs -P "${JOBS:-3}" -L 1 "$PY" -m phase3.rl --exp "$EXP"
"$PY" -m phase3.analysis compact "$EXP"
