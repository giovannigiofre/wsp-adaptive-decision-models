#!/usr/bin/env bash
# Runs every training run of one phase 3 experiment, JOBS at a time. Finished runs are skipped and
# interrupted RL runs resume, so the script can simply be relaunched.
# Usage (from the repo root): bash scripts/run_phase3.sh 3.1b [python]
# JOBS=3 by default: on this Mac three 2000-step RL runs in parallel took about 7 min (2.4 min per run),
# against 3.7 min per run in sequence. Timings in the run summaries are therefore taken under contention.
set -euo pipefail

main() {
  EXP=$1
  PY=${2:-python}
  export PYTHONPATH=src
  # Standard horizon fixed by 3.1a, best temperature fixed by 3.1b (see their READMEs).
  H=$(jq -r .rule_horizon results/phase3/3.1a/horizon.json 2>/dev/null || echo 2000)
  TS=$(jq -r .best_tau results/phase3/3.1b/best_tau.json 2>/dev/null || echo 0.1)
  SFT=~/dllm-cache/phase2/sft
  SUB=~/dllm-cache/phase3/3.1d/sft
  local rl="-m phase3.rl --exp $EXP" hs="rl.max_rl_steps=$H" op="rl.objective_params"

  runs() {  # one line of arguments per run
    case "$EXP" in
      3.1a) for s in 0 1 2 3 4; do for b in 1 0; do
              echo "$rl --seed $s --set rl.max_rl_steps=2000 $op.beta=$b"; done; done ;;
      3.1b) for s in 0 1 2; do for t in 1 0.25 0.1 0.05; do for b in 1 0; do
              echo "$rl --seed $s --set $hs $op.tau=$t $op.beta=$b"; done; done; done ;;
      3.1c) for s in 0 1 2; do for t in 1 "$TS"; do for b in 0.25 0.5 2; do   # beta 0 and 1 are in 3.1b
              echo "$rl --seed $s --set $hs $op.tau=$t $op.beta=$b"; done; done
              for f in -12 -15; do echo "$rl --seed $s --set $hs $op.floor=$f"; done; done ;;
      3.1d) for s in 0 1 2; do for b in 1 0; do     # the step-120 start (val 0.42) is in 3.1b
              for i in 100 130; do
                echo "$rl --seed $s --init $SFT/step_000$i.pt --tag _sft$i --set $hs $op.beta=$b"; done
              echo "$rl --seed $s --init $SUB/$(cat results/phase3/3.1d/chosen_subset.txt).pt --tag _subset --set $hs $op.beta=$b"
            done; done ;;
      3.1d-sft) for n in 2 5 10 20 40; do echo "-m phase3.sft_subset --n-solutions $n"; done ;;
      3.1e) for s in 0 1 2; do for t in 1 "$TS"; do for b in 1 0; do          # binary reward is in 3.1b
              echo "$rl --seed $s --set $hs rl.reward=graded $op.tau=$t $op.beta=$b"; done; done; done ;;
      3.1f) for s in 0 1 2; do for m in 0.04 0.2 0.5; do                      # no reference is in 3.1b
              echo "$rl --seed $s --set $hs rl.ref_mix=$m"; done; done ;;
      3.2)  for s in 3 4; do                                  # wd1 and positive-only, seeds 0-2, are in 3.1b
              echo "$rl --seed $s --set $hs"
              echo "$rl --seed $s --set $hs $op.beta=0"; done
            for s in 0 1 2 3 4; do
              echo "$rl --seed $s --set $hs rl.objective=reinforce"
              echo "$rl --seed $s --set $hs rl.objective=reinforce $op.normalize=true"
              echo "$rl --seed $s --set $hs rl.objective=filtered"
              echo "$rl --seed $s --set $hs rl.objective=self_diffusion"
              echo "$rl --seed $s --set $hs rl.objective=grpo"
              echo "$rl --seed $s --set $hs rl.objective=supervised"
            done ;;
      3.3)  for s in 0 1 2; do for p in last first; do for m in rl probe; do
              echo "-m phase3.conf_token train --exp 3.3 --seed $s --place $p --mode $m --set $hs"; done; done; done ;;
      3.4b) for s in 0 1 2 3 4; do for f in probs hidden; do for i in zero confidence; do
              echo "-m phase3.order_policy train --exp 3.4b --seed $s --features $f --init $i"; done; done; done
            for s in 0 1 2; do for f in probs hidden; do
              echo "-m phase3.order_policy train --exp 3.4b --seed $s --features $f --init zero --teacher oracle"; done; done ;;
      *) echo "unknown experiment $EXP" >&2; exit 1 ;;
    esac
  }

  runs | xargs -P "${JOBS:-3}" -L 1 "$PY"
  case "$EXP" in 3.1[a-f]|3.2) "$PY" -m phase3.analysis compact "$EXP" ;; esac
}

main "$@"
