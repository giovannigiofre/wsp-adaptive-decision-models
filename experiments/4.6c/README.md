# 4.6c Regime map for the push-down term of wd1

**Status:** running. Setup and predictions are committed before the runs; results are appended at the end.

**Goal.** In 4.4 plain wd1 ended below the start and positive-only above it, the opposite of the ablation in the wd1 paper (arXiv 2507.08838). `docs/RL_DLLM_DIRECTIONS.md` (direction 2, section 2.4) lists what differs between the paper's setting and ours. Here one factor is moved at a time, to see whether any of them changes the sign of "wd1 minus positive-only". This is a screening design: one arm per factor, no interactions; anything that flips needs a second run.

| arm | change against 4.4 (wd1, β = 1, τ = 1, binary reward, 4 sampler steps, 4 inner steps, lr 1e-5, clip 1.0) | compared with |
|---|---|---|
| R1 | weights over the batch, β = 0 | batch weights with β = 1 and floor (4.5a): is the gain of batch weights pull-up only? |
| R2 | 1 inner step (on-policy; 4 times fewer updates) | plain wd1 of 4.4 |
| R3 | lr 3e-6, clip 0.2, weight decay 0.1 (the paper's optimiser values; still full fine-tuning, not LoRA) | plain wd1 of 4.4 |
| R4 | dense reward: share of hidden cells equal to the stored solution (**uses labels**; the paper's Sudoku reward) | R5 |
| R5 | dense reward, β = 0 | R4 |
| R6 | 2 sampler steps in rollouts and evaluation | R7, R8 |
| R7 | 2 sampler steps, β = 0 | R6 |
| R8 | 2 sampler steps, filtered imitation | R6 |

Start at 2 sampler steps, measured before the runs (hard band, greedy): test 0.168, val 0.177; at temperature 1, 12% of rollouts are valid and 19 to 23% of groups of 8 have unequal rewards. This is inside the 0.02 to 0.30 range set in the memo for running R6 to R8.

Everything else as in 4.4 (`configs/phase4_rl.yaml`): hard band, 32 puzzles × 8 rollouts, 500 steps, seeds 0 to 2, start = plateau model of 4.1. New code: `phase3.rewards.cell_accuracy`, `rl.weight_decay`, summary fields; tests in `tests/test_phase4_regime.py`.

**Command**
```
for arm in r1 r2 r3 r4 r5 r6 r7 r8; do for s in 0 1 2; do SEED=$s bash scripts/run_phase4.sh rl46-$arm ~/dllm-cache/venv/bin/python & done; wait; done
```
Three launches (r1 to r3, r4 and r5, r6 to r8), each expected under 2 hours. Hardware: Apple M5, 24 GB, torch 2.15.0, MPS, fp32. Logs and checkpoints in `~/dllm-cache/phase4/4.6c/`.

## Predictions (written before the runs)

Seen before writing: a 4-step smoke run of R3, R4 and R8 (no solve rates beyond the start), and the 2-step start rate above. In the R4 smoke run the group weights were almost flat (effective sample size 7.96 of 8, because rewards lie in [0, 1] and τ = 1) and about 40% of groups had equal rewards.

1. **R1 equals batch weights with β = 1** (4.5a: 0.579 at 4 steps): paired difference on hard test with an interval that includes zero. The push-down of all-invalid groups adds nothing. Confidence: medium.
2. **R2 and R3 do not fall below the start**: their difference from the start has an interval that includes zero or lies above it (plain wd1 of 4.4: −0.031 [−0.058, −0.004]). Confidence: medium for R3 (smaller updates), low for R2.
3. **Dense reward: wd1 above positive-only** (R4 − R5 > 0, interval excluding zero), the direction of the paper. With flat weights positive-only imitates all its own rollouts, valid or not, so R5 stays within ±0.02 of the start; wd1 becomes close to a policy gradient with a linear advantage. Confidence: medium for the sign, low for the interval.
4. **2 sampler steps: the gap wd1 − positive-only is less negative than at 4 steps** (4.4: 0.505 − 0.590 = −0.085), and filtered imitation ends above the 2-step start. Confidence: low for the first part (3.6b on 4×4 points this way, at a much lower success rate), medium for the second.
5. **No arm of R1 to R8 beats self-training of 4.4** (+0.067 over the start at 4 steps) by more than its interval, for the arms evaluated at 4 steps.

Deciding measurement: hard test solve rate at the trained sampler budget at step 500, paired over the 667 test puzzles with seeds pooled, 95% bootstrap intervals over puzzles, as in 4.4; 16 steps next to it.

## What this shows / does not show

To be written with the results.
