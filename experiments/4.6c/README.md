# 4.6c Regime map for the push-down term of wd1

**Status:** done (2026-11-11). Setup and predictions were committed before the runs; results are appended at the end.

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

## Results (2026-11-11)

Wall-clock: R1 to R3 00:17 to 01:37 (80 min), R4 and R5 02:04 to 02:55, R6 to R8 02:54 to 03:32, test evaluation 7 min; the audit of 4.6a and other jobs shared the machine, so times are upper bounds. Apple M5 24 GB, MPS, torch 2.15.0, float32, seeds 0 to 2 (MPS is not bit-deterministic). No thermal warning at any launch change. Hard test band, 667 puzzles, final checkpoints (step 500), greedy decoding; seeds pooled per puzzle, 95% bootstrap over puzzles of the paired difference. Tables: `results/phase4/4.6c/arms.csv`, `arm_pairs.csv` (`python -m phase4.rl_analysis arms 4.6c --pairs ...`), per puzzle in `test_solved.csv`.

Start: 0.168 at 2 sampler steps, 0.535 at 4, 0.814 at 16. Reference rows of 4.4 at 4 steps: self-training 0.602, positive-only 0.590, plain wd1 0.505.

| arm | test at the trained budget | minus start | at 16 steps, minus start |
|---|---|---|---|
| R1 batch weights, β = 0 | 0.610 | +0.074 [+0.046, +0.101] | +0.032 [+0.008, +0.056] |
| R2 wd1, 1 inner step | 0.535 | −0.000 [−0.022, +0.021] | +0.000 [−0.016, +0.018] |
| R3 wd1, paper optimiser values | 0.537 | +0.001 [−0.020, +0.023] | +0.005 [−0.013, +0.022] |
| R4 dense reward, wd1 | 0.483 | −0.052 [−0.079, −0.024] | −0.049 [−0.073, −0.025] |
| R5 dense reward, β = 0 | 0.584 | +0.048 [+0.021, +0.074] | +0.022 [−0.000, +0.043] |
| R6 2 steps, wd1 | 0.168 (2 steps) | −0.000 [−0.021, +0.018] | −0.005 [−0.024, +0.013] |
| R7 2 steps, β = 0 | 0.215 (2 steps) | +0.047 [+0.023, +0.071] | +0.016 [−0.006, +0.038] |
| R8 2 steps, filtered imitation | 0.223 (2 steps) | +0.055 [+0.033, +0.079] | +0.017 [−0.004, +0.039] |

Paired differences between arms (at 4 sampler steps unless stated):

| a − b | difference |
|---|---|
| R1 − batch weights, β = 1, floored (4.5a) | +0.030 [+0.008, +0.052] |
| R1 − positive-only with group weights (4.4) | +0.020 [+0.003, +0.036] |
| R1 − self-training (4.4) | +0.007 [−0.009, +0.023] |
| R2 − plain wd1 (4.4) | +0.030 [+0.010, +0.051] |
| R3 − plain wd1 (4.4) | +0.032 [+0.012, +0.054] |
| R2 − self-training; R3 − self-training | −0.067 [−0.094, −0.039]; −0.065 [−0.092, −0.038] |
| R4 − R5 (dense reward: wd1 − positive-only) | −0.100 [−0.128, −0.074] |
| R4 − plain wd1 (4.4) | −0.021 [−0.041, −0.002] |
| R5 − positive-only (4.4) | −0.006 [−0.023, +0.010] |
| R6 − R7 (2 steps: wd1 − positive-only), at 2 steps | −0.047 [−0.068, −0.025] |
| R6 − R8, at 2 steps | −0.055 [−0.075, −0.035] |
| R7 − R8, at 2 steps | −0.008 [−0.021, +0.004] |

The models trained with 2-step rollouts, read at 4 steps: R6 +0.017 [−0.004, +0.040], R7 +0.060 [+0.033, +0.085], R8 +0.070 [+0.046, +0.096] over the start.

## Predictions against results

1. R1 equals batch weights with β = 1: **not held.** Without the push-down term batch weights gain 0.030 more, interval above zero. The push-down costs here too.
2. R2 and R3 do not fall below the start: **held.** Both end at the start (0.000 and +0.001) and above plain wd1 of 4.4 by 0.03. Neither gains: both are 0.065 below self-training. R2 also makes four times fewer updates, so "on-policy" and "less training" are not separated.
3. Dense reward, wd1 above positive-only: **not held, opposite sign.** R4 − R5 = −0.100; R4 is the lowest arm of phase 4 at 4 steps (0.483) and also loses at 16 steps. The second part fell too: R5 is not within ±0.02 of the start, it gains +0.048, the same as positive-only with the binary reward (difference −0.006 [−0.023, +0.010]).
4. 2 sampler steps, gap less negative, filtered above the start: **first part held in the point estimate only** (−0.047 at 2 steps against −0.085 at 4; the difference between the two gaps was not tested, and relative to a start of 0.168 the 2-step gap is not smaller). **Second part held** (+0.055).
5. No arm beats self-training by more than its interval: **held.** R1 has the highest mean of phase 4 (0.610) but its difference from self-training is +0.007 [−0.009, +0.023].

## What this shows / does not show

- Shows: none of the five factors moved one at a time (batch weights, one inner step, the paper's optimiser values, dense reward, lower success rate through 2 sampler steps) changes the sign of "wd1 minus positive-only" on this model; it stays negative in every pair measured (−0.030, −0.100, −0.047). Smaller or fewer updates bring plain wd1 back to the start, not above it. Every arm without the push-down term gains 0.05 to 0.07 at its trained budget.
- Does not show: that the paper's ablation is wrong. Still different from the paper: a 1.86M-parameter model trained from scratch against a pretrained 8B model with LoRA, 9×9 puzzles from our generator, 500 steps, group of 8, τ = 1 with a reward in [0, 1] (nearly flat weights in R4 and R5, so R5 is close to imitating all its own rollouts and its gain is not explained here). One factor at a time, no interactions; R2 confounds on-policy with four times fewer updates; R3 is not LoRA. Seeds are pooled per puzzle, so intervals cover puzzle sampling, not seed-to-seed variance (three seeds).

