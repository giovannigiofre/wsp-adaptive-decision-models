# 4.4b Three measurements on the plateau model before more RL (no training)

**Goal:** answer three questions that decide which of the proposals in `docs/RL_DLLM_SURVEY.md` are worth a training run (proposal 1 there). (a) Do wrong cells follow an earlier mistake, and are the first mistakes at 4 steps two cells committed together that clash? (b) Does a sampled unmasking order give more groups with unequal rewards than the fixed order? (c) Are valid rollouts on train puzzles just the stored solution?

**Command**
```
PYTHONPATH=src python -m phase4.pre_rl_checks --size medium
```
- Config `configs/phase4.yaml` (`sizes.medium`), model `~/dllm-cache/phase4/sft/sft/final.pt` (step 43,500), hard band.
- (a) greedy decoding (`confidence` order, temperature 0) of the 667 hard test puzzles at 4 and 16 sampler steps.
- (b), (c) the 512 hard train puzzles of step 4.3 (seed 0), 8 rollouts each, digits sampled at temperature 1, 4 sampler steps. Order: the cells to commit are the top k of `log(confidence) / τ_o + Gumbel noise`, which is a draw without replacement with weights `confidence^(1/τ_o)`; τ_o = 0 is the fixed order used so far.
- Hardware: Apple M5, 24 GB, macOS 27.0.1, torch 2.15.0, MPS, fp32, seed 0 (MPS is not fully deterministic). Wall-clock 2 minutes.
- Results: `results/phase4/4.4b/error_structure.csv`, `error_rate_by_step.csv`, `order_noise.csv`.

**Predictions** are the ones in the survey (commit `32ce7ed`, written before this script existed). The survey listed τ_o in {0.25, 0.5, 1, 2}; 0.02, 0.05 and 0.1 were added when the script was written, before any result.

**A failed first run, kept for the record.** The first run had a bug in the Gumbel noise (a misplaced clamp made every noise value NaN, so the order was arbitrary): valid rollouts were about 0.01 at every τ_o, including 0.02 where the order should be nearly the fixed one. That mismatch exposed it. Fixed, with two tests (noise is finite with the right mean and spread; a very small τ_o gives exactly the fixed order). Parts (a) and (c) at τ_o = 0 did not use the noise and did not change.

## (a) Errors come after an earlier error; direct clashes are rare

| sampler steps | failed puzzles | wrong cells per failed puzzle | at the first-error step | after it | share after | first errors that clash with a cell committed in the same step | failed puzzles with at least one such clash |
|---|---|---|---|---|---|---|---|
| 4 | 310 / 667 | 16.3 | 1.9 | 14.4 | 0.88 | 0.083 | 0.129 |
| 16 | 124 / 667 | 21.3 | 1.3 | 19.9 | 0.94 | 0.006 | 0.008 |

Error rate per committed cell, by whether the puzzle already has a wrong cell (`error_rate_by_step.csv`):

| sampler steps | step | no earlier error | after an earlier error |
|---|---|---|---|
| 4 | 0 | 0.026 | - |
| 4 | 1 | 0.039 | 0.341 |
| 4 | 2 | 0.012 | 0.432 |
| 4 | 3 | 0.002 | 0.554 |
| 16 | 0 to 15 | 0.000 to 0.014 | 0.33 to 0.72 |

- A wrong cell never repeats a digit that is already visible in its row, column or box (0 cases at both budgets). All solved grids equal the stored solution.
- **Prediction "more than half of wrong cells come after the first-error step at 16 steps": held (0.94).** A failed puzzle has one or two wrong cells at its first-error step and about 20 by the end; once one cell is wrong, the next commits are wrong 33% to 72% of the time, against about 1% before.
- **Prediction "a large share of first errors at 4 steps are same-step clashes": not held (0.083 of cells, 13% of failed puzzles).** The first error at 4 steps is mostly a cell that is wrong on its own, committed at step 0 or 1 (mean first-error step 0.68). At 4 steps the first step commits about 14 cells, and its error rate is 0.026 per cell against 0.001 for the 4 cells committed first at 16 steps: the extra cells are ones the model is less sure of.

## (b) A little order noise doubles the groups with unequal rewards

| τ_o | valid rollouts | groups with unequal rewards | all valid | none valid |
|---|---|---|---|---|
| 0 (fixed order) | 0.549 | 0.264 | 0.436 | 0.301 |
| 0.02 | 0.558 | 0.406 | 0.344 | 0.250 |
| 0.05 | 0.538 | 0.545 | 0.244 | 0.211 |
| 0.1 | 0.460 | 0.646 | 0.139 | 0.215 |
| 0.25 | 0.256 | 0.639 | 0.023 | 0.338 |
| 0.5 | 0.112 | 0.402 | 0.000 | 0.598 |
| 1 | 0.052 | 0.250 | 0.000 | 0.750 |
| 2 | 0.024 | 0.137 | 0.000 | 0.863 |

- **Prediction "some τ_o gives more than 0.40 unequal groups with a valid share above 0.35": held** at 0.02, 0.05 and 0.1. It would not have held on the survey's own grid: from 0.25 up the valid share falls below 0.35.
- At τ_o = 0.05 the valid share is the same as with the fixed order (0.54 against 0.55) while unequal groups go from 0.26 to 0.54. Groups where no rollout is valid also drop (0.30 to 0.21): some puzzles that the fixed order never solves in 4 steps are solved with another order.
- One draw of 512 puzzles, no interval; the 4.3 run of the same measurement gave 0.560 / 0.244 for the fixed order, so differences of about 0.02 are noise.

## (c) Every valid rollout is the stored solution

Share of valid rollouts equal to the stored solution: **1.000** at every τ_o (2,250 valid rollouts at τ_o = 0). **Prediction (above 0.99): held.** The puzzles have one solution, so "the model's own valid rollouts" as training targets are the dataset's targets for the subset of puzzles the model already solves in 4 steps.

## What this decides (by the rules written in the survey)

- (b) held, so the factorial of sampled order and baseline is worth running. τ_o = 0.05 keeps the valid share; 0.1 gives the most unequal groups at a cost of 0.09 in valid share.
- (a) at 4 steps: few direct clashes, so the survey's rule says proposal 4 (training on the states the 4-step sampler visits) loses its stated reason, and undoing early wrong commits (proposal 6, remasking) gains one: one early wrong cell is followed by about 14 to 20 more.
- (c) the gain of self-training and filtered imitation in 4.4 (+0.05 to +0.07 at 4 steps) did not come from new targets. The survey predicted a small gain for them on this ground; the gain was the largest of 4.4. What it comes from (which puzzles are selected, how their cells are masked) is not measured here.

**Shows:** on this model, failures are one or two early wrong cells followed by many wrong cells that fit them; the 4-step budget fails mostly through cells that are wrong on their own, not through pairs that clash; order noise with small τ_o gives unequal-reward groups without losing valid rollouts.
**Does not show:** that training with a sampled order helps (no gradient step here). The clash test only sees two cells of one step with the same digit in a row, column or box; cells committed together that are each allowed but cannot both be right are counted as "wrong on its own". "After an earlier error" does not prove the earlier error caused the later ones: failed puzzles may simply be the harder ones. One seed, one sample of puzzles.
