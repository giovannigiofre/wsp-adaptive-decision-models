# 4.8e: the givens-only control continued to the step count of the supervised plateau

**Status: done (2026-11-12). Setup and predictions were committed before the run (`3112cb7`).** Seen before writing them: all results of [4.8b](../4.8b/README.md), [4.8c](../4.8c/README.md) and [4.8d](../4.8d/README.md).

**Goal.** 4.8d: the label-free hindsight loop flattens at about 0.73 on the hard val band at 16 sampler steps by step 43,546 (0.45 at 4 steps, 0.84 at 81). Its control in 4.8b, training on the givens only (no harvest, no checker), had 0.650 / 0.210 / 0.689 at 16 / 4 / 81 steps at step 19,546 and was still rising by about 0.02 per round. Question: how much of the long hindsight result does the givens-only loss reach on its own with the same number of steps?

**Setup.** The seed-0 `givens` run of 4.8b is continued unchanged from its saved state (step 19,546, round 10) to round 34, step 43,546. Same settings as 4.8d (`configs/phase4.yaml`, batch 256, lr 3e-4, 1,000 steps per round); no rollouts are harvested. Per round: val solve per band at 16 and 4 sampler steps. At the end: valid share at 4, 16, 81 steps on the 1,002 val puzzles, and the paired difference `hindsight_long` − `givens_long` per val puzzle with a 95% bootstrap over puzzles.

**Fixed before the run:** the stop is round 34. Comparison rows: 4.8d (`hindsight_long`) and supervised 4.1, no rerun.

**Known limits, before the run.** One seed, one run per arm; single rounds move by ±0.02 to 0.03. Equal training steps, not equal wall-clock or forward passes (the hindsight loop also spends rollouts). Val only.

## Predictions, written before the run

1. **Hard at 16 steps at the end** (mean of the last three rounds): 0.68 to 0.74. The control keeps rising and ends within 0.05 of the hindsight loop (0.730), below or equal to it.
2. **Hard at 4 steps at the end:** 0.22 to 0.32, clearly below the hindsight loop (0.450); the paired difference at 4 steps has an interval above +0.10.
3. **Hard at 81 steps at the end:** 0.72 to 0.80, below the hindsight loop (0.841) with an interval above zero.
4. **Easy 1.000 and medium at 16 steps 0.98 or above** at the end.
5. **Slowing:** gain on hard at 16 steps from rounds 8-10 to rounds 20-22 larger than from rounds 20-22 to rounds 32-34.

**Results that would change the reading of 4.8, reported as such:** the control within the interval of the hindsight loop at 16 and 81 steps (then at long training the harvest adds only few-step quality); or the control above it.

## Command

```
bash scripts/run_phase4.sh self-a2-ctrl-long ~/dllm-cache/venv/bin/python
PYTHONPATH=src python -m phase4.selfloop_pairs --long        # results/phase4/4.8e/long_pairs.csv
```

Each launch pauses after 100 minutes at the next round end; repeat until it prints "finished". Results: `results/phase4/4.8e/givens_long.csv` (rounds 1 to 10 copied from 4.8b), `.json`, `_valid_val.csv`, `_solved_val.csv`. Checkpoints under `~/dllm-cache/phase4/4.8e/`.

## Results (2026-11-12, 01:00 to 03:48)

Two launches (01:00 to 02:41, paused after round 24; 02:41 to 03:48), 2 h 48 min for rounds 11 to 34 with the final evaluation. Apple M5 24 GB, MPS, torch 2.15.0, float32, seed 0 (MPS is not bit-deterministic). Unrelated jobs loaded the machine for part of the time; wall-clock is not a clean timing. No thermal warning at either launch change.

Hard val band (334 puzzles), solve rate per round, three-round means (`givens_long.csv`; hindsight from 4.8d):

| rounds | steps | `givens_long` at 16 steps | `hindsight_long` at 16 steps | `givens_long` at 4 steps | `hindsight_long` at 4 steps |
|---|---|---|---|---|---|
| 8 to 10 | 17,546 to 19,546 | 0.626 | 0.698 | 0.202 | 0.355 |
| 20 to 22 | 29,546 to 31,546 | 0.705 | 0.744 | 0.317 | 0.434 |
| 32 to 34 | 41,546 to 43,546 | 0.755 | 0.730 | 0.396 | 0.450 |

The control's highest single round at 16 steps is the last one (0.781). Supervised 4.1 over its last three evaluations: 0.811 at 16 steps.

Final models (step 43,546), valid share, and the paired difference per val puzzle with a 95% bootstrap over puzzles (`givens_long_valid_val.csv`, `long_pairs.csv`):

| hard band | 4 steps | 16 steps | 81 steps |
|---|---|---|---|
| `givens_long` | 0.404 [0.353, 0.458] | 0.781 [0.737, 0.826] | 0.847 [0.808, 0.883] |
| `hindsight_long` | 0.452 | 0.731 | 0.841 |
| `hindsight_long` − `givens_long` | +0.048 [−0.009, +0.102] | −0.051 [−0.096, −0.006] | −0.006 [−0.051, +0.039] |

Medium band, final: `givens_long` 0.937 / 0.988 / 0.991 at 4 / 16 / 81 steps; differences from `hindsight_long` +0.012, +0.009, +0.009, intervals including or touching zero. Easy: 1.000 at the end at every budget (two early rounds of the 4.8b part at 0.994 and 0.997 at 16 steps).

## Predictions against results

1. Hard at 16 steps, mean of the last three rounds 0.68 to 0.74, at or below the hindsight loop: **not held.** 0.755, above the hindsight loop's 0.730; in the final models the control is ahead by 0.051 with an interval just below zero.
2. Hard at 4 steps 0.22 to 0.32, paired difference with an interval above +0.10: **not held.** 0.396 (final 0.404); the hindsight loop is ahead by +0.048 with an interval that includes zero.
3. Hard at 81 steps 0.72 to 0.80, below the hindsight loop: **not held.** 0.847, equal to the hindsight loop (−0.006 [−0.051, +0.039]).
4. Easy 1.000 and medium at 16 steps 0.98 or above at the end: **held** (0.988).
5. Slowing: **held** (+0.079 from rounds 8-10 to 20-22, +0.050 to rounds 32-34), but unlike the hindsight loop the control was still rising at the stop.

The result named before the run as one that would change the reading of 4.8 occurred: at 43,546 steps the givens-only control is within the interval of the hindsight loop at 81 steps and above it at 16.

## What this shows / does not show

- Shows: with the same number of training steps as the supervised plateau, training on the givens alone (no rollouts, no checker, no solutions) reaches 0.76 to 0.78 on the hard val band at 16 sampler steps and 0.85 at 81, as good as or better than the hindsight loop that harvested 98.7% of the hard train solutions (0.73, 0.84). The advantages of the harvest arms read in 4.8b and 4.8c at 19,546 steps (+0.05 at 16 steps, +0.11 at 81, +0.15 at 4) were a head start, not a higher end point: by step 43,546 only the 4-step advantage is left, and it is +0.05 with an interval that includes zero. Predictions 1 to 3, written from the short runs, were all wrong in the same direction.
- Does not show: seed variance (one run per arm; single rounds move by ±0.02 to 0.03, and the final round is the control's best and one of the hindsight loop's lower ones, so the 16-step difference of the final models overstates the three-round means, 0.755 against 0.730); where either arm ends (the control was still rising, the hindsight loop was flat from about round 20); why the hindsight loop flattens lower (its batches are 75% buffer rows, so it sees a quarter of the givens-loss steps; not separated); anything on the test set, which is the next step.
