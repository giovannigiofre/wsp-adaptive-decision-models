# 4.9d: the two long runs continued with their own training at a low learning rate

**Status: done (2026-11-12). Setup and predictions were committed before the run (`0c67774`).** Seen before writing them: everything up to [4.9c](../4.9c/README.md).

**Goal.** 4.9b and 4.9c: 500 steps at lr 1e-5 on the model's own valid grids for hard puzzles lift both long label-free models from about 0.43 to 0.52 on hard test at 4 sampler steps; the same pass at lr 3e-4 loses. 4.9c could not say whether the learning rate is why the long loops (4.8d, 4.8e, constant 3e-4 for 43,546 steps) stopped where they did, because that arm also differed from the loops in batches, band and optimiser restart. Question here, one change against the loops themselves: if `hindsight_long` and `givens_long` simply go on with their own training and only the learning rate drops from 3e-4 to 1e-5, do they move?

**Setup.** Each run is resumed from its saved state at round 34, step 43,546 (model, AdamW moments, buffer, RNG) and continued for 4 rounds of 1,000 steps to step 47,546 with `--lr 1e-5` (new option of `phase4.selfloop`: sets the learning rate of the loaded optimiser and scheduler, moments kept). Everything else as in 4.8d / 4.8e: `configs/phase4.yaml`, batch 256, all bands, 16-step greedy harvest of up to 15,000 unsolved train puzzles per round for the hindsight arm, 25% givens batches.

| run | start | arm |
|---|---|---|
| `hindsight_lr1e-5` | `4.8d/hindsight_long/state.pt` | `hindsight` (buffer of self-found grids, hindsight masks) |
| `givens_lr1e-5` | `4.8e/givens_long/state.pt` | `givens` (givens loss only, checker unused) |

No stored solution enters any loss. 4,000 updates at batch 256 per run; the 500-step arms of 4.9b/c made 2,000 updates at batch 256 on hard puzzles only.

**Evaluation.** Per round: hard val band at 16 and 4 sampler steps (the log also holds rounds 28 to 34 at lr 3e-4 for the round-to-round noise). Final models: valid share on the test split at 4, 16, 81 steps, greedy; on the hard band (667 puzzles) paired differences with a 95% bootstrap over puzzles against each run's start, between the two runs, and against the 500-step arms from the same starts (`hind` of 4.9c, `self4` of 4.9b, seeds pooled per puzzle).

**Known limits, before the run.** One run per arm, one seed, one learning rate, a step change with no decay schedule; the intervals cover puzzle sampling only. There is no arm that continues at 3e-4 for the same 4 rounds: the reference for "no change" is the plateau of rounds 28 to 34 on val (single rounds move by ±0.02 to 0.03) and the start model on test. Test is read only at the end (4,000 updates), not at 2,000. The test band was used before.

## Predictions, written before the run

1. **`givens_lr1e-5` at 4 steps:** within ±0.03 of its start on hard test (0.438), interval including zero. The `ctrl` arm of 4.9b (givens loss at 1e-5 on hard puzzles) moved by +0.014.
2. **`hindsight_lr1e-5` at 4 steps:** +0.02 to +0.06 over its start (0.423). I expect less than the 500-step pass: three quarters of its batches are buffer grids of all bands with hindsight masks, not fresh grids for hard puzzles. Low confidence in the lower bound; an interval that includes zero would not surprise me.
3. **`hindsight_lr1e-5` stays below `hind` of 4.9c at 4 steps** (0.516): difference −0.02 to −0.08.
4. **16 steps:** both runs between 0.00 and +0.04 over their starts (0.777, 0.780). Removing the step noise of lr 3e-4 should not hurt; the 500-step arms did not move at 16 steps.
5. **Val noise:** over the four rounds at 1e-5 the hard val rate at 16 steps spans less than 0.03 in each run (rounds 28 to 34 at 3e-4 spanned about 0.05).
6. **Train loss falls in both runs** in the first round at 1e-5, by more than it moved over rounds 31 to 34.

**Results that would be reported as such:** either run gains as much as the 500-step pass (then the learning rate alone, with the loop's own data, explains the plateau at 4 steps); neither run moves at any budget (then the 500-step gain comes from what that pass trains on, not from the learning rate).

## Command

```
bash scripts/run_phase4.sh self49d-hind ~/dllm-cache/venv/bin/python      # the two side by side
bash scripts/run_phase4.sh self49d-givens ~/dllm-cache/venv/bin/python
PYTHONPATH=src python -m phase4.finish_pairs --lowlr                      # results/phase4/4.9d/pairs.csv, val_rounds.csv
```

Expected wall-clock: about 30 to 40 min per run (4 rounds, then val and test evaluation at 4 / 16 / 81 steps). Results: `results/phase4/4.9d/<run>.csv` (rounds 1 to 34 copied from 4.8d / 4.8e), `.json`, `_valid_{val,test}.csv`, `_solved_{val,test}.csv`. Checkpoints under `~/dllm-cache/phase4/4.9d/`.

## Results (2026-11-12, 15:05 to 16:03)

The two runs side by side, 58 min with the val and test evaluations (estimate was 30 to 40 min; two MPS jobs at once). Apple M5 24 GB, MPS, torch 2.15.0, float32, seed 0 (MPS is not bit-deterministic). No thermal warning at launch or at the end.

Hard test band (667 puzzles), valid share, paired difference to each run's own start with a 95% bootstrap over puzzles (`results/phase4/4.9d/pairs.csv`):

| hard test | 4 steps | 16 steps | 81 steps |
|---|---|---|---|
| `givens_long` (start) | 0.438 | 0.780 | 0.850 |
| `givens_lr1e-5` | 0.495 (+0.057 [+0.024, +0.090]) | 0.807 (+0.027 [−0.002, +0.054]) | 0.895 (+0.045 [+0.023, +0.068]) |
| `hindsight_long` (start) | 0.423 | 0.777 | 0.835 |
| `hindsight_lr1e-5` | 0.472 (+0.050 [+0.020, +0.078]) | 0.783 (+0.006 [−0.021, +0.035]) | 0.870 (+0.035 [+0.009, +0.062]) |
| supervised 4.1 (constant lr 3e-4, no low-lr rounds) | 0.535 | 0.814 | 0.871 |

Other pairs on hard test: `hindsight_lr1e-5` − `givens_lr1e-5` −0.023 [−0.060, +0.012] / −0.024 [−0.053, +0.006] / −0.026 [−0.050, 0.000] at 4 / 16 / 81 steps. Against the 500-step arms from the same starts: `givens_lr1e-5` − `self4` (4.9b) −0.021 [−0.048, +0.006] at 4 steps, +0.018 [−0.004, +0.039] at 16; `hindsight_lr1e-5` − `hind` (4.9c) −0.044 [−0.076, −0.016] at 4 steps, −0.015 [−0.038, +0.007] at 16. Against supervised 4.1: `givens_lr1e-5` −0.041 [−0.078, −0.003] / −0.008 [−0.038, +0.023] / +0.024 [−0.003, +0.050]; `hindsight_lr1e-5` −0.063 [−0.099, −0.027] / −0.032 [−0.062, +0.002] / −0.002 [−0.029, +0.027]. Medium test band at 4 steps: 0.939 (`givens_lr1e-5`, start 0.925), 0.949 (`hindsight_lr1e-5`, start 0.928).

Hard val band (334 puzzles) per round (`val_rounds.csv`):

| rounds | lr | `givens` 16 steps | `givens` 4 steps | `hindsight` 16 steps | `hindsight` 4 steps |
|---|---|---|---|---|---|
| 28 to 34, range | 3e-4 | 0.731 to 0.781 | 0.362 to 0.407 | 0.713 to 0.760 | 0.404 to 0.452 |
| 35, 36, 37, 38 | 1e-5 | 0.799, 0.808, 0.814, 0.814 | 0.461, 0.467, 0.479, 0.452 | 0.781, 0.781, 0.787, 0.790 | 0.479, 0.491, 0.485, 0.482 |

All of the val change is in the first round at 1e-5. Train loss: `givens` 0.6623 → 0.6606 → 0.6591; `hindsight` 0.4000 → 0.3947 → 0.3908.

## Predictions against results

1. `givens_lr1e-5` within ±0.03 of its start at 4 steps: **not held** (+0.057 [+0.024, +0.090]). The `ctrl` arm of 4.9b was the wrong guide: it trained on hard puzzles only, with a restarted optimiser, for 2,000 updates.
2. `hindsight_lr1e-5` +0.02 to +0.06 at 4 steps: **held** (+0.050 [+0.020, +0.078]).
3. `hindsight_lr1e-5` below `hind` of 4.9c by 0.02 to 0.08 at 4 steps: **held** (−0.044 [−0.076, −0.016]).
4. Both runs between 0.00 and +0.04 at 16 steps: **held** (+0.027, +0.006; both intervals include zero).
5. Hard val at 16 steps spans less than 0.03 over the four rounds at 1e-5: **held** (0.015 and 0.009, against 0.050 and 0.048 over rounds 28 to 34).
6. Train loss falls in the first round at 1e-5 by more than it moved over rounds 31 to 34: **not held for `hindsight`** (−0.0053 against −0.0088), **held by a hair for `givens`** (−0.0017 against −0.0015). The loss barely reacts; the solve rates do.

Of the two outcomes named before the run, neither occurred in full: both runs moved, the givens-only run to within 0.02 of the 500-step pass at 4 steps (interval including zero), the hindsight run to 0.044 below it.

## What this shows / does not show

- Shows: lowering the learning rate from 3e-4 to 1e-5 and changing nothing else lifts both long label-free models within 1,000 steps: about +0.05 at 4 sampler steps and +0.04 at 81 on hard test, with the round-to-round swing on val shrinking from 0.05 to under 0.02. The givens-only model, which uses neither a solution nor the checker, ends at 0.495 / 0.807 / 0.895, so about three quarters of what the 500-step pass on own valid grids added at 4 steps (4.9b, +0.077) is reached with the learning rate alone. The plateau levels read in 4.8d and 4.8e were in part an effect of the constant learning rate. The hindsight loop is again not ahead of the givens-only control at any budget.
- Does not show: a like-for-like gap to the supervised model. The supervised reference never had low-lr rounds, so the differences above (−0.04 at 4 steps, none at 16 and 81 for `givens_lr1e-5`) compare a model with a learning-rate drop to one without; the supervised model may gain from the same drop. Also not shown: seed variance (one run per arm); whether a decay schedule, another value, or more rounds would give more (val is flat over rounds 35 to 38); whether the 500-step pass still adds something on top of a model that already had the drop (not run); why the hindsight run gains less at 16 steps. Test was read once, at 4,000 updates.
