# 4.9e: the supervised model with the same learning-rate drop

**Status: done (2026-11-12). Setup and predictions were committed before the run (`cc15f41`).** Seen before writing them: everything up to [4.9d](../4.9d/README.md).

**Goal.** 4.9d: dropping the learning rate from 3e-4 to 1e-5 lifts the two long label-free models by about +0.05 at 4 sampler steps and +0.04 at 81 on hard test. The supervised model of 4.1 was trained at constant 3e-4 to its plateau and never had that drop, so the gap "with solutions against without" read in 4.9d compares unlike things. Question: what does the supervised model gain from the same drop, and what is the gap after both had it?

**Setup.** The supervised run of 4.1 (`sft`) is resumed from its saved state at step 43,500 (model, AdamW moments, RNG) and continued for 4,000 steps to step 47,500 with `--lr 1e-5` (new option of `phase4.train`, same helper as 4.9d; plateau rule off, stop fixed at step 47,500). Everything else as 4.1: `configs/phase4.yaml`, batch 256, the stored solutions of the train set as targets. Run name `sft_lr1e-5`. The label-free runs of 4.9d started at step 43,546 and made the same 4,000 steps.

**Evaluation.** Val every 500 steps (loss, solve per band at 16 sampler steps). Final model: valid share on the test split at 4, 16, 81 steps, greedy; hard band (667 puzzles), paired differences with a 95% bootstrap over puzzles against its start (`sft`) and against `givens_lr1e-5` and `hindsight_lr1e-5` of 4.9d.

**Known limits, before the run.** One run, one seed, one learning rate, a step change; intervals cover puzzle sampling only. The test band was used before. The supervised model after its 4.4 self-training (0.602 / 0.837) is another reference and is not rerun here.

## Predictions, written before the run

1. **Supervised gains at 4 steps:** +0.03 to +0.07 over 0.535 on hard test, interval above zero. The step noise of a constant 3e-4 should cost a supervised model as well.
2. **16 steps:** 0.00 to +0.04 over 0.814. **81 steps:** +0.01 to +0.05 over 0.871.
3. **Gap after both drops, `givens_lr1e-5` − `sft_lr1e-5`, hard test:** −0.03 to −0.08 at 4 steps with an interval below zero; −0.04 to 0.00 at 16 steps; within ±0.03 at 81 steps with an interval including zero.
4. **Val:** the hard val rate at 16 steps over the eight evaluations at 1e-5 spans less than 0.03 (0.799 to 0.829 over the last three at 3e-4) and its mean is above 0.82.
5. **Val loss falls** below the best value of the run so far within the first 1,000 steps at 1e-5.

**Results that would be reported as such:** the supervised model does not gain (then the drop helps only the label-free models and the 4.9d gap stands as measured); it gains more than they did (then the gap is wider than any earlier reading).

## Command

```
bash scripts/run_phase4.sh self49e-sft ~/dllm-cache/venv/bin/python
PYTHONPATH=src python -m phase4.finish_pairs --lowlr-sft                  # results/phase4/4.9e/pairs.csv
```

Expected wall-clock: 35 to 45 min, one MPS job. Results: `results/phase4/4.9e/sft_lr1e-5.csv` (steps up to 43,500 copied from 4.1), `.json`, `_valid_{val,test}.csv`, `_solved_{val,test}.csv`. Checkpoints under `~/dllm-cache/phase4/4.9e/`.

## Results (2026-11-12, 16:07 to 16:37)

One launch, 30 min with the val and test evaluations. Apple M5 24 GB, MPS, torch 2.15.0, float32, seed 0 (MPS is not bit-deterministic). No thermal warning at launch or at the end.

Hard test band (667 puzzles), valid share; paired differences with a 95% bootstrap over puzzles (`results/phase4/4.9e/pairs.csv`):

| hard test | 4 steps | 16 steps | 81 steps |
|---|---|---|---|
| supervised 4.1 (start, step 43,500) | 0.535 | 0.814 | 0.871 |
| `sft_lr1e-5` (step 47,500) | 0.634 (+0.099 [+0.068, +0.129]) | 0.874 (+0.060 [+0.038, +0.083]) | 0.897 (+0.026 [+0.003, +0.050]) |
| `givens_lr1e-5` (4.9d) − `sft_lr1e-5` | −0.139 [−0.177, −0.102] | −0.068 [−0.096, −0.039] | −0.002 [−0.026, +0.023] |
| `hindsight_lr1e-5` (4.9d) − `sft_lr1e-5` | −0.162 [−0.201, −0.124] | −0.092 [−0.123, −0.061] | −0.027 [−0.053, −0.002] |

For comparison, the same gaps before any drop (4.8e): `givens_long` − supervised −0.097 / −0.035 / −0.021 at 4 / 16 / 81 steps. Medium test band at 4 steps: `sft_lr1e-5` 0.976 (start 0.957; `givens_lr1e-5` 0.949, −0.027 [−0.042, −0.014]); at 16 and 81 steps all three models are at 0.99 or above.

Val every 500 steps (`sft_lr1e-5.csv`): val loss 0.0169 at step 43,500 (best of the run until then 0.0164), then 0.0159, 0.0155, 0.0153, 0.0152, 0.0151, 0.0150, 0.0149, 0.0148, still falling at the stop. Hard val at 16 sampler steps: 0.799 to 0.829 over the last eight evaluations at 3e-4; at 1e-5 0.820, 0.841, 0.850, 0.859, 0.844, 0.877, 0.865, 0.850 (mean 0.851).

## Predictions against results

1. Supervised +0.03 to +0.07 at 4 steps: **not held, the gain is larger** (+0.099 [+0.068, +0.129]).
2. 16 steps 0.00 to +0.04: **not held, larger** (+0.060 [+0.038, +0.083]). 81 steps +0.01 to +0.05: **held** (+0.026).
3. Gap `givens_lr1e-5` − `sft_lr1e-5`: at 4 steps −0.03 to −0.08: **not held, wider** (−0.139); at 16 steps −0.04 to 0.00: **not held, wider** (−0.068, interval below zero); at 81 steps within ±0.03 with zero inside: **held** (−0.002).
4. Hard val at 16 steps spans less than 0.03 at 1e-5, mean above 0.82: **span not held** (0.057, because the rate keeps rising over the 4,000 steps), **mean held** (0.851).
5. Val loss below the run's best within 1,000 steps: **held** (0.0159 at the first evaluation).

The second outcome named before the run occurred: the supervised model gains more from the drop than the label-free ones did.

## What this shows / does not show

- Shows: with the same learning-rate drop on both sides, the model trained on stored solutions is ahead of the givens-only model by 0.14 at 4 sampler steps and 0.07 at 16 on hard test, wider than the 0.10 and 0.035 read between the constant-lr models in 4.8e; at 81 steps (one cell per step) the two are level (0.897 and 0.895). The supervised model gains about twice what the label-free models gained from the drop at 4 and 16 steps. The supervised 4.1 numbers used as the reference throughout phase 4 (0.535 / 0.814) understate what that model reaches with a low-lr tail: 0.634 at 4 steps is above the 0.602 that 4.4 reached with self-training from the constant-lr model.
- Does not show: an end point for any of the three (the supervised val loss and solve rate were still improving at the stop; the label-free runs were flat over their four rounds, one run each); seed variance; whether the 4.4 self-training gain (+0.067 at 4 steps) is still there on top of `sft_lr1e-5`, or was in part the same low-lr effect (its learning rate was 1e-5 as well; not run); whether the 500-step pass on own grids adds to `givens_lr1e-5`. One learning rate, a step change, no schedule.
