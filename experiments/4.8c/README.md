# 4.8c: two more seeds of the harvest and hindsight arms of 4.8b

**Status: predictions written before the run (2026-11-11). Not run yet.** Seen before writing them: all results of [4.8b](../4.8b/README.md).

**Goal.** In 4.8b (one seed per arm) `hindsight` ended above `harvest` on the hard val band at 4 and 81 sampler steps (+0.063 [+0.012, +0.114], +0.084 [+0.036, +0.132]) and not clearly at 16 (+0.045 [−0.009, +0.099]). Those intervals cover puzzle sampling only. Question: does the difference hold over training seeds?

**Setup.** Exactly the `harvest` and `hindsight` arms of 4.8b (10 rounds of 1,000 steps from the 4.8a state at step 9,546, batch 256, lr 3e-4, `configs/phase4.yaml`), with `diffusion_training.seed` 1 and 2. Run names `harvest_s1`, `harvest_s2`, `hindsight_s1`, `hindsight_s2`; one arm at a time. Analysis: the seed-0 runs of 4.8b are added; per sampler budget, the three seeds are pooled per val puzzle (mean of valid over seeds) and the paired difference `hindsight` − `harvest` gets a 95% bootstrap over puzzles; the three per-seed differences are listed beside it.

**Known limits, before the run.** The seed changes the training batches and masks only: the start model (4.8a, seed 0) is the same in every run, and the first harvest is greedy, so the buffer after round 1 is the same up to MPS non-determinism. The spread measured here is therefore smaller than that of a full rerun from scratch. The control (`givens`) is not repeated, so differences from the control still rest on one control seed. Three seeds give three differences: a range, not an interval. Val only. Outside load on the machine varies; wall-clock is not a clean timing.

## Predictions, written before the run

1. **Hard at 81 steps:** `hindsight` − `harvest` is positive in both new seeds, and the pooled difference is at least +0.04 with an interval above zero. Moderate confidence: the 4.8b difference was the largest of the three budgets.
2. **Hard at 4 steps:** positive pooled difference, +0.02 to +0.08; I would not be surprised if one of the two new seeds has it at zero or below.
3. **Hard at 16 steps:** pooled difference inside ±0.05 with an interval that includes zero or just clears it. The per-round curve of one run moves by about ±0.02, so the sign per seed is not predictable.
4. **Seed spread of one arm** on hard at 16 steps: the three final values of each arm lie within 0.06 of each other.
5. **Buffer:** every run ends above 0.93 of hard train puzzles.
6. **Medium at 4 steps:** `hindsight` is not above `harvest` (4.8b: −0.030 [−0.060, 0.000]); pooled difference between −0.05 and +0.01.

**Negative result that would be reported as such:** pooled `hindsight` − `harvest` with intervals including zero at all three budgets, i.e. the 4.8b advantage was seed noise.

## Command

```
ARM=harvest SEED=1 bash scripts/run_phase4.sh self-a2-seed ~/dllm-cache/venv/bin/python     # also SEED=2, ARM=hindsight
```

Each launch pauses after 100 minutes at the next round end; repeat until it prints "finished". Results: `results/phase4/4.8c/<arm>_s<seed>.csv`, `.json`, `_valid_val.csv`, `_solved_val.csv`. Checkpoints under `~/dllm-cache/phase4/4.8c/`.
