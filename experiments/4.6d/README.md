# 4.6d: what does self-training learn? Supervised arms on the same puzzles

**Status: done (2026-11-11). Setup and predictions were committed before the runs (`efa8198`).**

**Goal.** Self-training (retrain on the model's own valid rollouts with the diffusion loss) was the best row of 4.4: +0.067 [+0.038, +0.095] on the hard test band at 4 sampler steps. Because every valid rollout equals the stored solution (4.4b), self-training is supervised training on the puzzles the model already solves. Question: is its gain a matter of *which* puzzles are trained on, or just of more diffusion-loss steps on hard train puzzles? Direction 4 of [`docs/RL_DLLM_DIRECTIONS.md`](../../docs/RL_DLLM_DIRECTIONS.md).

**Arms** (same loop, optimiser, batch of 32 puzzles × 8 rollouts and 500 steps as 4.4; the loss is the diffusion loss on the *stored solution*, so these arms use labels and are reference rows, not RL):

| arm | trained on |
|---|---|
| `sup-all` | the solution of every puzzle in the batch |
| `sup-solved` | only puzzles with at least one valid rollout in the batch |
| `sup-unsolved` | only puzzles with no valid rollout in the batch (queued last; run) |

Three seeds each, evaluated on the 667 hard test puzzles at 4 and 16 sampler steps, paired with the start model and with the self-training checkpoints of 4.4.

## Predictions, written before the runs

1. **`sup-solved` equals self-training** within the interval (difference from self-training inside ±0.03): it is the same data up to how many copies of a solved puzzle enter the loss.
2. **`sup-all` is at least as good as self-training** (memo). Its extra data are the puzzles the model fails, which are the informative ones; I expect +0.07 to +0.12 over the start.
3. **`sup-unsolved` alone still gains** over the start, less than `sup-all`.
4. **None of these moves the 16-step solve rate by more than ±0.02**: 500 steps at lr 1e-5 is a small amount of training next to the 43,500 steps of 4.1.

If prediction 2 holds, self-training is a label-free approximation of a few hundred more supervised steps, and `sup-all` is the reference row an RL objective has to be compared with on this task.

## Command

```
SEED=0 bash scripts/run_phase4.sh rl46-sup-all ~/dllm-cache/venv/bin/python      # seeds 0, 1, 2
SEED=0 bash scripts/run_phase4.sh rl46-sup-solved ~/dllm-cache/venv/bin/python
```

Config `configs/phase4_rl.yaml`. Checkpoints and logs under `~/dllm-cache/phase4/4.6d/`.

```
SEED=0 bash scripts/run_phase4.sh rl46-sup-unsolved ~/dllm-cache/venv/bin/python
PYTHONPATH=src python -m phase4.rl_analysis compact 4.6d && PYTHONPATH=src python -m phase4.rl_analysis evaluate 4.6d --budgets 4 16
PYTHONPATH=src python -m phase4.rl_analysis arms 4.6d --refs 4.4 4.6c --pairs supervised:self_diffusion ...
```

## Results (2026-11-11)

Wall-clock, three seeds side by side: `sup-all` 03:39 to 04:18 (39 min), `sup-solved` 04:18 to 04:49 (31 min), `sup-unsolved` 04:49 to 05:09 (20 min), test evaluation 3 min. Apple M5 24 GB, MPS, torch 2.15.0, float32, seeds 0 to 2 (MPS is not bit-deterministic). No thermal warning at any launch change. Hard test band, 667 puzzles, final checkpoints (step 500), greedy decoding; seeds pooled per puzzle, 95% bootstrap over puzzles of the paired difference. Tables: `results/phase4/4.6d/arms.csv`, `arm_pairs.csv`, per puzzle in `test_solved.csv`. Run names: `supervised`, `supervised_onsolved`, `supervised_onunsolved`.

Share of puzzles of a batch left out of the loss at step 500 (seed 0): `sup-solved` 0.28, `sup-unsolved` 0.78. So `sup-unsolved` trains on about a quarter of each batch.

| arm | test at 4 steps | minus start (0.535) | test at 16 steps | minus start (0.814) |
|---|---|---|---|---|
| `sup-all` | 0.610 | +0.074 [+0.045, +0.102] | 0.850 | +0.036 [+0.013, +0.058] |
| `sup-solved` | 0.607 | +0.072 [+0.043, +0.100] | 0.840 | +0.025 [+0.002, +0.048] |
| `sup-unsolved` | 0.603 | +0.068 [+0.042, +0.094] | 0.851 | +0.037 [+0.015, +0.057] |
| self-training (4.4) | 0.602 | +0.067 [+0.038, +0.095] | 0.837 | +0.022 [+0.000, +0.045] |

| a − b | at 4 steps | at 16 steps |
|---|---|---|
| `sup-all` − self-training | +0.007 [−0.006, +0.022] | +0.013 [+0.003, +0.024] |
| `sup-solved` − self-training | +0.005 [−0.008, +0.019] | +0.003 [−0.007, +0.014] |
| `sup-unsolved` − self-training | +0.001 [−0.014, +0.018] | +0.014 [+0.002, +0.028] |
| `sup-all` − `sup-solved` | +0.002 [−0.011, +0.016] | +0.010 [+0.001, +0.021] |
| `sup-all` − `sup-unsolved` | +0.006 [−0.010, +0.021] | −0.001 [−0.012, +0.010] |
| `sup-solved` − `sup-unsolved` | +0.004 [−0.012, +0.020] | −0.011 [−0.023, +0.000] |
| `sup-all` − batch weights, β = 0 (4.6c R1) | +0.000 [−0.015, +0.015] | +0.003 [−0.011, +0.019] |

## Predictions against results

1. `sup-solved` equals self-training within ±0.03: **held** (+0.005 [−0.008, +0.019]).
2. `sup-all` at least as good as self-training, +0.07 to +0.12 over the start: **held, at the bottom of the range** (+0.074). At 4 steps it is not distinguishable from self-training (+0.007, interval includes zero); the labels of the failed puzzles add nothing measurable there.
3. `sup-unsolved` gains over the start, less than `sup-all`: **first part held** (+0.068); **second part not established** (0.006 below `sup-all`, interval includes zero), with a quarter of the data per batch.
4. None moves the 16-step solve rate by more than ±0.02: **not held.** All three gain +0.025 to +0.037 at 16 steps, intervals above zero.

## What this shows / does not show

- Shows: at 4 sampler steps, 500 steps of the diffusion loss on stored solutions give the same gain whichever puzzles are used (all, only those the model already solves, only those it fails): 0.603 to 0.610, all within the interval of self-training (0.602) and of the best label-free arm of 4.6c (0.610). On this task self-training is as good as supervised training with labels on the same puzzles, and the gain at 4 steps does not depend on which puzzles carry the loss. The labels of failed puzzles show up only at 16 steps (+0.013 and +0.014 over self-training, intervals just above zero).
- Does not show: why the gain is the same. More steps on hard-band puzzles, on-distribution masking levels, or a ceiling of what 500 steps at lr 1e-5 can move were not separated; a control with the diffusion loss on puzzles of all bands, or a longer run, was not made. `sup-unsolved` sees fewer cells per step, so equal gain there is not equal data efficiency. Seeds are pooled per puzzle: intervals cover puzzle sampling, not seed-to-seed variance. One start model.
