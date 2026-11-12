# 4.9c: why did 500 steps do what 24,000 did not? Learning rate, and the same step on the hindsight model

**Status: done (2026-11-12). Setup and predictions were committed before the run (`4e35b13`).** Seen before writing them: everything up to [4.9b](../4.9b/README.md).

**Goal.** 4.9b: 500 steps at lr 1e-5 on its own valid grids take the givens-only model from 0.438 to 0.515 on hard test at 4 sampler steps. The hindsight loop of 4.8d trained on a buffer of self-found grids for 24,000 steps at lr 3e-4 and ended at 0.423. Between the two, learning rate, hard-only batches, fresh temperature-1 grids and the absence of givens batches differ together. Two questions, one change each against the `self4` arm of 4.9b:

- **Learning rate.** Does the gain survive at the loop's learning rate (3e-4), and at 1e-4?
- **Start.** Does the same 500-step procedure also lift `hindsight_long`, a model that already trained on self-found grids? The supervised model gained +0.067 from it (4.4).

**Setup.** `configs/phase4_rl_givens.yaml`, objective `self_diffusion`, rollouts with 4 sampler steps, everything as the `self4` arm of 4.9b (hard band, 32 puzzles × 8 rollouts, temperature 1, 4 updates per step, 500 steps, grad clip 1, seeds 0 to 2).

| arm | run name | start | lr |
|---|---|---|---|
| `lr1e-4` | `self_diffusion_seed<s>_lr1e-4` | `givens_long` (4.8e) | 1e-4 |
| `lr3e-4` | `self_diffusion_seed<s>_lr3e-4` | `givens_long` (4.8e) | 3e-4 |
| `hind` | `self_diffusion_seed<s>_hind` | `hindsight_long` (4.8d, sha256 `eef1eced…c6b836`) | 1e-5 |

Reference arm, not rerun: `self4` of 4.9b (lr 1e-5 from `givens_long`). No stored solution enters any loss.

**Evaluation.** Final checkpoints on the hard test band (667 puzzles), greedy decoding at 4 and 16 sampler steps, seeds pooled per puzzle, paired differences with a 95% bootstrap over puzzles: each arm against its own start (`givens_long` 0.438 / 0.780; `hindsight_long` 0.423 / 0.777, from `results/phase4/4.8d/hindsight_long_solved_test.csv`), the lr arms against `self4` of 4.9b, and `hind` against `self4`.

**Known limits, before the run.** Three learning rates on one schedule (constant, AdamW restarted with empty moments, which at 3e-4 is itself a disturbance the long loop did not have). The other three differences (hard-only batches, fresh grids, no givens batches) are not separated here. One start model each; test band used before.

## Predictions, written before the run

1. **`hind` gains like the others:** +0.04 to +0.10 at 4 steps over `hindsight_long`, interval above zero. If the low-learning-rate pass on hard puzzles is a generic finishing step (it worked from the supervised and from the givens-only start), it should work here too.
2. **`hind` ends level with `self4`:** `hind` − `self4` at 4 steps within ±0.03, interval including zero. 24,000 steps of loop bring nothing that the 500 steps do not.
3. **`lr3e-4` loses most of the gain at 4 steps:** between −0.05 and +0.04 against the start, and below `self4` with an interval below zero.
4. **`lr3e-4` falls at 16 steps:** −0.02 to −0.10 against the start (2,000 updates at the full learning rate on hard-band grids only, own samples as targets).
5. **`lr1e-4` is in between:** +0.02 to +0.08 at 4 steps; not above `self4`.
6. **Val curves:** at 3e-4 the 4-step val rate moves by more than 0.03 between consecutive evaluations at least three times; at 1e-5 (4.9b) it did so less often.

**Negative results that would be reported as such:** `lr3e-4` gains as much as `self4` (the learning rate is not what separates the two results); `hind` does not gain (the step is not generic).

## Command

```
SEED=0 bash scripts/run_phase4.sh self49c-hind ~/dllm-cache/venv/bin/python       # seeds 0, 1, 2 side by side
SEED=0 bash scripts/run_phase4.sh self49c-lr3e-4 ~/dllm-cache/venv/bin/python
SEED=0 bash scripts/run_phase4.sh self49c-lr1e-4 ~/dllm-cache/venv/bin/python
PYTHONPATH=src python -m phase4.rl_analysis compact 4.9c --config configs/phase4_rl_givens.yaml
PYTHONPATH=src python -m phase4.rl_analysis evaluate 4.9c --config configs/phase4_rl_givens.yaml
PYTHONPATH=src python -m phase4.finish_pairs                # results/phase4/4.9c/pairs.csv
```

## Results (2026-11-12, 13:20 to 14:46)

Three launches, three seeds side by side: `hind` 13:20 to 13:46 (27 min), `lr3e-4` 13:46 to 14:15 (28 min), `lr1e-4` 14:15 to 14:46 (31 min); test evaluation 2 min. Apple M5 24 GB, MPS, torch 2.15.0, float32, seeds 0 to 2 (MPS is not bit-deterministic). No thermal warning at any launch change.

Hard test band (667 puzzles), final checkpoints, greedy decoding; seeds pooled per puzzle, 95% bootstrap over puzzles of the paired difference (`results/phase4/4.9c/pairs.csv`, per puzzle in `test_solved.csv`; `phase4.finish_pairs`).

| arm | start (4 / 16 steps) | 4 steps: seed 0 / 1 / 2 | 4 steps, minus its start | 16 steps: seed 0 / 1 / 2 | 16 steps, minus its start |
|---|---|---|---|---|---|
| `self4` of 4.9b, lr 1e-5 | 0.438 / 0.780 | 0.513 / 0.517 / 0.516 | +0.077 [+0.047, +0.109] | 0.787 / 0.786 / 0.795 | +0.009 [−0.015, +0.033] |
| `lr1e-4` | 0.438 / 0.780 | 0.499 / 0.487 / 0.507 | +0.060 [+0.027, +0.095] | 0.802 / 0.774 / 0.798 | +0.012 [−0.015, +0.039] |
| `lr3e-4` | 0.438 / 0.780 | 0.388 / 0.400 / 0.391 | −0.045 [−0.077, −0.009] | 0.688 / 0.690 / 0.667 | −0.098 [−0.128, −0.069] |
| `hind`, lr 1e-5 | 0.423 / 0.777 | 0.510 / 0.528 / 0.511 | +0.094 [+0.063, +0.124] | 0.798 / 0.799 / 0.796 | +0.021 [−0.007, +0.049] |

| a − b | 4 steps | 16 steps |
|---|---|---|
| `lr1e-4` − `self4` | −0.018 [−0.042, +0.008] | +0.002 [−0.018, +0.023] |
| `lr3e-4` − `self4` | −0.122 [−0.152, −0.092] | −0.107 [−0.132, −0.083] |
| `lr3e-4` − `lr1e-4` | −0.104 [−0.129, −0.080] | −0.109 [−0.132, −0.087] |
| `hind` − `self4` | +0.001 [−0.031, +0.033] | +0.009 [−0.016, +0.033] |

Medium test band at 4 steps: start 0.925; `lr3e-4` 0.895 to 0.904; the other arms 0.948 to 0.957. Hard val at 4 steps, evaluations every 25 steps (`val_jumps.csv`): moves of more than 0.03 between consecutive evaluations 5 to 7 times per run at 3e-4, 2 to 7 at 1e-4, 0 to 2 at 1e-5 (4.9b), 0 in the `hind` runs. At 3e-4 the val rate is below the start from step 25 or 50 on in every seed and ends at 0.365 to 0.374 (start 0.404).

## Predictions against results

1. `hind` +0.04 to +0.10 at 4 steps over `hindsight_long`, interval above zero: **held** (+0.094 [+0.063, +0.124]).
2. `hind` − `self4` within ±0.03, interval including zero: **held** (+0.001 [−0.031, +0.033]).
3. `lr3e-4` between −0.05 and +0.04 against the start and below `self4`: **held, at the low end** (−0.045 with an interval below zero: it does not just lose the gain, it falls).
4. `lr3e-4` falls by 0.02 to 0.10 at 16 steps: **held, at the far end** (−0.098).
5. `lr1e-4` +0.02 to +0.08 at 4 steps, not above `self4`: **held** (+0.060; −0.018 against `self4`, interval including zero).
6. Val moves of more than 0.03 at least three times at 3e-4, less often at 1e-5: **held** (5 to 7 against 0 to 2).

Neither negative result named before the run occurred.

## What this shows / does not show

- Shows: the 500-step pass on own valid grids is a finishing step that depends on a low learning rate and not on the start. At 1e-5 it lifts the hindsight model by +0.09 at 4 steps, to the same 0.52 as the givens-only model (and the supervised model gained +0.07 from it in 4.4); 24,000 steps of the hindsight loop add nothing on top of it. At the loop's learning rate (3e-4) the same pass lowers the model at 4 and at 16 steps and on the medium band; at 1e-4 the gain is still there with a noisier val curve.
- Does not show: that the learning rate is why the long loop stopped at 0.42 to 0.45. The 3e-4 arm differs from the loop in more than one way (hard-band batches only, 32 puzzles per step, AdamW restarted), and it does worse than the loop, so it does not reproduce the loop. A learning-rate decay at the end of the long runs, with their own all-band batches, was not tried. Hard-only batches, fresh temperature-1 grids and the absence of givens batches are still not separated from each other. One start model per arm; three seeds are a range; the test band had been used before.
