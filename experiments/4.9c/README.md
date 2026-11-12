# 4.9c: why did 500 steps do what 24,000 did not? Learning rate, and the same step on the hindsight model

**Status: predictions written, not run yet (2026-11-12).** Seen before writing them: everything up to [4.9b](../4.9b/README.md).

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
