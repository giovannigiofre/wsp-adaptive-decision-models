# 4.9b: the givens-only model trained on its own valid grids, for the 4-step sampler

**Status: predictions written, not run yet (2026-11-12).** Seen before writing them: everything up to [4.9a](../4.9a/README.md), including 4.4 and 4.6d.

**Goal.** 4.9a: the givens-only model of 4.8e follows the row/column/box rule, and its deficit against the supervised model sits in the single pass on the puzzle as given (−0.03 per blank cell), which costs it at the first of 4 sampler steps (hard test 0.438 against 0.535 at 4 steps; 0.780 against 0.814 at 16). With 16 steps it solves 78% of hard puzzles, and a checker can tell which. Question: do 500 low-learning-rate steps on those self-found valid grids raise the 4-step solve rate, with no stored solution, and is it the grids that do it or just the extra steps?

**Setup.** The self-training arm of 4.4 (`phase3.rl`, objective `self_diffusion`: masked-diffusion loss on the rollouts the checker accepts), unchanged except for the start: `configs/phase4_rl_givens.yaml` is `configs/phase4_rl.yaml` with `init` = `4.8e/givens_long/final.pt` (sha256 pinned). Hard band, 32 puzzles × 8 rollouts per step at temperature 1, 4 gradient updates per step, lr 1e-5, 500 steps, seeds 0 to 2. Three arms:

| arm | run name | training rollouts | loss |
|---|---|---|---|
| `self4` | `self_diffusion` | 4 sampler steps (as 4.4) | diffusion loss on own valid grids |
| `self16` | `self_diffusion_h16` | 16 sampler steps (`rl.harvest_steps=16`, new option) | diffusion loss on own valid grids |
| `ctrl` | `givens` | 4 steps, not used by the loss | givens loss of 4.8a on the same batch puzzles (new objective `givens`) |

`ctrl` has the same puzzles, number of updates, learning rate and optimiser as the other two and never calls the checker for its loss: it separates "own valid grids" from "500 more steps at a low learning rate on hard puzzles". In two smoke steps the start model's rollouts at temperature 1 were valid in 0.34 of cases with 4 steps and 0.76 with 16.

**Use of labels.** None in training: the loss sees puzzles and checker-accepted grids only (the trainer loads the stored solutions of the train set but these objectives do not read them). Val and test solve rates are scored by the checker.

**Evaluation.** Final checkpoints (step 500), greedy `confidence` decoding on the hard test band (667 puzzles) at 4 and 16 sampler steps; seeds pooled per puzzle; paired difference from the start model and between arms with a 95% bootstrap over puzzles (`rl_analysis evaluate` / `arms`).

**Known limits, before the run.** Three seeds from one start model (one 4.8e run). 500 steps, one learning rate, no sweep. The test set was already used to compare the long models (4.8e), so this is not a first look at it. `hindsight_long` trained for 24,000 steps on a buffer of self-found grids at lr 3e-4 and is not above `givens_long` at 4 steps on test (0.423 against 0.438), which is a reason to expect little here; the differences are the low learning rate, hard puzzles only, temperature-1 rollouts drawn fresh each step, and no givens batches mixed in.

## Predictions, written before the run

1. **`ctrl` does nothing:** hard test at 4 steps within ±0.02 of the start (0.438), interval including zero; same at 16 steps.
2. **`self4` gains at 4 steps:** +0.03 to +0.09 over the start, interval above zero. Reasoning: in 4.6d the same 500 steps gave the supervised start +0.07 whichever puzzles the targets came from, and this start has never had a loss on blank cells. Moderate confidence, given the `hindsight_long` result above.
3. **`self16` is not below `self4`:** `self16` − `self4` between −0.02 and +0.04 at 4 steps, interval including zero. Its grids cover harder puzzles (0.76 valid against 0.34), but 4.6d found that which puzzles supply the targets does not matter.
4. **The grids, not the steps:** `self4` − `ctrl` and `self16` − `ctrl` are positive at 4 steps with intervals above zero.
5. **16 sampler steps:** `self4` and `self16` change the 16-step test rate by −0.01 to +0.04 (4.4: +0.02).
6. **Still below the supervised model:** both self arms end below 0.535 at 4 steps on hard test.

**Negative results that would be reported as such:** no arm above the start at 4 steps (the first-pass deficit is not repaired by own grids at this scale); `ctrl` gains as much as the self arms (the gain is extra steps, not the checker); a self arm falls at 16 steps.

## Command

```
SEED=0 bash scripts/run_phase4.sh self49-self4 ~/dllm-cache/venv/bin/python     # seeds 0, 1, 2 side by side
SEED=0 bash scripts/run_phase4.sh self49-self16 ~/dllm-cache/venv/bin/python
SEED=0 bash scripts/run_phase4.sh self49-ctrl ~/dllm-cache/venv/bin/python
PYTHONPATH=src python -m phase4.rl_analysis compact 4.9b --config configs/phase4_rl_givens.yaml
PYTHONPATH=src python -m phase4.rl_analysis evaluate 4.9b --config configs/phase4_rl_givens.yaml
PYTHONPATH=src python -m phase4.rl_analysis arms 4.9b --config configs/phase4_rl_givens.yaml --refs \
  --pairs self_diffusion:givens self_diffusion_h16:givens self_diffusion_h16:self_diffusion
```

Checkpoints and logs under `~/dllm-cache/phase4/4.9b/`; tables in `results/phase4/4.9b/`.
