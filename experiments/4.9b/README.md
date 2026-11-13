# 4.9b: the givens-only model trained on its own valid grids, for the 4-step sampler

**Status: done (2026-11-12). Setup and predictions were committed before the run (`963a2d4`).** Seen before writing them: everything up to [4.9a](../4.9a/README.md), including 4.4 and 4.6d.

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

## Results (2026-11-12, 09:17 to 11:54)

Three launches, three seeds side by side: `self4` 09:17 to 09:47 (30 min), `self16` 09:47 to 11:04 (77 min), `ctrl` 11:04 to 11:54 (50 min); test evaluation 2 min. 2 h 37 min in total, against "under an hour" told to the user before the control was added and about 1 h 30 min at launch: 16-step rollouts cost four times the forward passes, and unrelated jobs loaded the machine (load 14 to 23). Apple M5 24 GB, MPS, torch 2.15.0, float32, seeds 0 to 2 (MPS is not bit-deterministic). No thermal warning at any launch change.

Hard test band (667 puzzles), final checkpoints, greedy decoding; seeds pooled per puzzle, 95% bootstrap over puzzles of the paired difference (`results/phase4/4.9b/arms.csv`, `arm_pairs.csv`, `vs_supervised.csv`, per puzzle in `test_solved.csv`). Start model: 0.438 at 4 steps, 0.780 at 16.

| arm | 4 steps: seed 0 / 1 / 2 | 4 steps, minus start | 16 steps: seed 0 / 1 / 2 | 16 steps, minus start |
|---|---|---|---|---|
| `ctrl` | 0.439 / 0.457 / 0.460 | +0.014 [−0.014, +0.043] | 0.780 / 0.786 / 0.784 | +0.003 [−0.018, +0.025] |
| `self4` | 0.513 / 0.517 / 0.516 | +0.077 [+0.047, +0.109] | 0.787 / 0.786 / 0.795 | +0.009 [−0.015, +0.033] |
| `self16` | 0.520 / 0.525 / 0.520 | +0.084 [+0.052, +0.117] | 0.796 / 0.792 / 0.793 | +0.014 [−0.010, +0.038] |

| a − b | 4 steps | 16 steps |
|---|---|---|
| `self4` − `ctrl` | +0.063 [+0.040, +0.086] | +0.006 [−0.012, +0.025] |
| `self16` − `ctrl` | +0.069 [+0.046, +0.093] | +0.010 [−0.008, +0.029] |
| `self16` − `self4` | +0.006 [−0.008, +0.021] | +0.004 [−0.008, +0.017] |
| `self4` − supervised model of 4.1 (0.535 / 0.814) | −0.020 [−0.053, +0.015] | −0.025 [−0.053, +0.004] |
| `self16` − supervised model of 4.1 | −0.014 [−0.046, +0.021] | −0.021 [−0.050, +0.009] |
| `self4` − supervised + same self-training (4.4: 0.602 / 0.837) | −0.087 [−0.120, −0.055] | −0.048 [−0.071, −0.023] |
| `self16` − supervised + same self-training (4.4) | −0.081 [−0.111, −0.050] | −0.043 [−0.069, −0.018] |

Hard val at 4 steps at the end: `self4` 0.524 / 0.488 / 0.506, `self16` 0.515 / 0.518 / 0.512, `ctrl` 0.422 / 0.464 / 0.461 (start 0.404). Medium test band at 4 steps: start 0.925, `ctrl` 0.937 to 0.948, self arms 0.942 to 0.955; easy 0.997 to 1.000 everywhere.

## Predictions against results

1. `ctrl` within ±0.02 of the start, interval including zero: **held** (+0.014 and +0.003).
2. `self4` +0.03 to +0.09 at 4 steps, interval above zero: **held** (+0.077 [+0.047, +0.109]).
3. `self16` − `self4` between −0.02 and +0.04, interval including zero: **held** (+0.006 [−0.008, +0.021]).
4. Both self arms above `ctrl` at 4 steps with intervals above zero: **held** (+0.063, +0.069).
5. 16-step rate changes by −0.01 to +0.04 in the self arms: **held** (+0.009, +0.014; both intervals include zero).
6. Both self arms below 0.535 at 4 steps: **held as numbers** (0.515, 0.522), but the paired difference from the supervised model includes zero (−0.020 [−0.053, +0.015], −0.014 [−0.046, +0.021]).

None of the negative results named before the run occurred.

## What this shows / does not show

- Shows: 500 low-learning-rate steps on its own checker-accepted grids raise the givens-only model from 0.44 to about 0.52 on the hard test band at 4 sampler steps, in every seed, with no stored solution; the same number of steps on the givens loss does not (+0.014, interval including zero), so the gain comes from the self-found grids. Rollouts at 4 or at 16 steps give the same result, although the 16-step ones are valid twice as often and cost 2.5 times the wall-clock. After this step the label-free model is no longer distinguishable from the supervised model of 4.1 at 4 steps on this test band. The 16-step rate does not move measurably.
- Does not show: that the gap to supervised training is closed. The supervised model gains the same +0.07 from the same procedure (4.4), and against that the label-free model stays 0.08 behind at 4 steps and 0.04 at 16. Not examined: whether the first-pass accuracy of 4.9a moved (the probe was not rerun on these models), longer runs, other learning rates, why 24,000 steps on a buffer at lr 3e-4 (`hindsight_long`) did not give this gain while 500 steps at 1e-5 do (learning rate, hard-only batches, fresh temperature-1 grids and the absence of givens batches differ together). One start model; three seeds are a range; the test band had been used before.
