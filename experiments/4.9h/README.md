# 4.9h: closing check, own valid grids on top of the givens-only model after its learning-rate drop

**Status: done (2026-11-13). Setup and predictions were committed before the run (`d3c2275`).** Seen before writing them: everything up to [4.9f](../4.9f/README.md).

**Goal.** [4.9b](../4.9b/README.md): 500 steps at lr 1e-5 on the givens-only model's own valid grids added +0.077 on hard test at 4 sampler steps. [4.9d](../4.9d/README.md): dropping the learning rate alone added +0.057 to the same model. [4.9f](../4.9f/README.md): on the supervised side nothing is left of the self-training gain once the model had the drop. The same question on the label-free side was listed as not run: does the 500-step pass still add something to `givens_lr1e-5`? The answer fixes the final number of the no-solutions line.

**Setup.** `configs/phase4_rl_givenslr.yaml`: the 4.9b config with the start replaced by `4.9d/givens_lr1e-5/final.pt` (sha256 `f0b2d778…1d978c`). Everything else as 4.9b `self4`: hard band, 32 puzzles × 8 rollouts with 4 sampler steps, masked-diffusion loss on the valid rollouts, 500 steps, lr 1e-5, seeds 0 to 2. No stored solution enters any loss.

**Evaluation.** Final checkpoints on the hard test band (667 puzzles), greedy at 4 and 16 sampler steps, seeds pooled per puzzle, paired differences with a 95% bootstrap over puzzles: against the start, against `self4` of 4.9b, against `sft_lr1e-5`.

**Known limits, before the run.** One start, one learning rate, no control arm on the givens loss from this start (the `ctrl` of 4.9b moved by +0.014 from the model before the drop). The test band was used before.

## Predictions, written before the run

1. **4 steps:** 0.00 to +0.04 over the start (0.495), interval including zero or just clearing it; in any case less than the +0.077 of 4.9b.
2. **The end point is within ±0.03 of `self4` of 4.9b** (0.515) at 4 steps.
3. **16 steps:** between −0.03 and +0.01 of the start (0.807); `self` of 4.9f lost 0.028.
4. **Still below the supervised model after its drop** (0.634) by more than 0.08 at 4 steps, interval below zero.

**Results that would be reported as such:** a gain of +0.05 or more (the two effects add on the label-free side, unlike on the supervised side).

## Command

```
SEED=0 bash scripts/run_phase4.sh self49h-self ~/dllm-cache/venv/bin/python     # seeds 0, 1, 2 side by side
PYTHONPATH=src python -m phase4.rl_analysis compact 4.9h --config configs/phase4_rl_givenslr.yaml
PYTHONPATH=src python -m phase4.rl_analysis evaluate 4.9h --config configs/phase4_rl_givenslr.yaml
PYTHONPATH=src python -m phase4.finish_pairs --closing
```

Expected wall-clock: 20 to 30 min with three seeds side by side on an idle machine.

## Results (2026-11-13, 00:05 to 00:48)

Apple M5 24 GB, MPS, torch 2.15.0, float32, seeds 0 to 2 (MPS is not bit-deterministic). No thermal warning.

**The run was interrupted and resumed.** The three seeds started side by side at 00:05 on a heavily loaded machine (load above 35, swap nearly full, from other work on the same Mac). At 00:36 the three processes were terminated from outside the experiment session (`Terminated: 15` in the driver log), at steps 426, 451 and 451 of 500. They were resumed from their last `state.pt` (saved every 25 steps: model, optimiser, RNG) one at a time, 00:40 to 00:46, and evaluated at 00:48. `runs.csv` reports the wall-clock of the resumed part only (84 to 150 s) and code commit `d3c2275`; during the resume the working tree also held two uncommitted lines that were not part of the committed setup: `torch.mps.empty_cache()` after each step in `src/phase3/rl.py`, and (from 00:40:24, so possibly not for seed 0) a cap on the MPS memory fraction in `src/phase2/utils.py`. Both only affect memory handling; no rerun on a clean tree was made to confirm that the numbers are unchanged. The step-by-step log is continuous across the interruption.

Hard test band (667 puzzles), valid share, three seeds pooled per puzzle, paired differences with a 95% bootstrap over puzzles (`results/phase4/4.9h/pairs.csv`):

| | 4 steps | seeds | 16 steps |
|---|---|---|---|
| `givens_long` (4.8e, lr 3e-4) | 0.438 | | 0.780 |
| start `givens_lr1e-5` (4.9d) | 0.495 | | 0.807 |
| `self` | 0.528 (+0.033 [+0.007, +0.060]) | 0.537, 0.528, 0.519 | 0.805 (−0.002 [−0.023, +0.021]) |
| `self4` (4.9b, from `givens_long`) | 0.515 | | 0.789 |
| `sft_lr1e-5` (4.9e) | 0.634 | | 0.874 |

`self` − `self4` (4.9b): +0.013 [−0.006, +0.030] at 4 steps, +0.016 [+0.001, +0.032] at 16. `self` − `sft_lr1e-5`: −0.106 [−0.142, −0.071] / −0.069 [−0.096, −0.043]. `self` − `givens_long`: +0.090 [+0.056, +0.124] / +0.026 [−0.001, +0.052].

Training: share of skipped groups 0.39 → 0.39, mean rollout reward on hard train puzzles at 4 steps 0.47 → 0.51 (first 50 against last 50 steps).

## Predictions against results

1. 0.00 to +0.04 over the start at 4 steps, less than the +0.077 of 4.9b: **held** (+0.033 [+0.007, +0.060]; the interval clears zero).
2. Within ±0.03 of `self4` of 4.9b at 4 steps: **held** (+0.013 [−0.006, +0.030]).
3. 16 steps between −0.03 and +0.01 of the start: **held** (−0.002).
4. More than 0.08 below `sft_lr1e-5` at 4 steps, interval below zero: **held** (−0.106 [−0.142, −0.071]).

The outcome named in advance as surprising (+0.05 or more) did not occur.

## What this shows / does not show

- Shows: on the label-free side a small part of the finishing step survives the learning-rate drop. 500 steps on the model's own checker-valid grids add 0.033 at 4 sampler steps to `givens_lr1e-5`, in all three seeds, and nothing at 16; on the supervised side the same step added nothing (4.9f, −0.005). Of the +0.077 read in 4.9b, about 0.03 is left once the model had the drop. The final numbers of the no-solutions line on hard test are 0.528 at 4 steps and 0.805 at 16 (0.807 without the step), against 0.634 / 0.874 for the supervised model: stored solutions are worth 0.11 at 4 steps and 0.07 at 16 when each side has its best finishing step tried here.
- Does not show: that the 0.033 comes from the checker. There is no control arm from this start (500 more steps on the givens loss of hard puzzles, as `ctrl` in 4.9b), so part of it may be the hard-band focus of the step; the lower end of the interval is 0.007 and the test band was read many times. Also not shown: seed variance of the start (one run), other learning rates or lengths, whether iterating the step keeps adding. The run was interrupted and resumed as described above.
