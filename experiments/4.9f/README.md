# 4.9f: is the self-training gain of 4.4 still there after the learning-rate drop?

**Status: done (2026-11-12). Setup and predictions were committed before the run (`ab7a73c`).** Seen before writing them: everything up to [4.9e](../4.9e/README.md).

**Goal.** 4.4: 500 steps at lr 1e-5 on the supervised model's own valid grids for hard puzzles took it from 0.535 to 0.602 on hard test at 4 sampler steps. 4.9e: 4,000 ordinary supervised steps at lr 1e-5 take the same model to 0.634. The 4.4 start had only ever seen lr 3e-4, so its gain may have been the learning rate and not the self-found grids. Question: from the model that already had the drop (`sft_lr1e-5`), does the same self-training step still add something, and more than the same number of supervised steps?

**Setup.** `configs/phase4_rl_sftlr.yaml`: `configs/phase4_rl.yaml` with the start replaced by `4.9e/sft_lr1e-5/final.pt` (sha256 `f95b9327…c435ff`). Everything else as 4.4: hard band, 32 puzzles × 8 rollouts, 4 sampler steps, temperature 1, 4 updates per step, 500 steps, lr 1e-5, AdamW restarted, seeds 0 to 2.

| arm | run name | objective |
|---|---|---|
| `self` | `self_diffusion_seed<s>` | masked-diffusion loss on the model's own valid rollouts (checker only) |
| `sup` | `supervised_seed<s>` | the same loss on the stored solutions of the sampled puzzles (control: same steps, same puzzles, labels) |

**Evaluation.** Final checkpoints on the hard test band (667 puzzles), greedy at 4 and 16 sampler steps, seeds pooled per puzzle, paired differences with a 95% bootstrap over puzzles: each arm against the start, `self` − `sup`, and `self` against the `self_diffusion` arm of 4.4.

**Known limits, before the run.** `sft_lr1e-5` was still improving when 4.9e stopped, so some gain in both arms is expected from continued low-lr training alone; the `sup` arm is there to measure that, on hard puzzles only and with a restarted optimiser like `self`. One start model. The test band was used before.

## Predictions, written before the run

1. **`self` gains less than in 4.4:** +0.01 to +0.05 at 4 steps over 0.634 (4.4: +0.067).
2. **`sup` gains about the same:** 0.00 to +0.05 at 4 steps.
3. **`self` − `sup` at 4 steps within ±0.03, interval including zero.** Low confidence: in 4.4 self-training was the best objective, but supervised steps were not among its arms.
4. **16 steps:** both arms within ±0.02 of the start (0.874).
5. **`self` ends above the `self_diffusion` arm of 4.4** (0.602) by at least 0.03, interval above zero.

**Results that would be reported as such:** `self` gains as much as in 4.4 and more than `sup` (the self-training gain is real and adds to the drop); neither arm gains (the model is at its level for 4 steps).

## Command

```
SEED=0 bash scripts/run_phase4.sh self49f-self ~/dllm-cache/venv/bin/python     # seeds 0, 1, 2 side by side
SEED=0 bash scripts/run_phase4.sh self49f-sup ~/dllm-cache/venv/bin/python
PYTHONPATH=src python -m phase4.rl_analysis compact 4.9f --config configs/phase4_rl_sftlr.yaml
PYTHONPATH=src python -m phase4.rl_analysis evaluate 4.9f --config configs/phase4_rl_sftlr.yaml
PYTHONPATH=src python -m phase4.finish_pairs --after-drop                      # results/phase4/4.9f/pairs.csv
```

Expected wall-clock: about 30 min per arm with three seeds side by side, two launches, about 1 h in all.

## Results (2026-11-12, 16:41 to 18:23)

Two launches of three seeds side by side: `self` 45 min, `sup` 54 min, then test evaluation; 1 h 42 min in all against the estimated hour (unrelated jobs loaded the machine, load average 34 to 45 during the second launch). Apple M5 24 GB, MPS, torch 2.15.0, float32 (MPS is not bit-deterministic). No thermal warning at either launch change or at the end.

Hard test band (667 puzzles), greedy; seeds pooled per puzzle, paired differences with a 95% bootstrap over puzzles (`results/phase4/4.9f/pairs.csv`):

| hard test | 4 steps | 16 steps |
|---|---|---|
| `sft_lr1e-5` (start) | 0.634 | 0.874 |
| `self`, seeds 0 / 1 / 2 | 0.636 / 0.618 / 0.636 | 0.846 / 0.841 / 0.852 |
| `self` − start | −0.005 [−0.027, +0.017] | −0.028 [−0.047, −0.011] |
| `sup`, seeds 0 / 1 / 2 | 0.637 / 0.625 / 0.636 | 0.861 / 0.862 / 0.862 |
| `sup` − start | −0.002 [−0.021, +0.021] | −0.013 [−0.031, +0.006] |
| `self` − `sup` | −0.003 [−0.016, +0.010] | −0.016 [−0.027, −0.005] |
| `self` − `self_diffusion` of 4.4 (0.602 / 0.837) | +0.028 [+0.010, +0.045] | +0.010 [−0.004, +0.024] |
| start − `self_diffusion` of 4.4 | +0.032 [+0.011, +0.052] | +0.038 [+0.021, +0.055] |

## Predictions against results

1. `self` +0.01 to +0.05 at 4 steps: **not held** (−0.005, interval including zero: no gain).
2. `sup` 0.00 to +0.05 at 4 steps: **at the bottom edge** (−0.002: no gain).
3. `self` − `sup` within ±0.03 at 4 steps, zero inside: **held** (−0.003 [−0.016, +0.010]).
4. Both arms within ±0.02 of the start at 16 steps: **held for `sup`** (−0.013), **not held for `self`** (−0.028, interval below zero).
5. `self` above the 4.4 arm by at least 0.03 at 4 steps: **just short** (+0.028 [+0.010, +0.045]); all of it is the start, which is already +0.032 above the 4.4 arm.

The second outcome named before the run occurred: neither arm gains at 4 steps.

## What this shows / does not show

- Shows: from a supervised model that already had 4,000 steps at lr 1e-5, the self-training step of 4.4 adds nothing at 4 sampler steps and costs about 0.03 at 16; 500 supervised steps on the same puzzles add nothing either. Together with 4.9e (the lr drop alone ends 0.03 above the 4.4 result at 4 steps and 0.04 at 16), the +0.067 that 4.4 read as a gain from training on the model's own valid grids is accounted for by the low learning rate of that step applied to a model trained at constant 3e-4. The same caveat now applies to the 4.4 ranking of objectives as a whole: all of them started from that model.
- Does not show: that training on own valid grids can never help (one start, one learning rate, hard band, 500 steps, 4-step rollouts); what the 500-step pass does on top of `givens_lr1e-5` (in 4.9b it added +0.077 before the drop, in 4.9d the drop alone gave +0.057; not run); why `self` loses at 16 steps (hard-band-only grids from 4-step rollouts as targets; not examined). 4.4 itself was not rerun, the comparison is across experiments on the same test puzzles.
