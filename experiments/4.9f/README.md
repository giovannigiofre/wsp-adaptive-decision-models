# 4.9f: is the self-training gain of 4.4 still there after the learning-rate drop?

**Status: setup and predictions, written before the run (2026-11-12).** Seen before writing them: everything up to [4.9e](../4.9e/README.md).

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
