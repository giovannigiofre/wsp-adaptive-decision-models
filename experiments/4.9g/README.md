# 4.9g: closing check, the push-down term after the learning-rate drop

**Status: setup and predictions, written before the run (2026-11-12).** Seen before writing them: everything up to [4.9f](../4.9f/README.md).

**Goal.** 4.9e and 4.9f: the +0.05 to +0.07 that the three pull-up objectives of [4.4](../4.4/README.md) gained at 4 sampler steps is accounted for by their learning rate (1e-5) applied to a start trained at constant 3e-4. Every arm of 4.4 had that start, so its other reading, that lowering invalid rollouts hurts (plain wd1 −0.030), is also confounded: wd1 may have lost more than the table shows, or the loss may belong to that start. Question: from the supervised model that already had the drop (`sft_lr1e-5`), does plain wd1 still end below the start, and below the same objective without its push-down term?

**Setup.** `configs/phase4_rl_sftlr.yaml` (start `4.9e/sft_lr1e-5/final.pt`, sha256 pinned). Everything else as 4.4 and 4.9f: hard band, 32 puzzles × 8 rollouts, 4 sampler steps, temperature 1, 4 updates per step, 500 steps, lr 1e-5, binary reward, one-step likelihood estimate, seeds 0 to 2.

| arm | run name | objective |
|---|---|---|
| `wd1` | `wd1_beta1_tau1_seed<s>` | plain wd1, τ 1: raises valid rollouts, lowers invalid ones |
| `pos` | `wd1_beta0_tau1_seed<s>` | the same code path with β 0: the push-down term removed |

The two arms differ in β only. The `self` and `sup` arms of 4.9f, from the same start with the same settings, are read beside them.

**Evaluation.** Final checkpoints on the hard test band (667 puzzles), greedy at 4 and 16 sampler steps, seeds pooled per puzzle, paired differences with a 95% bootstrap over puzzles: each arm against the start, `pos` − `wd1`, each against `self` of 4.9f.

**Known limits, before the run.** One start, the phase 3 settings untuned (τ, lr, no floor, no KL), 500 steps, the one-step likelihood estimate; the bounded variants and GRPO of 4.4 are not rerun. Intervals cover puzzle sampling, not seeds. The test band was used before.

## Predictions, written before the run

1. **`wd1` ends below the start at 4 steps:** −0.02 to −0.08 from 0.634, interval below zero.
2. **`pos` does not move at 4 steps:** within ±0.02 of the start, interval including zero (as `self` in 4.9f).
3. **`pos` − `wd1` at 4 steps:** +0.02 to +0.08, interval above zero.
4. **16 steps:** `wd1` −0.02 to −0.06 from 0.874; `pos` between −0.03 and +0.01.
5. **Diagnostics as in 4.4:** the one-step log p̂ of valid rollouts falls under `wd1` over the run and the share of skipped groups rises.

**Results that would be reported as such:** `wd1` within the interval of the start (then its loss in 4.4 belonged to that start, and "lowering invalid rollouts hurts" is not supported on a properly finished model); `pos` gains (then weighted pull-up adds something that plain self-training did not in 4.9f).

## Command

```
SEED=0 bash scripts/run_phase4.sh self49g-wd1 ~/dllm-cache/venv/bin/python     # seeds 0, 1, 2 side by side
SEED=0 bash scripts/run_phase4.sh self49g-pos ~/dllm-cache/venv/bin/python
PYTHONPATH=src python -m phase4.rl_analysis compact 4.9g --config configs/phase4_rl_sftlr.yaml
PYTHONPATH=src python -m phase4.rl_analysis evaluate 4.9g --config configs/phase4_rl_sftlr.yaml
PYTHONPATH=src python -m phase4.finish_pairs --closing                         # results/phase4/4.9g/pairs.csv, 4.9h/pairs.csv
```

Expected wall-clock: 20 to 30 min per arm with three seeds side by side on an idle machine, two launches.
