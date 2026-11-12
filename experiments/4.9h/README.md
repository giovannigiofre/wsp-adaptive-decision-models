# 4.9h: closing check, own valid grids on top of the givens-only model after its learning-rate drop

**Status: setup and predictions, written before the run (2026-11-12).** Seen before writing them: everything up to [4.9f](../4.9f/README.md).

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
