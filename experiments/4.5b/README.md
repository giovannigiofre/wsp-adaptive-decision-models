# 4.5b: remasking at inference ("eraser"), no training

**Goal.** 4.4b showed that a failed grid has about two wrong cells at its first error and 16 to 21 at the end. Here a committed cell can be masked again and re-predicted (proposal 6 of [`docs/RL_DLLM_SURVEY.md`](../../docs/RL_DLLM_SURVEY.md)). Question: at the same number of forward passes, does any remasking rule solve more hard puzzles than the plain sampler, and how good must the detector of wrong cells be?

**Arms** (N forward passes, N in 4, 8, 16, 32; hard test band, 667 puzzles, greedy):

| arm | what it does | information used |
|---|---|---|
| `plain` | confidence order, N steps | model |
| `rcr` | running-confidence remasking (as in MDPO, 2508.13148): at every step committed cells compete with their commit-time confidence against masked cells with their current confidence | model |
| `fix_lowconf` | fill in N − R steps, then R = N/4 rounds: mask the 20% of filled cells with the lowest commit confidence, re-predict them in one pass | model |
| `fix_conflict` | same, masking the filled cells that share their digit with a peer | model + Sudoku rules |
| `fix_oracle` | same, masking exactly the wrong cells | model + solution (upper reference) |

Second measurement: AUROC of four scores for separating right from wrong filled cells on the final grid of the plain sampler (4 and 16 steps): commit-time confidence; probability the model gives to the written digit with nothing masked (a position the training loss never covered); the same with the cell masked (4 passes, each masking a random quarter of the filled cells); "no conflict with a peer".

The same arms are run on the three self-training checkpoints of 4.4, to see whether a gain from remasking adds to the gain from RL.

## Predictions, written before the run

1. **Commit confidence is a poor detector.** AUROC of commit-time confidence on all filled cells below 0.70 at 16 steps (4.2 measured 0.58 at commit time on a different cut). `rcr` and `fix_lowconf` stay within ±0.02 of `plain` at 16 and 32 passes.
2. **Re-prediction is not much better inside failed grids.** After a cascade the wrong cells agree with each other, so the leave-out probability has AUROC below 0.75 on failed grids. Low confidence in this one.
3. **The untrained "visible" output is not a usable detector** (AUROC within 0.5 ± 0.1), since the loss never touched visible positions.
4. **Rules help.** Every failed grid has a conflict flag, and `fix_conflict` gains more than 0.03 over `plain` at 16 passes.
5. **Even a perfect detector does not solve everything in one round**: with about 20 wrong cells masked at once the refill fails again on some grids, so `fix_oracle` at 16 passes lands between 0.87 (the plain sampler at 81 steps) and 0.97.
6. **At 4 passes no arm beats `plain`**: giving up one of four steps costs more than one fix round returns, the oracle excepted.
7. **Remasking and self-training add up at 16 passes** for the arms that gain on the start model (same sign, at least half the size).

## Command

```
PYTHONPATH=src python -m phase4.eraser --size medium \
  --ckpts self0=$HOME/dllm-cache/phase4/4.4/ckpt/self_diffusion_seed0/final.pt \
          self1=$HOME/dllm-cache/phase4/4.4/ckpt/self_diffusion_seed1/final.pt \
          self2=$HOME/dllm-cache/phase4/4.4/ckpt/self_diffusion_seed2/final.pt
```

Config `configs/phase4.yaml`. Results: `results/phase4/4.5b/remask.csv`, `detectors.csv`.
