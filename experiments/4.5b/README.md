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

## Results (2026-11-10, 23:58 to 00:04, 6 min)

Hardware Apple M5 24 GB, MPS, torch 2.15.0, float32, seed 0, greedy decoding (MPS is not bit-deterministic). Hard test band, 667 puzzles. Intervals: 95% bootstrap over puzzles of the paired difference from `plain` at the same number of forward passes.

**Remasking on the start model** (solve rate; difference from `plain`):

| passes | plain | rcr | fix_lowconf | fix_conflict | fix_oracle |
|---|---|---|---|---|---|
| 4 | 0.535 | 0.549, +0.013 [−0.002, +0.029] | 0.457, −0.078 | 0.463, −0.072 | 0.907, +0.372 |
| 8 | 0.736 | 0.750, +0.013 [−0.005, +0.032] | 0.669, −0.068 | 0.670, −0.066 | 1.000, +0.264 |
| 16 | 0.814 | 0.838, +0.024 [0.000, +0.048] | 0.790, −0.024 | 0.790, −0.024 | 1.000, +0.186 |
| 32 | 0.849 | 0.852, +0.003 [−0.014, +0.020] | 0.853, +0.005 | 0.852, +0.003 | 1.000, +0.151 |

**Detectors of wrong cells** on the final grid of the plain sampler (AUROC; 0.5 = chance):

| score | 4 steps, all cells | 4 steps, failed grids | 16 steps, all cells | 16 steps, failed grids |
|---|---|---|---|---|
| commit-time confidence | 0.58 | 0.49 | 0.58 | 0.41 |
| probability with nothing masked | 0.62 | 0.58 | 0.59 | 0.54 |
| probability with the cell masked | 0.69 | 0.61 | 0.64 | 0.55 |
| no conflict with a peer | 0.61 | 0.59 | 0.57 | 0.55 |

Conflict flag: every failed grid has at least one flagged cell; the flag covers 25% of the wrong cells at 4 steps (14% at 16) and 61% (66%) of flagged cells are wrong.

**On the self-training checkpoints of 4.4** (three seeds): `rcr` minus `plain` at 16 passes +0.021 [+0.002, +0.041], +0.014 [−0.006, +0.035], +0.008 [−0.011, +0.027]; at 4 passes −0.002, +0.006, +0.002; the fix arms with a model-only or rules detector are below `plain` as on the start model; `fix_oracle` 0.91 to 0.92 at 4 passes and 1.000 from 8. Full table in `results/phase4/4.5b/remask.csv`.

## Predictions against results

1. Commit confidence is a poor detector: **held** for the AUROC (0.58). `rcr` at 16 passes is +0.024 with an interval that starts at 0.000, just outside the predicted ±0.02; at 32 it is +0.003. `fix_lowconf` is below `plain`, not level with it.
2. Re-prediction is weak inside failed grids: **held** (0.61 and 0.55).
3. The output at visible positions is not a usable detector: **held in substance** (0.54 to 0.62; one value 0.02 above the predicted range). It is no worse than commit confidence.
4. Rules help: **first half held, second half not.** Every failed grid has a flag, but `fix_conflict` is −0.024 at 16 passes, not above +0.03.
5. A perfect detector does not solve everything in one round: **not held.** With the wrong cells masked the model re-predicts them correctly: 1.000 from 8 passes on (two fix rounds), 0.907 at 4 passes (one round).
6. At 4 passes no arm beats `plain`: **held** (`rcr` +0.013, interval includes zero; fix arms −0.07 to −0.08; oracle excepted).
7. Remasking and self-training add up at 16 passes: **undecided.** `rcr` has the same sign on all three checkpoints, but two of the three intervals include zero.

## What this shows / does not show

- Shows: on this model the missing piece is *detecting* wrong cells, not repairing them. Given the exact set of wrong cells, two rounds of mask-and-re-predict solve every hard test puzzle; given any score the model or the rules provide here, remasking gains at most about 0.02 and the fill-then-fix arms lose what the shorter fill costs. The gap between the best real detector and the oracle is 0.16 to 0.37 in solve rate, larger than any RL gain measured in 4.4 and 4.5a.
- Does not show: that no detector can be learned (none was trained here); that `rcr` helps (largest effect +0.024, interval from 0.000); anything about sampled decoding. The `fix_lowconf` and `fix_conflict` rows often have the same solve rate, which fits "the fix rounds change almost nothing and the result is that of a plain sampler with N − R steps", but the plain sampler was not run at 3, 6, 12 and 24 steps to confirm it. One seed for the leave-out partition; `rcr` uses a slightly different commit schedule (rounded cumulative target) from `plain`.
