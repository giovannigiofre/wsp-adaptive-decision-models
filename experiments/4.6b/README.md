# 4.6b: detector gate. How good must a detector of wrong cells be, and is the information in the frozen model?

**Status: done (2026-11-11). Setup and predictions were committed before the full run (`dbdd690`).** A smoke run of the code (CPU, 300 probe puzzles, 4 passes only, ρ in {0.7, 1.0} × φ in {0, 0.3}) was seen before writing the predictions; what it showed is listed below so the predictions can be read with that in mind.

**Goal.** 4.5b: with exactly the wrong cells masked the model repairs every hard test grid in two rounds, and no score from the model or the rules detects them (AUROC 0.41 to 0.69). Before training any detector (with labels or with reward), two cheap questions. Direction 5 of [`docs/RL_DLLM_DIRECTIONS.md`](../../docs/RL_DLLM_DIRECTIONS.md); no model is trained here.

1. **Degraded oracle.** Fill-then-fix as in 4.5b (N forward passes, R = N/4 fix rounds), with a detector that masks each wrong cell with probability ρ and each right filled cell with probability φ. ρ in {0.5, 0.7, 0.9, 1.0}, φ in {0, 0.1, 0.3, 0.6}, N in {4, 8}. Which (ρ, φ) beat the plain sampler at the same N?
2. **Probes on frozen features.** Logistic regression on the last-layer features of each filled cell, plus the nine digit probabilities and the probability of the written digit; features taken with the grid fully visible, or with the cell masked (4 passes, each masking a random quarter). Fitted on greedy rollouts of 3,000 hard *train* puzzles with right/wrong labels, tested on the 667 hard test puzzles: AUROC on all cells and inside failed grids, its recall ρ and false-alarm rate φ at threshold 0.5. The "visible" probe is then used as the detector in fill-then-fix, once with the detector passes not charged to the budget and once charged.

**Seen in the smoke run (not results):** at 4 passes ρ = 0.7 lost about 0.08 against the plain sampler at φ = 0 and at φ = 0.3; ρ = 1.0 gave 0.907 at φ = 0 and 0.84 at φ = 0.3; a probe fitted on 300 puzzles had AUROC about 0.73 on all cells and 0.63 inside failed grids, and fill-then-fix with it was below the plain sampler.

## Predictions, written before the full run

1. **Recall matters far more than false alarms.** At 8 passes every ρ = 1.0 cell of the grid beats the plain sampler (0.736) up to φ = 0.3; at ρ ≤ 0.7 no φ beats it. (Partly informed by the smoke run at 4 passes.)
2. **ρ = 0.9 is the open case:** I expect it above the plain sampler at 8 passes (two rounds give a missed cell a second chance) and not at 4.
3. **Probes beat every score of 4.5b on all cells** (AUROC above 0.69) **but stay below 0.75 inside failed grids**, the threshold the memo sets for going on to a label-trained head. (The smoke run points this way; 3,000 puzzles could move it.)
4. **Masking the cell adds little** to the probe: leave-out AUROC within ±0.03 of the visible one.
5. **The probe as a detector does not beat the plain sampler** at 4 or 8 passes, charged or not: its (ρ, φ) falls in the losing region of the degraded-oracle table.

Decision rule from the memo: if a probe reaches AUROC above 0.75 inside failed grids, or the degraded oracle tolerates the probe's (ρ, φ), the next step is a detector head trained with labels, then with reward. Otherwise the detector needs information the frozen features do not hold linearly, and that is reported as such.

## Command

```
PYTHONPATH=src python -m phase4.detector_gate --size medium
```

Config `configs/phase4.yaml`. Results: `results/phase4/4.6b/degraded_oracle.csv`, `probes.csv`.

**Limits known before the run:** the probe is fitted on grids after the fill and reused on grids after a fix round; the "charged" arm reuses the probe fitted for the longer fill; a linear probe on the last layer only; random-detector draws from one seed.

## Results (2026-11-11, 01:37 to 01:41, 4 min)

Hardware Apple M5 24 GB, MPS, torch 2.15.0, float32, seed 0, greedy decoding (MPS is not bit-deterministic). Hard test band, 667 puzzles. Intervals: 95% bootstrap over puzzles of the paired difference from the plain sampler at the same number of forward passes. The gradient audit of 4.6a ran beside it.

**Degraded oracle**, solve rate minus plain sampler (plain: 0.535 at 4 passes, 0.736 at 8):

| recall ρ | 4 passes: φ = 0 | 0.1 | 0.3 | 0.6 | 8 passes: φ = 0 | 0.1 | 0.3 | 0.6 |
|---|---|---|---|---|---|---|---|---|
| 0.5 | −0.093 | −0.090 | −0.092 | −0.093 | −0.064 | −0.068 | −0.066 | −0.081 |
| 0.7 | −0.081 | −0.080 | −0.078 | −0.082 | −0.052 | −0.052 | −0.042 | −0.064 |
| 0.9 | +0.042 [+0.005, +0.077] | +0.017 [−0.020, +0.053] | +0.026 [−0.009, +0.060] | −0.015 | +0.054 [+0.026, +0.086] | +0.077 [+0.047, +0.108] | +0.051 [+0.023, +0.083] | +0.020 [−0.009, +0.051] |
| 1.0 | +0.372 | +0.364 | +0.307 | +0.147 | +0.264 | +0.262 | +0.262 | +0.214 |

Every ρ ≤ 0.7 cell has an interval entirely below zero; every ρ = 1.0 cell entirely above. With ρ = 1.0 the solve rate at 8 passes is 1.000, 0.999, 0.999 and 0.951 for φ = 0, 0.1, 0.3, 0.6.

**Probes** (logistic regression, fitted on 160,289 cells of 3,000 hard train puzzles):

| fill steps | pass | inputs | AUROC, all cells | AUROC, failed grids | recall ρ | false alarms φ |
|---|---|---|---|---|---|---|
| 3 | visible | features + probabilities | 0.75 | 0.64 | 0.60 | 0.22 |
| 3 | cell masked | features + probabilities | 0.76 | 0.64 | 0.58 | 0.19 |
| 6 | visible | features + probabilities | 0.76 | 0.59 | 0.62 | 0.21 |
| 6 | cell masked | features + probabilities | 0.77 | 0.60 | 0.60 | 0.17 |
| 3 / 6 | visible | probabilities only | 0.54 / 0.53 | 0.53 / 0.52 | | |
| 3 / 6 | cell masked | probabilities only | 0.61 / 0.59 | 0.59 / 0.56 | | |

**The visible probe as detector:** 4 passes −0.054 [−0.087, −0.021]; 8 passes −0.044 [−0.072, −0.014]; with the detector passes charged to the budget −0.241 and −0.153.

## Predictions against results

1. Recall matters far more than false alarms: **held.** ρ = 1.0 beats the plain sampler at every φ tested (also 0.6); no cell with ρ ≤ 0.7 does.
2. ρ = 0.9 above the plain sampler at 8 passes and not at 4: **held at 8** for φ ≤ 0.3; **at 4 in part** (above only at φ = 0, interval from +0.005).
3. Probes above 0.69 on all cells and below 0.75 inside failed grids: **held** (0.75 to 0.77; 0.59 to 0.64).
4. Masking the cell adds little: **held** (+0.01 on all cells).
5. The probe as detector does not beat the plain sampler: **held**; it is below it.

## What this shows / does not show

- Shows: fill-then-fix needs a detector that finds about nine wrong cells in ten; false alarms are cheap (masking 30% of the right cells costs nothing at 8 passes), misses are not. A linear readout of the frozen model's last layer finds six in ten and ranks wrong cells worse inside failed grids (0.59 to 0.64) than over all cells (0.75 to 0.77): it mostly tells failed grids from solved ones. By the memo's rule the label-trained head on frozen features is not the next step; a detector would have to change the features or use more than one pass.
- Does not show: that a non-linear head, earlier layers, or a fine-tuned model cannot reach ρ = 0.9 (none was tried); how a detector with errors concentrated in particular grids behaves (the degraded oracle misses cells independently, a real detector's misses are correlated); anything about a lower threshold on the probe, which trades φ for ρ and was not swept. One seed for the random detector draws.
