# 4.6b: detector gate. How good must a detector of wrong cells be, and is the information in the frozen model?

**Status: setup and predictions written before the full run.** A smoke run of the code (CPU, 300 probe puzzles, 4 passes only, ρ in {0.7, 1.0} × φ in {0, 0.3}) was seen before writing the predictions; what it showed is listed below so the predictions can be read with that in mind.

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
