# 2.4 Calibration before/after RL (+ likelihood-estimator sanity check)

**Goal:**
- Check whether RL changes how well the sampler's commit confidence predicts correctness, at the cell level and at the puzzle level.
- Check whether the one-step log-likelihood used by wd1 tracks a sequential (one cell per pass) log-likelihood.

**Commands**
```
PYTHONPATH=src python -m phase2.analysis likelihood
PYTHONPATH=src python -m phase2.plots likelihood
PYTHONPATH=src python -m phase2.analysis calibration
PYTHONPATH=src python -m phase2.plots calibration
```

**Checkpoint choice.** "rl_best" is set by `results/phase2/rl/best_run.json`. The rule is: among the wd1 τ = 1 runs, take the one with the highest val solve rate and use its best-val checkpoint. That gives `wd1_tau1_seed0`, step 500. All 12 RL checkpoints are also evaluated.

**Calibration protocol**
- Data: test set, 1,194 puzzles and 10,632 hidden cells.
- Decoding: greedy, confidence strategy, S = 8.
- Cell confidence = probability of the committed value at commit time. Cell correct = it matches the unique solution.
- Puzzle confidence = mean cell confidence. Puzzle correct = solved.
- ECE uses 10 equal-width bins. AUROC is from sklearn.

**Setup**
- Hardware: Apple M5, torch 2.15.0, MPS, fp32.
- Wall-clock: likelihood 0.7 s, calibration 3.8 s.

**Results**
- Calibration: `results/phase2/calibration.csv`, `calibration_bins.csv` and `calibration_before_after.png`.
- Likelihood: `results/phase2/likelihood_estimators.{csv,json,png}`.

| test, greedy S=8 | solve | cell acc | mean cell conf | cell ECE | cell AUROC | puzzle ECE | puzzle AUROC |
|---|---|---|---|---|---|---|---|
| weak SFT | 0.482 | 0.760 | 0.715 | 0.045 | 0.618 | 0.251 | 0.816 |
| rl_best (wd1 τ=1 seed 0) | 0.735 | 0.876 | 0.949 | 0.073 | 0.658 | 0.216 | 0.881 |
| wd1, all 9 runs (mean) | 0.736 | 0.883 | 0.945 | 0.063 | 0.648 | 0.211 | 0.863 |
| positive-only, 3 runs (mean) | 0.932 | 0.964 | 0.899 | 0.066 | 0.733 | 0.051 | 0.931 |
| full SFT | 1.000 | 1.000 | 1.000 | 0.000 | n/a | 0.000 | n/a |

The per-run rows are in `calibration.csv`. The across-seed std for every metric is ≤ 0.015.

**What the calibration numbers show**
- **wd1 makes cell confidence overconfident.** The weak SFT model was slightly *under*-confident (conf 0.72 vs acc 0.76). After wd1 the model is over-confident (conf 0.95 vs acc 0.88), so cell ECE rises from 0.045 to about 0.06–0.07.
- **Ranking still improves a little.** Cell AUROC goes from 0.62 to 0.65, and puzzle AUROC from 0.82 to 0.86–0.88.
- **Positive-only is under-confident.** Its cells are under-confident (conf 0.90 vs acc 0.96). Its confidence separates right from wrong cells best (AUROC 0.73), and its puzzle-level ECE is low (0.05).

**Likelihood estimators.** These use 128 test puzzles × 2 rollouts at T = 1, compared with the sequential log-likelihood.

| | Pearson | Spearman | per-cell Pearson |
|---|---|---|---|
| weak SFT, one-step, p_mask = 0 | 0.80 | 0.86 | 0.61 |
| weak SFT, one-step, p = 0.15 (1 draw / 16-draw mean) | 0.76 / 0.82 | 0.80 / 0.86 | 0.48 / 0.60 |
| rl_best, one-step, p_mask = 0 | 0.71 | 0.79 | 0.65 |

The one-step estimate is on average lower than the sequential one. The gap is 2.8 nats for weak SFT and 4.6 for rl_best. That is expected: the one-step estimate scores every cell without the context of the others.

**Shows:**
- The one-step estimator wd1 relies on is a biased but rank-correlated proxy for the sequential likelihood (Spearman about 0.8).
- RL improved accuracy and discrimination, but wd1 also turned slight under-confidence into over-confidence at the cell level.

**Does not show:**
- Calibration at other temperatures or step budgets.
- Whether the sequential likelihood itself is the right target. Row-major order is one arbitrary choice among many.
- Anything about calibration of large dLLMs.
