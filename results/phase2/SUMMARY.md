# Phase 2 summary: a toy 4×4 Sudoku dLLM + hand-written wd1 on a MacBook

This is a small-scale replication exercise. Nothing here is a new method.

- Numbers are test solve rates on 1,194 unseen puzzles (greedy, confidence strategy, S = 8).
- Intervals are mean ± std over 3 seeds unless noted.
- Each step has its own README in `experiments/2.0` … `experiments/2.5`.

## Setup

- **Data (2.0):** the 288 valid 4×4 grids are split by *solution* (200 / 44 / 44).
  - Puzzles hide k ∈ {6, 8, 10, 12} cells.
  - Uniqueness is checked exactly against all 288 grids.
  - Split sizes: 25,632 train / 305 val / 1,194 test.
- **Model (2.1):** bidirectional transformer, 4 layers, d = 128, **798,212 params**.
  - Learned position embedding plus row/column/box embeddings.
  - The plain-position variant plateaued at 14% val. That failed attempt is kept in `sft_attempts/`.
- **Mac training time:** Apple M5 24 GB, MPS, fp32. The GPU was shared with a concurrent LLaDA-8B job.
  - SFT (masked diffusion, 20k steps): **778 s**.
  - One RL run (500 steps): **61–68 s**. All 12 RL runs: **787 s**.
  - Evaluations: < 15 s in total.

## Results

| | test solve rate |
|---|---|
| weak SFT start (step 120, chosen on val ∈ [0.3, 0.6]) | 0.482 (one checkpoint) |
| **SFT + wd1, τ = 1** | **0.738 ± 0.005** |
| SFT + wd1, τ = 0.5 / 2 | 0.740 ± 0.009 / 0.743 ± 0.009 |
| **SFT + positive-only (w⁻ = 0)** | **0.932 ± 0.016** |
| fully trained SFT (reference) | 1.000 |

### Decoding strategy (2.2)

For the weak model, unmasking order matters as much as the step budget:

- confidence order: 0.48 at S = 8 and 0.56 at S = 16;
- random order: 0.14 at S = 8.

The fully trained model solves 99.8% in one step, so strategy makes no difference for it.

### wd1 vs positive-only (2.3a/b): mixed

**The early instability reproduces.** Positive-only shows the early instability the wd1 paper warns about.

- In all 3 seeds, val falls from 0.42 to ≈ 0.10 within 75–150 steps.
- Meanwhile the model grows more confident in its own (mostly wrong) samples.
- wd1 never dips and improves steadily.

**The end result does not.** Positive-only then recovers and overtakes wd1 at ≈ step 400. It finishes far ahead (0.93 vs 0.74). So the claim that positive-only *ends* worse does **not** reproduce here at 500 steps.

Possible reasons (untested):

- binary rewards on a near-trivial task;
- a start checkpoint that is a mid-warm-up snapshot, one sharpening step away from solving;
- no KL/reference term to temper wd1's negative push.

### Other checks

- **τ (2.3c):** wd1 is insensitive to τ ∈ {0.5, 1, 2}.
- **Likelihood proxy (2.3 sanity):** wd1's one-step log-likelihood correlates with the sequential one-cell-per-pass log-likelihood (Pearson 0.80, Spearman 0.86 on weak SFT). It is biased low by ≈ 3 nats.

### Calibration before/after (2.4)

| | cell ECE | cell AUROC | puzzle ECE | puzzle AUROC | mean conf vs acc (cells) |
|---|---|---|---|---|---|
| weak SFT | 0.045 | 0.62 | 0.25 | 0.82 | 0.72 vs 0.76 (under-confident) |
| wd1 (9 runs) | 0.063 | 0.65 | 0.21 | 0.86 | 0.95 vs 0.88 (over-confident) |
| positive-only (3 runs) | 0.066 | 0.73 | 0.05 | 0.93 | 0.90 vs 0.96 (under-confident) |

- RL improves how well confidence *ranks* right vs wrong (AUROC).
- wd1 moves cells from slightly under-confident to over-confident.

## Limitations

- **Toy task:** 4×4 Sudoku has 288 grids, and a fully trained 0.8M model solves it perfectly.
  - The interesting regime exists only because the start checkpoint is deliberately under-trained.
  - The 0.3–0.6 window lasts only ~40 SFT steps.
- **Simplified RL:**
  - binary rewards;
  - no reference/KL term;
  - one-step likelihood estimate;
  - 500 steps and 3 seeds;
  - one hyper-parameter setting (lr 1e-5, μ = 4, B·G = 32·8).
- **Measurement:** MPS is not bit-deterministic, and timings were taken on a shared GPU.

## What this does and does not say about large dLLMs

**What it does give:** a working, tested reference for the mechanics:

- the masked-diffusion loss;
- confidence-ordered unmasking;
- the wd1 sign convention and weights;
- a one-step likelihood proxy.

It also illustrates qualitatively that:

- (a) decoding order matters for an under-trained denoiser;
- (b) positive-only weighting can collapse early.

**What it does not say:**

- how LLaDA-8B-scale models respond to wd1;
- the long-run ranking of wd1 vs positive-only on real reasoning tasks;
- calibration effects at scale.

The one contrary result here (positive-only ends best) may be an artefact of the toy setting. It should not be read as evidence against wd1.
