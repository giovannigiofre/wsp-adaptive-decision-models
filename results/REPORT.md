# Diffusion language models on a MacBook: what I learned

Everything below ran on one Apple M5 MacBook with 24 GB of unified memory, using PyTorch on MPS. No cloud GPUs were used. This is a learning exercise, not research: nothing here is a new method, and negative or contrary results are kept as they came out. Details, commands and per-step notes are in `results/phase1/SUMMARY.md`, `results/phase2/SUMMARY.md`, `experiments/<id>/README.md`, and the two executed notebooks in `notebooks/`.

## 1. A real dLLM: LLaDA-8B-Instruct (inference only)

**It runs, slowly.** In bf16 the weights take about 16 GB on MPS. One denoising step takes about 0.4 s when the machine is idle and about 0.9 s under contention, so a 64-token answer at 64 steps takes about a minute. This speed is what limited the sample sizes: n = 12 Sudoku, 12 Countdown and 5 GSM8K (the spec asked for 100/100/50), about 5 h of compute in total. Every phase 1 number is therefore a pilot estimate with intervals of ±0.2–0.4.

**Confidence-ordered unmasking is almost autoregressive.** With LLaDA's `low_confidence` remasking, each 32-token block fills in mostly left to right: the Spearman correlation between position and commit step is 0.64 within a block. Random order gives −0.18, and its intermediate text is incoherent.

![unmasking heatmap](phase1/unmasking/gsm8k-416_heatmap.png)

**Strategy matters on Countdown; step budget barely mattered.** Confidence ordering beat random by +0.28 [0.11, 0.44] (paired, 95% bootstrap), the only difference whose interval excludes zero. Confidence, margin and entropy cannot be separated at this n. Halving the steps halved the wall time with no detectable accuracy loss.

![countdown strategies](phase1/strategies_countdown.png)

**Commit confidence is not a good error signal here.** Answers are strongly overconfident: 0.87 confidence against 0.42 accuracy on Countdown (ECE 0.45). On Sudoku, cell-level confidence is *anti*-predictive (AUROC 0.10): the model mostly copies the puzzle's `0` placeholders at confidence about 0.75, and those cells are always wrong. Confidence tracks how easy a token is to produce, not whether it is right. Sudoku got 0 full solves in 144 generations, so the Sudoku strategy comparison is uninformative.

![sudoku cell calibration](phase1/calibration/sudoku_token_level.png)

## 2. A toy dLLM trained from scratch, then RL with a hand-written wd1

**Setup.** All 288 valid 4×4 Sudoku grids were split by solution (200/44/44) into unique-solution puzzles. The model is an 0.8M-parameter bidirectional transformer trained with the LLaDA/MDLM masked-diffusion loss (1/t weighting); SFT took 13 minutes on MPS. A model with plain position embeddings plateaued at 14% val, and adding row/column/box embeddings fixed it (the failed attempt is kept). To leave RL room to improve, I deliberately took an under-trained checkpoint at 0.48 test solve rate. The fully trained model solves 100%.

**Decoding order matters for a weak denoiser.** At 8 steps, confidence order solves 0.48 and random order 0.14. The fully trained model solves 99.8% in a single step, so order stops mattering once the model is competent.

**wd1 vs positive-only: the early collapse reproduces; the final ranking does not.** Three seeds each, 500 RL steps, about 1 minute per run:

| | test solve rate |
|---|---|
| weak SFT start | 0.482 |
| wd1, τ = 1 | 0.738 ± 0.005 |
| wd1, τ = 0.5 / 2 | 0.740 / 0.743 |
| positive-only (w⁻ = 0) | 0.932 ± 0.016 |

Positive-only collapses early in every seed (val 0.42 → ≈0.10 within 75–150 steps, while the likelihood of its own wrong samples rises), as the wd1 paper warns. It then recovers and passes wd1 around step 400. wd1 never dips but plateaus lower. Likely reasons, none tested: binary rewards on a near-trivial task, a starting checkpoint one sharpening step away from solving, and no KL/reference term. This should not be read as evidence against wd1 at scale. τ barely matters.

![wd1 vs positive-only](phase2/rl_wd1_vs_positive_only.png)

**The likelihood proxy is usable.** wd1's one-step mean-field log-likelihood correlates with a sequential one-cell-at-a-time log-likelihood (Spearman 0.86), but it is biased low by about 3 nats.

**RL changes calibration in two directions at once.** After wd1, confidence ranks right vs wrong better (puzzle AUROC 0.82 → 0.86), but cells move from slightly under-confident to over-confident (cell ECE 0.045 → 0.063).

![calibration before/after](phase2/calibration_before_after.png)

## What surprised me
- How close LLaDA's default decoding is to left-to-right generation.
- That fewer denoising steps cost almost nothing at these sizes. Confidence-based adaptive step budgets look like the obvious next thing to try.
- That LLaDA's token confidence can be actively misleading (the Sudoku zeros).
- That the positive-only ablation won in the end on the toy task, despite collapsing early.

## Limitations
- Phase 1 samples are tiny, GSM8K used 128-token answers instead of 256, and the d1/paper settings were not matched. Phase 1 numbers are not comparable to published ones.
- Phase 2 is a toy: 4×4 only, binary rewards, no KL term, one hyper-parameter setting, 3 seeds.
- MPS is not bit-deterministic, and timings were taken on a shared machine (phases 1 and 2 ran concurrently, plus a Docker VM).

## Reproduce
- Phase 1: `scripts/run_phase1.sh`. The grid is resumable; raise `N_SUDOKU`/`N_COUNTDOWN`/`N_GSM8K` to extend the same nested samples. The full spec sizes need roughly 30+ h on this Mac.
- Phase 2: the commands in `experiments/2.*/README.md` and `scripts/run_phase2_rl.sh`.
