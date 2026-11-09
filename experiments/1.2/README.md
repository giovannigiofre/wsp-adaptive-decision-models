# 1.2 Remasking strategies vs step budget

- **Goal:** compare the four unmasking strategies (`random`, `confidence` = LLaDA `low_confidence`, `margin` = top1 - top2, `entropy`) at 0.25x, 0.5x and 1x denoising steps per token, on Sudoku 4x4, Countdown-3 and GSM8K.
- **Command:** `scripts/run_phase1.sh` stage 2, i.e. `python -m src.phase1.run_grid --task {sudoku,countdown,gsm8k} --n {12,12,5} --gen_length {64,64,128}`. Analysis: `python -m src.phase1.report_grid`.
- **Config:** `configs/phase1.yaml`. block_length 32, temperature 0, no CFG, zero-shot d1 prompts. Steps = gen_length x {0.25, 0.5, 1.0}, so 16/32/64 at gen 64 and 32/64/128 at gen 128. Examples come from `tasks.subsample(seed=42)` (nested prefixes, so a run can be extended), listed in `ids.json`. Every example is run under all 12 configs (paired design).
- **Hardware:** Apple M5, 24 GB, MPS, shared with the Docker VM and the phase-2 job (see `results/phase1/env_and_speed.md`).
- **Software / dtype:** torch 2.15.0, transformers 4.38.2. bf16 weights; strategy scores in fp32.
- **Seed:** 42. The `random` strategy is seeded per example with 42 + example index. Greedy decoding is otherwise deterministic up to MPS numerics.
- **Wall-clock:** 4 h 34 min in total (Sudoku 1 h 54 min, Countdown 1 h 15 min, GSM8K 1 h 24 min). That is 384 generations at about 0.8-1.3 s per forward pass.
- **Results:**
  - `results/phase1/strategies.csv` (per config: accuracy with 95% bootstrap CI, full-solve rate, wall time, forward passes).
  - `results/phase1/strategies_{sudoku,countdown,gsm8k}.png`.
  - `results/phase1/strategies_paired.csv` (paired confidence-minus-other differences).
  - Per-generation rows in `results/phase1/raw/<task>.csv` and per-token commit data in `results/phase1/tokens/<task>.jsonl`.

## Deviations from the spec (compute budget)
The spec asks for n = 100/100/50 at gen lengths 64/64/256. At the measured ~1 s per forward pass, one example costs 4 x (16+32+64) = 448 passes at gen 64, or 896 passes at gen 128. Fitting Phase 1 into about 6 h meant:
- **n = 12 Sudoku, 12 Countdown, 5 GSM8K** instead of 100/100/50;
- **GSM8K gen_length 128 instead of 256.** At 256 a single example would take about 30 min for the 4 x 3 configs, and 128 tokens was enough for the 1.0 smoke answer.

With these n, the 95% intervals are wide (often +/- 0.25). Treat every number here as a pilot estimate, not a benchmark. Sudoku "accuracy" is d1's metric: the fraction of the puzzle's empty cells that are filled correctly. Full solves are reported separately and were 0 in every config. Two of the 12 Countdown examples (countdown-26 and countdown-239) are also among the 1.1 prompts: the small 256-item test set made a collision likely despite the different subsampling seed.

## Outcome (accuracy at 1x steps; 95% bootstrap CI over examples)
| task | confidence | margin | entropy | random |
|---|---|---|---|---|
| Sudoku (cell acc., n=12) | 0.05 [0.00, 0.11] | 0.06 [0.02, 0.11] | 0.02 [0.00, 0.05] | 0.07 [0.03, 0.14] |
| Countdown (n=12) | 0.42 [0.17, 0.67] | 0.25 [0.00, 0.50] | 0.17 [0.00, 0.42] | 0.08 [0.00, 0.25] |
| GSM8K (n=5, gen 128) | 0.60 [0.20, 1.00] | 0.80 [0.40, 1.00] | 0.80 [0.40, 1.00] | 0.60 [0.20, 1.00] |

- Paired over examples and averaged over the three budgets, the only difference whose interval excludes 0 is **Countdown confidence - random = +0.28 [0.11, 0.44]**. All other confidence-vs-X differences have intervals that include 0. For GSM8K, `margin` gave the same per-example outcomes as `confidence`, so its paired difference is exactly 0.
- Fewer steps were not clearly worse. On Countdown, confidence rose from 0.17 to 0.42 between 0.25x and 1x steps, but the intervals overlap. Sudoku cell accuracy was highest at 0.25x for most strategies (0.16 vs 0.05 for confidence), but it sits at the floor throughout.
- LLaDA-8B-Instruct essentially cannot do 4x4 Sudoku zero-shot at 64 tokens. It usually copies the puzzle, zeros included, into `<answer>` (see 1.3). d1 also reports low base Sudoku accuracy for this model.
- Wall time is proportional to steps (about 20/41/81 s for Sudoku at 16/32/64 steps). The strategies cost the same per step.

**Shows:** on Countdown, confidence-ordered unmasking beats random order (paired interval excludes 0), and margin and entropy also score above random at every budget. Halving or quartering the steps cut time 2-4x without a detectable accuracy loss at this n.
**Does not show:** a ranking among confidence, margin and entropy (the intervals overlap everywhere), anything at the spec's sample sizes or at GSM8K gen 256, or effects with temperature > 0.
