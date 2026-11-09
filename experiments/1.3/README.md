# 1.3 Does confidence predict errors?

- **Goal:** check whether LLaDA's commit confidence (the softmax probability of a token when it is unmasked) predicts correctness, at the answer level (all tasks) and the cell level (Sudoku), and whether early-committed tokens are more often right than late ones.
- **Command:** `~/dllm-cache/venv/bin/python -m src.phase1.calibration` (CPU only; reads the 1.2 outputs, no model).
- **Config:** 1.2 runs with the `confidence` strategy at the full step budget: n = 12 Sudoku and 12 Countdown (gen 64, 64 steps), n = 5 GSM8K (gen 128, 128 steps). 10 equal-width bins; 1000 bootstrap resamples (`configs/phase1.yaml`).
  - Answer-level confidence is the mean, or the minimum, commit confidence over the content tokens (end-of-text padding excluded).
  - Sudoku cell confidence is the commit confidence of the token holding that cell's digit in `<answer>`. Digits are tokenized one per token here; no cell shared a token.
- **Hardware / software / dtype / seed:** as 1.2 (the computation itself is CPU numpy/sklearn, a few seconds).
- **Wall-clock:** < 10 s.
- **Results:** `results/phase1/calibration/`:
  - `metrics.csv` (ECE and AUROC with 95% bootstrap CIs)
  - `early_vs_late.csv`
  - `sudoku_cells_by_digit.csv`
  - `reliability_answer_level.png`
  - `sudoku_token_level.png`

## Outcome
| task / level | n | accuracy | mean conf | ECE [95% CI] | AUROC [95% CI] |
|---|---|---|---|---|---|
| Sudoku cells | 96 cells | 0.05 | 0.72 | 0.66 [0.60, 0.73] | **0.10 [0.04, 0.17]** |
| Countdown answers, mean conf | 12 | 0.42 | 0.87 | 0.45 [0.20, 0.71] | 0.63 [0.22, 0.96] |
| Countdown answers, min conf | 12 | 0.42 | 0.30 | 0.18 [0.05, 0.45] | 0.57 [0.19, 0.94] |
| GSM8K answers, mean conf | 5 | 0.60 | 0.95 | 0.35 [0.04, 0.74] | 1.00 (n=5, not meaningful) |
| Sudoku answers | 12 | 0.00 (no full solve) | - | - | undefined (one class) |

- **Mean commit confidence is overconfident on every task.** Mean commit confidence is 0.87-0.95, while accuracy is 0.42-0.60 on Countdown and GSM8K. On Countdown it ranks correct above incorrect answers only weakly; the AUROC interval includes 0.5.
- **Sudoku cell confidence is inversely related to correctness (AUROC 0.10).** The cause is in `sudoku_cells_by_digit.csv`: 71 of 96 empty cells were "filled" with `0`, copied from the puzzle string, at mean confidence 0.75, and all 71 are wrong. Cells where the model wrote an actual digit 1-4 had lower confidence (0.54-0.71) and are sometimes right. The confidence measures how easy a token is to copy, not whether the cell is right.
- **Early vs late commits show no clear difference.** Sudoku cell accuracy by commit order is 0.06 / 0.08 / 0.03 (first / middle / last third, CIs overlapping). Countdown answers whose answer span was committed earlier than the median scored 0.50 [0.17, 0.83], against 0.33 [0.00, 0.67] for later ones.

**Shows:** raw commit confidence from LLaDA is badly calibrated as a correctness signal on these tasks. On Sudoku it is actively misleading, because high confidence marks copied puzzle digits.
**Does not show:** calibration at useful sample sizes (n = 5-12 answers), calibration of other strategies or budgets, or whether a post-hoc recalibration (temperature scaling, min-vs-mean aggregation) would fix it.
