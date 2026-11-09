# 2.2 Sampler: unmasking strategies × step budgets

**Goal:** solve rate on the 1,194 test puzzles for each strategy (`random`, `confidence`, `margin`, `entropy`) and step budget S ∈ {1, 2, 4, 8, 16}, for the weak SFT checkpoint and the fully trained reference.

**Command**
```
PYTHONPATH=src python -m phase2.analysis strategies
PYTHONPATH=src python -m phase2.plots strategies
```

- Config: `configs/phase2.yaml` (`sampler:`). Greedy (T = 0). Commits ⌈remaining / steps_left⌉ cells per step. `confidence` scores the probability of the value being committed, as LLaDA's `low_confidence` does (equal to the max probability when greedy). `margin` = top-1 − top-2, `entropy` = −H. `random` is averaged over 3 order seeds (0, 1, 2).
- Code: `src/phase2/sampler.py` (`score_positions` / `select_topk` hold the selection logic; they only need (B, L, V) probabilities, so phase 1 can reuse them). Tests: `tests/test_sampler.py`.
- Hardware: Apple M5, torch 2.15.0, MPS, fp32. Wall-clock: 9.8 s for the whole grid (2 checkpoints × 4 strategies × 5 budgets, plus 2 extra random seeds).
- Results: `results/phase2/strategies.csv` (with 95% bootstrap CIs and per-k solve rates), `results/phase2/strategies_table.csv`, `results/phase2/strategies_solve_rate.png`.

| test solve rate | S=1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| weak SFT, confidence | 0.018 | 0.077 | 0.282 | 0.482 | 0.559 |
| weak SFT, margin | 0.018 | 0.076 | 0.289 | 0.479 | 0.539 |
| weak SFT, entropy | 0.018 | 0.075 | 0.220 | 0.397 | 0.461 |
| weak SFT, random | 0.018 | 0.058 | 0.102 | 0.142 | 0.147 |
| full SFT, any strategy | 0.998 | ≥0.999 | ≥0.999 | ≥0.999 | 1.000 |

**Shows:** for the weak model, multi-step unmasking matters a lot: one-shot (S = 1) solves 1.8%, S = 16 with confidence ordering solves 56%. *Which* cells are committed first matters as much as how many steps there are: random order reaches only 15%. Confidence ≈ margin > entropy ≫ random.
**Does not show:** any strategy effect for a well-trained model (it solves 99.8% in one shot, because 4×4 is too easy for it). Also, this tiny model's ranking need not carry over to LLaDA-8B (phase 1.2).
