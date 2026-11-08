# 2.0 Sudoku 4×4 data

**Goal:** enumerate all solved 4×4 grids, split them by solution, build unique-solution puzzles with k ∈ {6..12} hidden cells, and test the checker.

**Command**
```
PYTHONPATH=src python -m phase2.data --config configs/phase2.yaml   # writes results/phase2/data_stats.csv
python -m pytest -q tests/test_sudoku.py
```

- Config: `configs/phase2.yaml` (`data:` block; `split_seed: 0`, puzzle seeds `split_seed + 1000 * split_index`).
- Hardware: Apple M5, 24 GB, macOS 27.0.1. Python 3.12.13, torch 2.15.0. Device: CPU (pure numpy/python). dtype: int64.
- Wall-clock: enumeration + all three splits in 0.7 s.
- Results: `results/phase2/data_stats.csv` (puzzles per split × k).

**Numbers.** Backtracking finds exactly 288 solutions. Split 200 / 44 / 44 by solution (disjoint, tested). Up to 20 (train), 1 (val), 4 (test) distinct puzzles per (solution, k); kept only if the givens are consistent with exactly one of the 288 grids (exact, since every completion is one of the 288; cross-checked against the backtracking solver in `test_uniqueness_count_matches_backtracking`). Totals: train 25,632, val 305, test 1,194 puzzles. k = 12 (4 givens) has fewer unique puzzles (e.g. val 41 instead of 44).

**Tokens.** `PAD=0, MASK=1`, digit d → token d+1; 16 cells row-major. The model's output head predicts only the 4 digits.

**Shows:** the data pipeline is correct (288 grids, uniqueness, disjoint splits, vectorised checker equals the python checker).
**Does not show:** anything about learning; val/test puzzles come from unseen solution grids, so they test generalisation of the rules, not memorisation.
