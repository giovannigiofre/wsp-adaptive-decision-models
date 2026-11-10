# 4.0 Sudoku 9×9 data and the data gate

**Goal:** build the 9×9 puzzle sets for phase 4 with a local generator and show, before any training, that the data is correct, free of leakage and graded in difficulty.

**Command**
```
PYTHONPATH=src python -m phase4.generate --config configs/phase4.yaml       # writes ~/dllm-cache/phase4/{train,val,test}.npz
PYTHONPATH=src python -m phase4.analysis data --config configs/phase4.yaml  # writes results/phase4/data_report.md
python -m pytest -q tests/test_sudoku9.py
```
Cheapest end-to-end run first: `--scale 0.01 --out <dir>` (1,029 puzzles, 1.5 s), then the report with `--root <dir>`.

- Config: `configs/phase4.yaml` (`data:` block; seed 0; each 1,000-puzzle chunk is seeded by `seed/split/band/chunk`, so the output is identical for any number of worker processes, tested with 4 and 2).
- Hardware: Apple M5 (Mac17,3), 24 GB, macOS 27.0.1. Python 3.12.13, numpy only (torch 2.15.0 installed, not used). Device: CPU, 4 processes. dtype: uint8.
- Wall-clock: generation 93 s; report 3 s; tests 0.3 s.
- Results: `results/phase4/data_report.md`, `results/phase4/data_difficulty_by_k.png`. Data (7 MB) stays in `~/dllm-cache/phase4/`, not committed.

**How a puzzle is made.** A solved grid from randomized backtracking, then a random symmetry (digit relabelling, row/column permutations inside bands, band and stack permutations, transpose). k is drawn uniformly inside the band; cells are hidden in random order and a cell stays given if hiding it would allow a second solution. The order is kept only if it hides exactly k cells; otherwise a new order is tried on the same grid (up to 20), then a new grid. Every puzzle has its own grid, so no grid is shared between puzzles or splits.

**Numbers.** Train 100,002, val 1,002, test 2,001, in equal thirds per band (easy k 30–40, medium 41–50, hard 51–56). All gate checks pass: every stored solution is valid and is the solver's only solution (all val and test, 5,000 train); 0 shared solutions across splits, also after digit relabelling; digit frequency per position 0.1085–0.1149. Test puzzles solved by repeating naked singles: easy 99.4%, medium 76.6%, hard 10.5%; with hidden singles as well: 100%, 98.8%, 67.9%. Filling single-candidate cells in one pass solves no puzzle in any band (it fills 32.5%, 10.6%, 2.8% of the hidden cells).

**What was tried before fixing the bands.** A 200-grid trial per k (seed 0): share of removal orders reaching k was 1.00 up to k = 53, 0.90 at 56, 0.21 at 58, 0.00 at 60; generation cost 0.2 ms per puzzle at k = 30, 7 ms at 56, 12 ms at 58. The planned bands were kept.

**Shows:** the pipeline produces correct, unique-solution, non-overlapping puzzles whose difficulty rises with k, and one-step filling by the rules is not enough in any band.
**Does not show:** that the puzzles are hard by human standards. The easy band is almost entirely naked singles and only 32% of the hard band needs more than singles (11% of all training puzzles); equivalence under grid symmetries other than digit relabelling was not checked; at k = 55–56 the retry rule selects grids and orders that happen to allow sparse puzzles.
