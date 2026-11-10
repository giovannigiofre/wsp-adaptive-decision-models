# Phase 4: a 9×9 Sudoku dLLM from scratch, then the phase 3 objectives again

**Status: plan agreed on 10 Nov 2026, nothing has run.** The decisions taken are listed at the end.

Goal: find out which phase 3 conclusions survive a harder task and a bigger model. Train a masked diffusion model from scratch on 9×9 Sudoku, then rerun the RL objectives of phase 3 on it, this time varying the settings that phase 3 kept fixed (learning rate, group size, rollout temperature, inner steps). Same rules as before: local MacBook only, a prediction committed before each experiment, failed runs kept, no novelty claims.

## What changes from the toy

| | 4×4 (phases 2–3) | 9×9 (phase 4) |
|---|---|---|
| Cells, digits | 16, 4 | 81, 9 |
| Solved grids | 288, all enumerated | about 6.7 × 10²¹, sampled |
| Model | 0.8M parameters | about 2M (to be confirmed in 4.1) |
| Fully trained model | solves everything in one step | expected not to; to be measured |
| RL start | an early snapshot, artificial and fragile | preferably the *converged* model on puzzles it does not yet solve (4.3) |
| One RL step | about 0.11 s | about 2.7 s (estimate below) |
| Runs affordable | about 240 | about 80 |

The last two rows shape the whole plan: RL is about 25 times more expensive per step, so phase 4 cannot sweep the way phase 3 did.

## Measured cost (synthetic benchmark, 10 Nov 2026)

Random tokens, sequence length 81, batch 256, MPS, float32, M5 24 GB, nothing else running. Not a training run.

| Model | Parameters | One training step | One forward pass |
|---|---|---|---|
| 4 layers, d 128 | 0.80M | 125 ms | 28 ms |
| 6 layers, d 160 | 1.86M | 292 ms | 81 ms |
| 6 layers, d 192 | 2.67M | 330 ms | 77 ms |
| 8 layers, d 256 | 6.32M | 637 ms | 162 ms |
| 8 layers, d 384 | 14.2M | 1153 ms | 389 ms |

Derived estimates for the 1.86M model, to be replaced by measured numbers in 4.1 and 4.4:
- SFT: 20,000 steps ≈ 1 h 40 min; 50,000 steps ≈ 4 h (split into resumable launches of under 2 h).
- RL step with 32 puzzles × 8 rollouts, 16 sampler steps, 4 inner steps: 16 × 81 ms + 4 × 292 ms ≈ 2.5 s, plus evaluation. A 500-step run ≈ 22 min alone, about 14 min of machine time each when three run together (phase 3 measured 1.6× throughput for three jobs).

## 4.0 Data
- Solved grids: randomized backtracking fill, seeded. Split by *solution* into train / val / test before any puzzle is built, as in phase 2.
- Puzzle: hide k cells of a solution, keep it only if the solution is unique (solver that stops at the second solution). Store `(puzzle, solution, k, singles_only)`, where `singles_only` says whether the puzzle falls to naked and hidden singles alone. That flag is the cheap difficulty measure; k alone is a poor one.
- Bands: easy (k 30–40), medium (41–50), hard (51–56). Exact ranges fixed after measuring how many unique puzzles the generator yields per band and how long it takes.
- Size target: 100,000 train puzzles, 1,000 val, 2,000 test, generated once and cached under `~/dllm-cache/phase4/` (not committed; generator and seed are).
- Vocabulary: `PAD=0, MASK=1`, digits 1–9 → tokens 2–10.
- Tests: validity checker on known grids, solver agrees with the stored solution, uniqueness check rejects a puzzle with two solutions, no solution shared between splits.

### Is the dataset sensible? A gate before any training
The generator is ours, so nothing external vouches for it. `results/phase4/data_report.md` must show all of the following, and training does not start until it does:

| Check | Passes when |
|---|---|
| Correctness | every stored solution is a valid grid and agrees with its puzzle's givens; the solver, run on every test puzzle and on a 5,000-puzzle train sample, finds exactly one solution and it is the stored one |
| Solver is right | it solves a handful of published puzzles hard-coded in the tests (including one with 17 givens) and counts 2 solutions on a puzzle built to have two |
| No leakage | no solution and no puzzle appears in two splits; no exact duplicate puzzles inside a split. Also checked after relabelling digits to a canonical form, the cheapest symmetry |
| No shortcut in the grids | each digit appears in each cell position with frequency 1/9 within sampling noise (a biased fill order would let the model guess from position alone) |
| No shortcut in the masks | hidden cells are spread uniformly over positions; the number of givens per row, column and box is reported |
| Difficulty is real and graded | per band: distribution of k, share solvable by naked singles only, by naked + hidden singles, and share needing more. The bands must differ on these, and the hard band must contain puzzles that singles alone do not solve |
| Not trivially learnable | two baselines on the test set, per band: "fill every cell that has one candidate, in one pass" and "repeat until stuck". If repeat-until-stuck solves everything in every band, the data only tests constraint propagation and the bands are made harder before training |
| Enough of it | number of distinct puzzles per band and generation time; the yield of unique puzzles at large k is expected to drop sharply (random removal rarely gets below about 24 givens) and decides where the hard band ends |

The report also shows ten puzzles per band printed as grids, to be looked at by eye. Known limit, stated up front: puzzles made by random removal are easier on average than curated "hard" collections with the same number of givens; the difficulty table is what says how much.

## 4.1 Model and masked diffusion training
- Same architecture as phase 2 with the grid size as a parameter: bidirectional pre-LN encoder, learned position plus row / column / box embeddings (phase 2 showed the plain position embedding fails), head over 9 digits.
- Same objective: t ~ U(0.05, 1), mask non-given cells with probability t, cross-entropy on masked cells weighted by 1/t.
- **Size check first (about 1 h total):** train the 0.8M and the 1.86M model for 20 minutes each on the same data and compare val loss and val solve rate per band. Rule, fixed here: use 1.86M unless it is not better than 0.8M (then use 0.8M, cheaper), or unless it is still at zero solve rate on the easy band (then try 6.3M once, and ask before training it fully).
- Then train the chosen size to a plateau: stop when val loss has not improved by 1% over 5,000 steps, or at 50,000 steps. Checkpoints every 500 steps, dense early ones as in phase 2.
- Report: training curve, wall-clock, solve rate per band × sampler steps.

## 4.2 Sampler (inference only, cheap)
- Strategies `random`, `confidence`, `margin`, `entropy` × step budgets {1, 4, 8, 16, 32, 81} on the test set, per band. Bootstrap intervals.
- Oracle-order headroom as in 3.4a (one cell per step, commit a cell the model currently gets right), on the hard band.
- Calibration of commit confidence (reliability diagram, ECE, AUROC), per band.
- Questions carried over: does order matter for a *converged* model here (it did not on the toy)? Does confidence predict errors?

## 4.3 Choosing the RL start (rule fixed before looking)
Phase 3's main weakness was a start that was "a model about to become good on its own". Preference order:
1. **Converged model, harder puzzles.** If the plateau model solves between 0.2 and 0.7 of some band with the default sampler, RL starts from it and trains on that band. No artificial snapshot.
2. If it solves more than 0.7 of every band: make the task harder (larger k, or a smaller step budget), not the model weaker.
3. If it solves less than 0.2 of every band: add an easier band (smaller k). If that is still under 0.2, the model is too weak for the task: train longer or go one size up, and ask first.

Before any RL run, measure at the chosen start the two quantities that decided things in phase 3: the share of sampled rollouts (T = 1) that are valid, and the share of groups of 8 with unequal rewards. 3.6b showed the imitation objectives learn nothing when about 1 rollout in 700 is valid.

## 4.4 The phase 3 objectives at their phase 3 settings
One experiment, 3 seeds, horizon H = 500 steps (to be confirmed by one 1,000-step wd1 run: keep 500 if the ranking at 500 and 1,000 is the same).

| Arm | Setting carried from phase 3 |
|---|---|
| wd1 | τ = 1 |
| positive-only | τ = 1 |
| wd1, floored | floor rescaled to this task: the 10th percentile of the one-step log p̂ of invalid rollouts at the start (rule fixed here; −12 was a toy value) |
| wd1 + KL to start | weight 0.01 |
| filtered imitation | |
| self-training, SFT loss | |
| d1-style GRPO | |

21 runs, about 5 h of machine time in three launches. Diagnostics as in phase 3: share of positive weight on invalid rollouts, share of groups skipped, log p̂ of valid rollouts, plus solve rate per band.

**Phase 3 claims this tests** (each becomes a written prediction in the experiment README):
1. wd1 improves steadily but slowly; its log p̂ of valid rollouts falls.
2. Bounding the push down (floor or KL) is what makes wd1 fast.
3. Positive-only at τ = 1 puts about half of its positive weight on invalid rollouts and collapses early.
4. Training on one's own valid rollouts is the best objective when valid rollouts are not rare.

## 4.5 Settings sweeps
Phase 3 never varied learning rate, group size, rollout temperature or inner steps. Budget does not allow a grid, so: **screen, then confirm.**
- Arms: wd1, plus the two best arms of 4.4.
- Screen, 1 seed, 300 steps, one factor at a time around the default:

| Factor | Values (default in bold) |
|---|---|
| learning rate | 3e-6, **1e-5**, 3e-5, 1e-4 |
| group size G (puzzles per batch adjusted to keep 256 rollouts) | 4, **8**, 16 |
| rollout temperature | 0.5, **1.0**, 1.5 |
| inner steps μ | 1, **4**, 8 |
| sampler steps in rollouts | 8, **16**, 32 |
| τ (wd1 and positive-only only) | 0.1, **1** |

  About 14 non-default settings × 3 arms = 42 short runs, about 6 h.
- Confirm: for each arm, the best screened setting and the default, 3 seeds, full horizon. If two factors both helped, also their combination. About 12–15 runs, about 3.5 h.
- Honesty rule: settings are chosen on val, reported on test, and the write-up says how many settings were tried per arm. A screen with one seed can pick noise; the confirm step is what counts, and "the default was already as good" is a result.

## 4.6 Optional, only if budget remains
- The likelihood-estimate check of 3.6a/3.6c on the two best arms (chain-rule likelihood along the rollout path costs 16 passes with gradient per inner step: about 4× the RL step cost).
- A reward that is not all-or-nothing (share of satisfied row / column / box constraints), which killed positive-only on the toy.

## 4.7 Write-up
`results/phase4/SUMMARY.md` with a table "phase 3 claim → held / did not hold / not testable on 9×9", the prediction scorecard, run notes and limitations; a phase 4 section in both reports; `notebooks/phase4_sudoku9.ipynb` built from committed results; the Lab page on the website updated only if a headline changes.

## Compute budget (estimates)

| Step | Machine time |
|---|---|
| 4.0 data generation and data report | under 1 h (to be measured; the uniqueness check is the slow part) |
| 4.1 size check + training | 1 h + 2–4 h |
| 4.2 sampler, order headroom, calibration | under 1 h |
| 4.3 start selection | 15 min |
| 4.4 objectives | 5 h |
| 4.5 sweeps | 9–10 h |
| 4.6 optional | 3–5 h |
| **Total without 4.6** | **about 18–22 h**, in launches of under 2 h each |

## Code
- Generalise rather than copy: `src/phase2/` (sudoku, data, model, sampler) takes the box size as a parameter; `src/phase3/rl.py` reads the grid size from the config. New code only where needed: `src/phase4/generate.py` (grids, puzzles, difficulty flag), `src/phase4/analysis.py`, `configs/phase4.yaml`, `scripts/run_phase4.sh`.
- Regression guard before anything else: after the generalisation, one phase 2 SFT evaluation and one short phase 3 RL run must reproduce their committed numbers exactly (MPS was bit-repeatable in phase 3).
- The runner does not get edited while a launch is running (lesson from 3.6b).

## Decisions taken (10 Nov 2026)
1. **Data:** our own generator, with the gate in 4.0 passed before training.
2. **Compute:** the 18–22 h estimate is accepted, in launches of under 2 h. 4.6 is not included.
3. **Model size:** decided by the rule in 4.1.
4. **RL start:** the converged model, by the rule in 4.3. No half-trained snapshot.
