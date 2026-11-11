# 4.8a: cold start with no stored solutions. Masked training on the givens only

**Status: done (2026-11-11). Setup and predictions were committed before the run (`3fb5f4c`, reference row corrected in `e6705cb`).** A 100-step smoke run of the code was seen (valid share 0 on every band, as expected at step 100); nothing else.

**Goal.** Stage A1 of [`docs/SELF_LEARNING_FROM_CHECKER.md`](../../docs/SELF_LEARNING_FROM_CHECKER.md). Question of the stage: can the 1.86M model teach itself Sudoku from puzzles alone, with a rule checker and no stored solutions? A valid completed grid is its own label, so everything depends on a cold start: the model has to produce *some* valid grids first. A1 tests the cheapest cold start, which needs no tool: the givens of a puzzle are correct labels for their own cells.

**Setup.** Random weights, model and optimiser of 4.1 (`configs/phase4.yaml`, `sizes.medium`, batch 256, lr 3e-4, seed 0). Loss `phase4.selflearn.givens_loss`: blanks always masked, each given masked with probability t ~ U(0.05, 1), cross-entropy on the masked givens weighted 1/t. The loss function receives the puzzles only, and the stored solutions of the train set are replaced by zeros before training (tested in `tests/test_phase4_selflearn.py`). 60 minutes of training wall-clock, then the valid share of greedy confidence-order rollouts on the 1,002 val puzzles per band at 4, 16 and 81 sampler steps, checked from the rules (`batch_validity`).

**Where a solution is still read:** only in the `val_cell_acc` column of the training log (share of hidden val cells equal to the stored solution), a measurement that does not enter training or any decision here.

**Reference, no rerun:** the supervised model of 4.1 after the same 60 minutes (step 9,500 at 3,649 s, `results/phase4/sft/sft.csv`): val solve at 16 steps 1.000 easy, 0.985 medium, 0.668 hard.

**Known limits, before the run.** In training at least all blanks are masked (30 or more cells), so states with fewer masked cells, which the sampler visits as it fills the grid, are never seen. One seed, one model size. Val puzzles only.

## Predictions, written before the run

The memo's (A1): easy valid share at 16 steps between 0.05 and 0.6; higher at 81 steps than at 16; hard below 0.01.

Mine differ in part, and I record both:

1. **Easy at 16 steps is above 0.3**, more likely above the memo's upper bound than below its lower one. Reason: the givens are a random subset of the solution's cells, so predicting a hidden given from the visible ones is the same per-cell problem as ordinary masked training at mask ratios above 30/81, with the loss on fewer cells.
2. **More sampler steps help:** valid share at 81 steps at least that at 16, and 4 steps clearly below 16, on every band where it is not zero.
3. **Hard is above 0.01 and below 0.3 at 16 steps** (the memo says below 0.01). Hard puzzles have 25 to 30 givens, so the model sees few labelled cells per hard puzzle, but easy and medium puzzles masked at large t give the same kind of state.
4. **Below the supervised model at equal time on every band** (it has 1.000 / 0.985 / 0.668).

**Done when** (memo): easy valid share at least 0.05 at some step budget; enough valid grids to start the harvest stage A2. **Negative result:** below 0.01 at every budget; reported as such, then the rule mask or a small-k curriculum, one at a time.

## Command

```
bash scripts/run_phase4.sh self-a1 ~/dllm-cache/venv/bin/python
```

Results: `results/phase4/4.8a/a1.csv` (training log), `a1.json`, `a1_valid_val.csv`. Checkpoints under `~/dllm-cache/phase4/4.8a/a1/`.

## Results (2026-11-11, 08:46 to 09:47)

60 min of training (9,546 steps, stopped by the time budget, 3,600 s evaluations included), about 1 min of final evaluation. Apple M5 24 GB, MPS, torch 2.15.0, float32, seed 0 (MPS is not bit-deterministic). No thermal warning at launch or at the end. Val puzzles, 334 per band, greedy decoding, confidence order; validity from the rules; 95% bootstrap over puzzles.

| sampler steps | easy | medium | hard |
|---|---|---|---|
| 4 | 0.964 [0.943, 0.982] | 0.710 [0.662, 0.758] | 0.105 [0.072, 0.138] |
| 16 | 0.997 [0.991, 1.000] | 0.949 [0.925, 0.970] | 0.485 [0.434, 0.536] |
| 81 | 1.000 | 0.973 [0.955, 0.988] | 0.578 [0.524, 0.629] |
| supervised (4.1) after the same 60 min, 16 steps | 1.000 | 0.985 | 0.668 |

Share of hidden val cells equal to the stored solution at 16 steps: 0.914 (supervised at 60 min: 0.946). The easy column of the training log is not monotone: it moves between 0.92 and 1.00 from one evaluation to the next over the last 30 minutes, while hard rises steadily (0.21 at 20 min, 0.39 at 45 min, 0.485 at 60 min) and had not flattened when the budget ended.

## Predictions against results

- Memo, easy at 16 steps between 0.05 and 0.6: **not held, far above** (0.997). Memo, 81 above 16: **held** on medium and hard. Memo, hard below 0.01: **not held** (0.485).
1. Easy at 16 steps above 0.3: **held** (0.997).
2. More sampler steps help: **held** on every band (hard 0.105, 0.485, 0.578).
3. Hard between 0.01 and 0.3 at 16 steps: **not held, above** (0.485). I underestimated it too.
4. Below the supervised model at equal time on every band: **held on medium and hard** (0.949 against 0.985, 0.485 against 0.668); on easy both are at the ceiling (0.997 against 1.000, one puzzle).

Done-when rule (easy at least 0.05 at some budget): passed.

## What this shows / does not show

- Shows: on this dataset the cold start needs no tool and no stored solution. One hour of masked training on the givens alone gives a model that solves nearly all easy, 95% of medium and about half of the hard val puzzles at 16 sampler steps, about 0.18 below the supervised model on the hard band at equal training time. The states with fewer than 30 masked cells, never seen in training, did not prevent it.
- Does not show: that the model reaches the supervised model with more time (the run stopped at 60 min with hard still rising; the supervised plateau is 0.805 hard at 16 steps after 5 h); that the checker adds anything on top (no rollout or validity check was used in training; that is stage A2); anything at the test split, other seeds or model sizes. It depends on how the puzzles were made: givens here are a random subset of the cells of a full grid, so a hidden given is distributed like any hidden cell; puzzles whose givens are chosen differently (minimal puzzles, hand-made ones) were not tried. The evaluation's unstable easy column was not explained.
