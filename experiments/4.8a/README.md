# 4.8a: cold start with no stored solutions. Masked training on the givens only

**Status: predictions written before the run (2026-11-11). Not run yet.** A 100-step smoke run of the code was seen (valid share 0 on every band, as expected at step 100); nothing else.

**Goal.** Stage A1 of [`docs/SELF_LEARNING_FROM_CHECKER.md`](../../docs/SELF_LEARNING_FROM_CHECKER.md). Question of the stage: can the 1.86M model teach itself Sudoku from puzzles alone, with a rule checker and no stored solutions? A valid completed grid is its own label, so everything depends on a cold start: the model has to produce *some* valid grids first. A1 tests the cheapest cold start, which needs no tool: the givens of a puzzle are correct labels for their own cells.

**Setup.** Random weights, model and optimiser of 4.1 (`configs/phase4.yaml`, `sizes.medium`, batch 256, lr 3e-4, seed 0). Loss `phase4.selflearn.givens_loss`: blanks always masked, each given masked with probability t ~ U(0.05, 1), cross-entropy on the masked givens weighted 1/t. The loss function receives the puzzles only, and the stored solutions of the train set are replaced by zeros before training (tested in `tests/test_phase4_selflearn.py`). 60 minutes of training wall-clock, then the valid share of greedy confidence-order rollouts on the 1,002 val puzzles per band at 4, 16 and 81 sampler steps, checked from the rules (`batch_validity`).

**Where a solution is still read:** only in the `val_cell_acc` column of the training log (share of hidden val cells equal to the stored solution), a measurement that does not enter training or any decision here.

**Reference, no rerun:** the supervised model of 4.1 after the same 60 minutes (step 8,500, `results/phase4/sft/sft.csv`): val solve at 16 steps 1.000 easy, 0.997 medium, 0.647 hard.

**Known limits, before the run.** In training at least all blanks are masked (30 or more cells), so states with fewer masked cells, which the sampler visits as it fills the grid, are never seen. One seed, one model size. Val puzzles only.

## Predictions, written before the run

The memo's (A1): easy valid share at 16 steps between 0.05 and 0.6; higher at 81 steps than at 16; hard below 0.01.

Mine differ in part, and I record both:

1. **Easy at 16 steps is above 0.3**, more likely above the memo's upper bound than below its lower one. Reason: the givens are a random subset of the solution's cells, so predicting a hidden given from the visible ones is the same per-cell problem as ordinary masked training at mask ratios above 30/81, with the loss on fewer cells.
2. **More sampler steps help:** valid share at 81 steps at least that at 16, and 4 steps clearly below 16, on every band where it is not zero.
3. **Hard is above 0.01 and below 0.3 at 16 steps** (the memo says below 0.01). Hard puzzles have 25 to 30 givens, so the model sees few labelled cells per hard puzzle, but easy and medium puzzles masked at large t give the same kind of state.
4. **Below the supervised model at equal time on every band** (it has 1.000 / 0.997 / 0.647).

**Done when** (memo): easy valid share at least 0.05 at some step budget; enough valid grids to start the harvest stage A2. **Negative result:** below 0.01 at every budget; reported as such, then the rule mask or a small-k curriculum, one at a time.

## Command

```
bash scripts/run_phase4.sh self-a1 ~/dllm-cache/venv/bin/python
```

Results: `results/phase4/4.8a/a1.csv` (training log), `a1.json`, `a1_valid_val.csv`. Checkpoints under `~/dllm-cache/phase4/4.8a/a1/`.
