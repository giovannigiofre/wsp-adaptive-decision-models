# 4.9d: the two long runs continued with their own training at a low learning rate

**Status: setup and predictions, written before the run (2026-11-12).** Seen before writing them: everything up to [4.9c](../4.9c/README.md).

**Goal.** 4.9b and 4.9c: 500 steps at lr 1e-5 on the model's own valid grids for hard puzzles lift both long label-free models from about 0.43 to 0.52 on hard test at 4 sampler steps; the same pass at lr 3e-4 loses. 4.9c could not say whether the learning rate is why the long loops (4.8d, 4.8e, constant 3e-4 for 43,546 steps) stopped where they did, because that arm also differed from the loops in batches, band and optimiser restart. Question here, one change against the loops themselves: if `hindsight_long` and `givens_long` simply go on with their own training and only the learning rate drops from 3e-4 to 1e-5, do they move?

**Setup.** Each run is resumed from its saved state at round 34, step 43,546 (model, AdamW moments, buffer, RNG) and continued for 4 rounds of 1,000 steps to step 47,546 with `--lr 1e-5` (new option of `phase4.selfloop`: sets the learning rate of the loaded optimiser and scheduler, moments kept). Everything else as in 4.8d / 4.8e: `configs/phase4.yaml`, batch 256, all bands, 16-step greedy harvest of up to 15,000 unsolved train puzzles per round for the hindsight arm, 25% givens batches.

| run | start | arm |
|---|---|---|
| `hindsight_lr1e-5` | `4.8d/hindsight_long/state.pt` | `hindsight` (buffer of self-found grids, hindsight masks) |
| `givens_lr1e-5` | `4.8e/givens_long/state.pt` | `givens` (givens loss only, checker unused) |

No stored solution enters any loss. 4,000 updates at batch 256 per run; the 500-step arms of 4.9b/c made 2,000 updates at batch 256 on hard puzzles only.

**Evaluation.** Per round: hard val band at 16 and 4 sampler steps (the log also holds rounds 28 to 34 at lr 3e-4 for the round-to-round noise). Final models: valid share on the test split at 4, 16, 81 steps, greedy; on the hard band (667 puzzles) paired differences with a 95% bootstrap over puzzles against each run's start, between the two runs, and against the 500-step arms from the same starts (`hind` of 4.9c, `self4` of 4.9b, seeds pooled per puzzle).

**Known limits, before the run.** One run per arm, one seed, one learning rate, a step change with no decay schedule; the intervals cover puzzle sampling only. There is no arm that continues at 3e-4 for the same 4 rounds: the reference for "no change" is the plateau of rounds 28 to 34 on val (single rounds move by ±0.02 to 0.03) and the start model on test. Test is read only at the end (4,000 updates), not at 2,000. The test band was used before.

## Predictions, written before the run

1. **`givens_lr1e-5` at 4 steps:** within ±0.03 of its start on hard test (0.438), interval including zero. The `ctrl` arm of 4.9b (givens loss at 1e-5 on hard puzzles) moved by +0.014.
2. **`hindsight_lr1e-5` at 4 steps:** +0.02 to +0.06 over its start (0.423). I expect less than the 500-step pass: three quarters of its batches are buffer grids of all bands with hindsight masks, not fresh grids for hard puzzles. Low confidence in the lower bound; an interval that includes zero would not surprise me.
3. **`hindsight_lr1e-5` stays below `hind` of 4.9c at 4 steps** (0.516): difference −0.02 to −0.08.
4. **16 steps:** both runs between 0.00 and +0.04 over their starts (0.777, 0.780). Removing the step noise of lr 3e-4 should not hurt; the 500-step arms did not move at 16 steps.
5. **Val noise:** over the four rounds at 1e-5 the hard val rate at 16 steps spans less than 0.03 in each run (rounds 28 to 34 at 3e-4 spanned about 0.05).
6. **Train loss falls in both runs** in the first round at 1e-5, by more than it moved over rounds 31 to 34.

**Results that would be reported as such:** either run gains as much as the 500-step pass (then the learning rate alone, with the loop's own data, explains the plateau at 4 steps); neither run moves at any budget (then the 500-step gain comes from what that pass trains on, not from the learning rate).

## Command

```
bash scripts/run_phase4.sh self49d-hind ~/dllm-cache/venv/bin/python      # the two side by side
bash scripts/run_phase4.sh self49d-givens ~/dllm-cache/venv/bin/python
PYTHONPATH=src python -m phase4.finish_pairs --lowlr                      # results/phase4/4.9d/pairs.csv, val_rounds.csv
```

Expected wall-clock: about 30 to 40 min per run (4 rounds, then val and test evaluation at 4 / 16 / 81 steps). Results: `results/phase4/4.9d/<run>.csv` (rounds 1 to 34 copied from 4.8d / 4.8e), `.json`, `_valid_{val,test}.csv`, `_solved_{val,test}.csv`. Checkpoints under `~/dllm-cache/phase4/4.9d/`.
