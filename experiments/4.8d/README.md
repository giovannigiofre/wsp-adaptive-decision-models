# 4.8d: the hindsight loop continued to the step count of the supervised plateau

**Status: predictions written before the run (2026-11-11). Not run yet.** Seen before writing them: all results of [4.8b](../4.8b/README.md) and [4.8c](../4.8c/README.md).

**Goal.** With no stored solution, the hindsight arm of 4.8b reached 0.698 / 0.802 on the hard val band at 16 / 81 sampler steps after 19,546 training steps, 0.03 below the supervised model at the same step (0.731 at 16 steps), and was still moving. The supervised run of 4.1 stopped on its plateau at step 43,500 with 0.805 hard at 16 steps (0.521 at 4). Question: where does the label-free loop stand at that step count?

**Setup.** The seed-0 `hindsight` run of 4.8b is continued unchanged from its saved state (model, optimiser, buffer, RNG; step 19,546, round 10) to round 34, step 43,546. Same loop and settings (`configs/phase4.yaml`, batch 256, lr 3e-4, 1,000 steps per round, 15,000 unsolved train puzzles tried per round, 16-step greedy harvest). Per round: val solve per band at 16 and 4 sampler steps, buffer share. At the end: valid share at 4, 16, 81 steps on the 1,002 val puzzles.

**Fixed before the run:** the stop is round 34, not a plateau rule. Comparison row: supervised 4.1 per-round log (`results/phase4/sft/sft.csv`), no rerun.

**Known limits, before the run.** One seed, one run; the last-round value of one run moves by about ±0.02 between rounds (4.8b), so the mean of the last three rounds is reported beside the final value. Equal steps, not equal wall-clock: the loop also spends time on rollouts. The supervised reference has 81-step values only from other experiments, so the 81-step row has no like-for-like reference. Val only.

## Predictions, written before the run

1. **Hard at 16 steps at the end:** 0.74 to 0.80 (mean of the last three rounds), below the supervised 0.805 by less than 0.07. I do not expect it to pass the supervised plateau.
2. **Hard at 4 steps at the end:** 0.42 to 0.52 (supervised 0.521); the gap to supervised is smaller here than at 16 steps, since the loop trains on fully visible givens.
3. **Hard at 81 steps at the end:** 0.84 to 0.90.
4. **Buffer:** above 0.97 of hard train puzzles at the end; new grids per round under 100 in the last rounds.
5. **Easy and medium do not fall:** easy 1.000 at 16 steps in every round; medium at 16 steps ends at 0.98 or above.
6. **Slowing:** the gain on hard at 16 steps over rounds 11 to 22 is larger than over rounds 23 to 34.

**Negative results that would be reported as such:** hard at 16 steps flat from round 10 (the loop saturates below the supervised model); a band falling (drift on self-found grids).

## Command

```
bash scripts/run_phase4.sh self-a2-long ~/dllm-cache/venv/bin/python
```

Each launch pauses after 100 minutes at the next round end; repeat until it prints "finished". Results: `results/phase4/4.8d/hindsight_long.csv` (rounds 1 to 10 copied from 4.8b), `.json`, `_valid_val.csv`, `_solved_val.csv`. Checkpoints under `~/dllm-cache/phase4/4.8d/`.
