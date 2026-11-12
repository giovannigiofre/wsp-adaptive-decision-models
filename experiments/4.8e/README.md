# 4.8e: the givens-only control continued to the step count of the supervised plateau

**Status: predictions written before the run (2026-11-12). Not run yet.** Seen before writing them: all results of [4.8b](../4.8b/README.md), [4.8c](../4.8c/README.md) and [4.8d](../4.8d/README.md).

**Goal.** 4.8d: the label-free hindsight loop flattens at about 0.73 on the hard val band at 16 sampler steps by step 43,546 (0.45 at 4 steps, 0.84 at 81). Its control in 4.8b, training on the givens only (no harvest, no checker), had 0.650 / 0.210 / 0.689 at 16 / 4 / 81 steps at step 19,546 and was still rising by about 0.02 per round. Question: how much of the long hindsight result does the givens-only loss reach on its own with the same number of steps?

**Setup.** The seed-0 `givens` run of 4.8b is continued unchanged from its saved state (step 19,546, round 10) to round 34, step 43,546. Same settings as 4.8d (`configs/phase4.yaml`, batch 256, lr 3e-4, 1,000 steps per round); no rollouts are harvested. Per round: val solve per band at 16 and 4 sampler steps. At the end: valid share at 4, 16, 81 steps on the 1,002 val puzzles, and the paired difference `hindsight_long` − `givens_long` per val puzzle with a 95% bootstrap over puzzles.

**Fixed before the run:** the stop is round 34. Comparison rows: 4.8d (`hindsight_long`) and supervised 4.1, no rerun.

**Known limits, before the run.** One seed, one run per arm; single rounds move by ±0.02 to 0.03. Equal training steps, not equal wall-clock or forward passes (the hindsight loop also spends rollouts). Val only.

## Predictions, written before the run

1. **Hard at 16 steps at the end** (mean of the last three rounds): 0.68 to 0.74. The control keeps rising and ends within 0.05 of the hindsight loop (0.730), below or equal to it.
2. **Hard at 4 steps at the end:** 0.22 to 0.32, clearly below the hindsight loop (0.450); the paired difference at 4 steps has an interval above +0.10.
3. **Hard at 81 steps at the end:** 0.72 to 0.80, below the hindsight loop (0.841) with an interval above zero.
4. **Easy 1.000 and medium at 16 steps 0.98 or above** at the end.
5. **Slowing:** gain on hard at 16 steps from rounds 8-10 to rounds 20-22 larger than from rounds 20-22 to rounds 32-34.

**Results that would change the reading of 4.8, reported as such:** the control within the interval of the hindsight loop at 16 and 81 steps (then at long training the harvest adds only few-step quality); or the control above it.

## Command

```
bash scripts/run_phase4.sh self-a2-ctrl-long ~/dllm-cache/venv/bin/python
PYTHONPATH=src python -m phase4.selfloop_pairs --long        # results/phase4/4.8e/long_pairs.csv
```

Each launch pauses after 100 minutes at the next round end; repeat until it prints "finished". Results: `results/phase4/4.8e/givens_long.csv` (rounds 1 to 10 copied from 4.8b), `.json`, `_valid_val.csv`, `_solved_val.csv`. Checkpoints under `~/dllm-cache/phase4/4.8e/`.
