# 4.8b: self-training from the rule checker, no stored solutions. Does the checker add anything?

**Status: predictions written before the run (2026-11-11). Not run yet.** Seen before writing them: the 4.8a results, and a smoke run of the code (`hindsight`, 2 rounds of 30 steps). The predictions below were written while the smoke run was still going; its output was read before this commit and shows the first harvest (buffer 0.997 / 0.929 / 0.483 after round 1, 12 min 46 s for the round), so prediction 1 is not blind at commit time. The text of the predictions was not changed after reading it.

**Goal.** Stage A2 of [`docs/SELF_LEARNING_FROM_CHECKER.md`](../../docs/SELF_LEARNING_FROM_CHECKER.md). 4.8a: one hour of masked training on the givens alone gives 0.997 / 0.949 / 0.485 (easy / medium / hard val, 16 sampler steps) with no stored solution and no checker. Question here: starting from that model, does a loop that keeps the model's own rule-valid grids and trains on them beat simply continuing the givens loss for the same number of steps? And does re-masking those grids on arbitrary cells (hindsight) add to it?

**Setup.** All arms resume the 4.8a training state (model, optimiser; step 9,546) and run 10 rounds of 1,000 steps, batch 256, lr 3e-4, seed 0, `configs/phase4.yaml`. The stored solutions of the train set are replaced by zeros; a buffer grid is a rollout accepted by `batch_validity` (rules and givens only).

| arm | each round |
|---|---|
| `givens` (control) | 1,000 steps of the 4.8a loss on random train puzzles; no rollouts |
| `harvest` | greedy 16-step rollouts on train puzzles not yet in the buffer (all 100,002 in round 1, then 15,000 per round), valid grids into the buffer; then 1,000 steps on batches of 192 buffer rows with the puzzle's own mask (givens visible, blanks masked with probability t) and 64 givens-loss rows on puzzles not in the buffer |
| `hindsight` | as `harvest`, but half of the buffer rows (96) use hindsight masks: any of the 81 cells masked with probability t |

After every round: val solve per band at 16 and 4 sampler steps, buffer share per band. At the end: valid share per band at 4, 16, 81 steps on the 1,002 val puzzles, paired bootstrap between arms.

**Reference, no rerun** (supervised 4.1, `results/phase4/sft/sft.csv`): at step 19,500 (2 h 8 min) 1.000 / 0.991 / 0.731 at 16 steps; plateau after 5 h 0.805 hard.

**Known limits, before the run.** One seed per arm: the intervals cover puzzle sampling on 334 val puzzles per band (about ±0.05 on hard), not seed-to-seed variance, so only differences well outside that can be read. Equal steps, not equal wall-clock: the harvest arms spend extra time on rollouts (reported). The three arms run side by side on one GPU, so wall-clock per arm is not that of a run alone. Batch composition differs between arms in more than one way (the harvest arms give 75% of the batch to solved puzzles). `val_cell_acc` reads val solutions, as a measurement only.

## Predictions, written before the run

The memo's A2 predictions (hard at 16 steps above 0.3 with hindsight, below 0.1 without; buffer over 90% of easy after the first launch) were written for a much weaker start than 4.8a gave; the hard threshold is already passed by the start model.

1. **First harvest:** the buffer after round 1 holds about what val says, 0.99 easy, 0.93 to 0.96 medium, 0.43 to 0.53 hard.
2. **The control keeps improving:** `givens` ends at 0.55 to 0.68 hard at 16 steps (it was still rising at the end of 4.8a).
3. **Harvest beats the control on hard:** `harvest` minus `givens` at 16 steps at least +0.05 at the end, and `harvest` ends at 0.65 to 0.78. Reason: a buffer grid labels all blanks of a puzzle, the givens loss never trains a state with the givens all visible. Low confidence in the size.
4. **Hindsight adds nothing measurable** to `harvest` at 16 steps (difference inside ±0.05), against the memo. Own masks at t from 0.05 to 1 already cover the states the sampler visits; hindsight states are not unique-solution puzzles.
5. **At 4 sampler steps the harvest arms gain more than at 16** relative to the control (hard: at least +0.08), because the control never sees a state with few masked cells.
6. **Buffer at the end:** above 0.75 of hard train puzzles in both harvest arms, above 0.99 of easy.
7. **Still below the supervised model at equal steps on hard** (0.731 at step 19,500), by less than 0.10 for the best arm.

**Done when** the three arms have their 10 rounds and the final evaluation. **Negative results that would be reported as such:** harvest equal to the control (the checker adds nothing over the givens loss at this budget); easy band falling below 0.99 in a harvest arm (collapse onto self-found grids); buffer stalling.

## Command

```
ARM=givens bash scripts/run_phase4.sh self-a2 ~/dllm-cache/venv/bin/python      # also ARM=harvest, ARM=hindsight, side by side
```

Each launch pauses after 100 minutes at the next round end; repeat until it prints "finished". Results: `results/phase4/4.8b/<arm>.csv` (one row per round), `<arm>.json`, `<arm>_valid_val.csv`, `<arm>_solved_val.csv`. Checkpoints and buffer under `~/dllm-cache/phase4/4.8b/<arm>/`.
