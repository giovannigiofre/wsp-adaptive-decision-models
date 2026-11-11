# 4.8b: self-training from the rule checker, no stored solutions. Does the checker add anything?

**Status: done (2026-11-11). Setup and predictions were committed before the run (`cff1820`).** Seen before writing them: the 4.8a results, and a smoke run of the code (`hindsight`, 2 rounds of 30 steps). The predictions below were written while the smoke run was still going; its output was read before this commit and shows the first harvest (buffer 0.997 / 0.929 / 0.483 after round 1, 12 min 46 s for the round), so prediction 1 is not blind at commit time. The text of the predictions was not changed after reading it.

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
ARM=givens bash scripts/run_phase4.sh self-a2 ~/dllm-cache/venv/bin/python      # then ARM=harvest, ARM=hindsight (run one at a time, see Results)
```

Each launch pauses after 100 minutes at the next round end; repeat until it prints "finished". Results: `results/phase4/4.8b/<arm>.csv` (one row per round), `<arm>.json`, `<arm>_valid_val.csv`, `<arm>_solved_val.csv`. Checkpoints and buffer under `~/dllm-cache/phase4/4.8b/<arm>/`.

```
PYTHONPATH=src python -m phase4.selfloop_pairs --exp 4.8b      # paired differences, results/phase4/4.8b/arm_pairs.csv
```

## Results (2026-11-11)

**How it was run, against the setup above.** The three arms were first launched side by side at 10:19. The machine was also loaded by unrelated jobs, and after 40 minutes only the control had finished round 1; the two harvest arms were still inside their first harvest and had saved nothing. They were stopped and the arms were then run one at a time: `givens` 105 min (its first round took 40 min under the parallel launch), `harvest` 97 min, `hindsight` 99 min, each in a single launch, finished 15:23. The comparison is at equal steps, as planned; wall-clock per arm is that of a run alone on a machine with varying outside load, and is not a clean timing. Apple M5 24 GB, MPS, torch 2.15.0, float32, seed 0 (MPS is not bit-deterministic). No thermal warning at any launch change.

Valid share on the 1,002 val puzzles (334 per band) at step 19,546, greedy decoding; 95% bootstrap over puzzles. Tables: `<arm>_valid_val.csv`, `arm_pairs.csv`; per round in `<arm>.csv`.

| arm | hard, 4 steps | hard, 16 steps | hard, 81 steps | medium, 4 / 16 / 81 | easy, 4 / 16 / 81 |
|---|---|---|---|---|---|
| start (4.8a, step 9,546) | 0.105 | 0.485 | 0.578 | 0.710 / 0.949 / 0.973 | 0.964 / 0.997 / 1.000 |
| `givens` (control) | 0.210 | 0.650 [0.599, 0.698] | 0.689 [0.644, 0.737] | 0.850 / 0.982 / 0.988 | 0.994 / 1.000 / 1.000 |
| `harvest` | 0.311 [0.261, 0.362] | 0.653 [0.599, 0.704] | 0.719 [0.671, 0.764] | 0.925 / 0.988 / 0.988 | 1.000 / 1.000 / 1.000 |
| `hindsight` | 0.374 [0.323, 0.425] | 0.698 [0.647, 0.746] | 0.802 [0.758, 0.844] | 0.895 / 0.982 / 0.994 | 1.000 / 1.000 / 1.000 |
| supervised (4.1, step 19,500, no rerun) | | 0.731 | | 0.991 at 16 | 1.000 at 16 |

Paired differences on the hard band (334 puzzles):

| a − b | 4 steps | 16 steps | 81 steps |
|---|---|---|---|
| `harvest` − `givens` | +0.102 [+0.048, +0.159] | +0.003 [−0.051, +0.057] | +0.030 [−0.027, +0.081] |
| `hindsight` − `givens` | +0.165 [+0.114, +0.222] | +0.048 [−0.006, +0.102] | +0.114 [+0.063, +0.162] |
| `hindsight` − `harvest` | +0.063 [+0.012, +0.114] | +0.045 [−0.009, +0.099] | +0.084 [+0.036, +0.132] |

On the medium band at 4 steps: `harvest` − `givens` +0.075 [+0.039, +0.111], `hindsight` − `givens` +0.045 [+0.006, +0.084].

Hard val at 16 steps by round (the per-round evaluation, same 334 puzzles):

| round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| `givens` | 0.461 | 0.482 | 0.509 | 0.524 | 0.569 | 0.572 | 0.614 | 0.608 | 0.620 | 0.650 |
| `harvest` | 0.548 | 0.569 | 0.590 | 0.632 | 0.650 | 0.659 | 0.635 | 0.659 | 0.680 | 0.653 |
| `hindsight` | 0.560 | 0.581 | 0.626 | 0.653 | 0.638 | 0.656 | 0.698 | 0.695 | 0.701 | 0.698 |

Buffer (share of train puzzles with a rule-valid grid found by the model, easy / medium / hard): after round 1 0.997 / 0.929 / 0.483 in both arms (80,285 of 100,002 puzzles); at the end `harvest` 1.000 / 0.998 / 0.941, `hindsight` 1.000 / 0.998 / 0.941. New grids per round fall from about 6,300 in round 2 to under 300 in round 10.

## Predictions against results

1. First harvest 0.99 / 0.93 to 0.96 / 0.43 to 0.53: **held** (0.997 / 0.929 / 0.483), not blind, as noted above.
2. The control ends at 0.55 to 0.68 hard at 16 steps: **held** (0.650). It was still rising in its last round.
3. `harvest` − `givens` at least +0.05 at 16 steps, `harvest` at 0.65 to 0.78: **first part not held** (+0.003 [−0.051, +0.057]); the level is at the bottom of the range (0.653). The harvest arm led the control by about 0.08 for the first five rounds and the control caught up.
4. Hindsight adds nothing measurable to `harvest` at 16 steps (inside ±0.05): **held by the letter at 16 steps** (+0.045, interval includes zero) **and wrong in spirit**: at 4 and 81 steps hindsight is above harvest with intervals above zero (+0.063, +0.084). The memo's expectation that hindsight helps is the better description.
5. Harvest arms gain at least +0.08 over the control at 4 steps on hard: **held** (+0.102 and +0.165, intervals above zero), and more than at 16 steps.
6. Buffer above 0.75 hard and 0.99 easy: **held** (0.941, 1.000).
7. Best arm below the supervised model at equal steps by less than 0.10: **held** (`hindsight` 0.698 against 0.731 at 16 steps).

Negative results named before the run: harvest equal to the control at 16 steps **occurred**; easy band below 0.99 in a harvest arm did not occur; the buffer did not stall.

## What this shows / does not show

- Shows: from the 4.8a model, with no stored solution, 10,000 more steps bring the hard val band at 16 sampler steps to 0.65 to 0.70 in all three arms, 0.03 to 0.08 below the supervised model at the same step count. Keeping the model's own rule-valid grids does not beat continuing the givens loss at 16 steps at the end (+0.003), but it does at 4 steps (+0.10), and it finds a valid grid for 94% of the hard train puzzles. Re-masking those grids on arbitrary cells is the only arm clearly above the control at 81 steps (+0.114) and is above plain harvest at 4 and 81 steps.
- Does not show: that these differences survive another seed (one seed per arm; the per-round curve of one arm moves by about ±0.02 between rounds, and the 16-step differences between arms are inside the puzzle-sampling interval). Whether the harvest arms' gain comes from the checker or from the batch composition (75% of each batch on solved puzzles with all givens visible) was not separated. Why hindsight helps at 81 steps more than at 16 was not examined. Buffer grids were accepted by rules and givens only and were not compared with the stored solutions. Val only, no test evaluation. All arms were still moving at the stop, so nothing here is a plateau; the time to match the supervised plateau (0.805) was not measured.
