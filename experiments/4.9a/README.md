# 4.9a: what did the givens-only model learn? Measurements on the three long models, no training

**Status: done (2026-11-12). Setup and predictions were committed before the run (`ff5f906`).** Seen before writing them: all results of 4.8a to 4.8e. Nothing of this script had been run.

**Goal.** 4.8e: trained on the printed digits only (no solutions, no checker), the model reaches 0.780 on the hard test band at 16 sampler steps, equal to the hindsight loop (0.777) and 0.035 below the supervised model (0.814); at 4 steps it is 0.098 below (0.438 against 0.535). Two questions, measured without training: (1) what does the givens-only model predict on the blank cells, which never carried a loss; (2) where in a 4-step rollout does it lose against the supervised model?

**Setup.** Final models at about 43,500 steps: `supervised` (4.1), `givens_long` (4.8e), `hindsight_long` (4.8d). The 1,002 val puzzles, hard band (334) reported first. The stored solutions are read for measurement only. Greedy decoding, confidence order. Every model sees the same states (fixed seed).

- **m1, one forward pass on the puzzle as given.** Per blank cell: top-1 equal to the solution; probability on *excluded* digits (held by a visible cell of the same row, column or box); share of top-1 predictions that are excluded; all by the number of digits left by that elimination (1, 2, 3, 4+).
- **m1b, hidden givens against blanks.** The same pass with each given hidden with probability 0.25. Accuracy on the hidden givens (the only kind of cell the givens loss ever scored) against the blanks, same state.
- **m2, states past the givens.** A share r = 0.25, 0.5, 0.75 of the blanks is shown with the true digit; accuracy on the blanks still hidden. The givens-only loss never saw a state with more visible cells than the givens.
- **m3, rollouts with 4 and 16 steps.** Wrong share of the cells committed at each step; per puzzle, the step of its first wrong cell.

Paired differences per puzzle on the hard band with a 95% bootstrap over puzzles (`pairs_hard.csv`).

**Known limits, before the run.** One model per arm, so every difference is between three single runs, not between methods. Val puzzles (the test set stays unused here). Hiding givens can make a puzzle ambiguous: the generator keeps as givens the cells whose removal breaks uniqueness, so in m1b a hidden given can have more than one consistent digit and its accuracy has a ceiling below 1 that was not computed. In m2 the shown cells are true digits, not the model's own, so it measures the states of an error-free rollout. Row, column and box elimination is the simplest rule reading; "follows the rules" here means only that. m3 describes where errors appear, not why.

## Predictions, written before the run

1. **Rules (m1, hard band).** All three models put less than 0.03 of the probability on excluded digits on the blank cells, and fewer than 1% of their top-1 predictions are excluded. `givens_long` is within 0.01 of `supervised` on both.
2. **By candidates (m1, hard band).** On cells with one digit left, all three are at 0.98 or above. On cells with 4 or more left, `givens_long` is below `supervised` by 0.02 to 0.08; over all blank cells the paired difference is negative with an interval below zero.
3. **Hidden givens against blanks (m1b, hard band).** In every model the hidden givens are *harder* than the blanks (accuracy lower by more than 0.03), because of how the generator chooses givens. The difference between that gap in `givens_long` and in `supervised` is inside ±0.03: the model trained only on givens is not better on givens than on blanks.
4. **Past the givens (m2, hard band).** `givens_long` gets more accurate as cells are shown (higher at r = 0.75 than at r = 0), and its distance from `supervised` at r = 0.75 is no larger than at r = 0. Moderate confidence; the opposite (a distance that grows with r) would say the missing states cost something.
5. **Where 4 steps fail (m3, hard band).** In all three models more than 60% of the puzzles with an error have their first wrong cell at step 1. The share of puzzles with a wrong cell at step 1 is higher for `givens_long` than for `supervised` by 0.05 to 0.15 (interval above zero), which is most of the 0.10 gap in solve rate. `hindsight_long` and `givens_long` do not differ there.

**Results that would be reported as negative:** more than 0.05 of the probability on excluded digits in `givens_long` (it would not be reading the rules from the givens); no difference at step 1 between `givens_long` and `supervised` (the 4-step gap would come from later steps).

## Command

```
bash scripts/run_phase4.sh givens-probe ~/dllm-cache/venv/bin/python
```

Config `configs/phase4.yaml` (`sizes.medium`). Results: `results/phase4/4.9a/cells.csv`, `reveal.csv`, `steps.csv`, `pairs_hard.csv`.

## Results (2026-11-12, 08:48)

One launch, 40 s. Apple M5 24 GB, MPS, torch 2.15.0, float32, seed 0 (MPS is not bit-deterministic). No thermal warning at launch. Hard val band, 334 puzzles, 17,872 blank cells. As a check, the 4-step and 16-step solve rates of the rollouts equal those of 4.1, 4.8d and 4.8e (0.521 / 0.404 / 0.452 and 0.805 / 0.781 / 0.731).

**m1, one pass on the puzzle as given** (`cells.csv`):

| blank cells with | n | `supervised` | `givens_long` | `hindsight_long` |
|---|---|---|---|---|
| 1 digit left | 507 | 1.000 | 1.000 | 1.000 |
| 2 left | 2,721 | 0.835 | 0.804 | 0.807 |
| 3 left | 5,521 | 0.749 | 0.719 | 0.715 |
| 4 or more left | 9,123 | 0.661 | 0.628 | 0.615 |
| all | 17,872 | 0.724 | 0.694 | 0.686 |
| probability on excluded digits | | 0.0016 | 0.0004 | 0.0006 |
| top-1 excluded | | 0 | 0 | 0 |

Per puzzle, `givens_long` − `supervised` −0.031 [−0.037, −0.024]; `hindsight_long` − `givens_long` −0.007 [−0.015, −0.001].

**m1b, a quarter of the givens hidden** (2,308 hidden givens): accuracy on hidden givens 0.267 / 0.283 / 0.265, on blanks 0.451 / 0.452 / 0.401 (`supervised` / `givens_long` / `hindsight_long`). Blanks minus hidden givens per puzzle: +0.184 [+0.164, +0.204], +0.164 [+0.144, +0.183], +0.129 [+0.108, +0.150]. `givens_long` − `supervised`: +0.021 [+0.005, +0.038] on hidden givens, +0.001 [−0.005, +0.007] on blanks. `hindsight_long` − `givens_long` on blanks in this state: −0.050 [−0.058, −0.043].

**m2, a share of the blanks shown with the true digit** (`reveal.csv`), accuracy on the blanks still hidden:

| shown | `supervised` | `givens_long` | `hindsight_long` | `givens_long` − `supervised` |
|---|---|---|---|---|
| 0 | 0.724 | 0.694 | 0.686 | −0.031 [−0.037, −0.024] |
| 0.25 | 0.986 | 0.974 | 0.977 | −0.011 [−0.015, −0.008] |
| 0.5 | 1.000 | 0.999 | 1.000 | −0.001 [−0.002, 0.000] |
| 0.75 | 1.000 | 1.000 | 1.000 | 0 |

**m3, 4-step rollouts** (`steps.csv`):

| | `supervised` | `givens_long` | `hindsight_long` |
|---|---|---|---|
| wrong share of cells committed at step 1 / 2 / 3 / 4 | 0.034 / 0.111 / 0.185 / 0.258 | 0.045 / 0.145 / 0.247 / 0.328 | 0.059 / 0.155 / 0.238 / 0.301 |
| mean confidence of the cells committed at step 1 | 0.967 | 0.928 | 0.953 |
| puzzles with first wrong cell at step 1 / 2 / 3 / 4 | 0.237 / 0.195 / 0.045 / 0.003 | 0.326 / 0.201 / 0.063 / 0.006 | 0.374 / 0.120 / 0.045 / 0.009 |
| puzzles with no wrong cell | 0.521 | 0.404 | 0.452 |

Puzzles with a wrong cell at step 1: `givens_long` − `supervised` +0.090 [+0.048, +0.132]; `hindsight_long` − `givens_long` +0.048 [0.000, +0.096]. With 16 steps the first step is nearly error-free in all three (0.009 / 0.003 / 0.012 of puzzles) and first errors spread over steps 2 to 9.

## Predictions against results

1. Rules: under 0.03 on excluded digits, under 1% top-1 excluded, `givens_long` within 0.01 of `supervised`: **held** (0.0004 to 0.0016; no excluded top-1 in any model).
2. One digit left at 0.98 or above; `givens_long` 0.02 to 0.08 below `supervised` at 4 or more; overall difference below zero: **held** (1.000; −0.033; −0.031 [−0.037, −0.024]).
3. Hidden givens harder than blanks by more than 0.03 in every model; the gap of `givens_long` within ±0.03 of that of `supervised`: **held** (gaps 0.184 and 0.164, difference 0.020). Not predicted: on hidden givens `givens_long` is slightly above `supervised` (+0.021, interval just above zero).
4. `givens_long` more accurate as cells are shown, distance from `supervised` not larger at r = 0.75: **held**; the distance shrinks from 0.031 to 0.011 at r = 0.25 and to zero by r = 0.5.
5. More than 60% of failed puzzles fail first at step 1 in all three models: **not held** (49% `supervised`, 55% `givens_long`, 68% `hindsight_long`). `givens_long` 0.05 to 0.15 above `supervised` at step 1 with an interval above zero: **held** (+0.090, against a 0.117 gap in solve rate). `hindsight_long` and `givens_long` equal there: **not clearly held** (+0.048, interval touching zero; `hindsight_long` fails more often at step 1 and less at step 2).

Neither negative result named before the run occurred.

## What this shows / does not show

- Shows: the model trained only on printed digits never puts a digit where the row, column or box already has it (as the other two), and is exact on cells with one digit left. Its deficit against the supervised model is in the pass on the puzzle as given: 0.03 lower accuracy on blank cells, on cells with two or more digits left. That deficit is gone once a quarter to a half of the blanks is filled correctly, although the givens loss never showed it such states. In a 4-step rollout, where the first step commits a quarter of the blanks from that one pass, it has an error at step 1 in 0.33 of the hard puzzles against 0.24, which accounts for about three quarters of its gap in solve rate; with 16 steps the first step commits only the surest cells and the gap is small. Training only on givens did not make the model better on givens than on blanks beyond +0.02.
- Does not show: why the pass on the puzzle as given is weaker. Candidates not separated here: the supervised loss scores every blank of that state, the givens loss scores 25 to 30 givens per hard puzzle; hidden givens are partly ambiguous targets (the puzzle can lose uniqueness), blanks are not. Nothing beyond row, column and box elimination was tested as "rules". One model per arm, so the differences between `hindsight_long` and `givens_long` (for example −0.050 on blanks when givens are hidden) are differences between two runs. m2 uses true digits; a rollout that has already made an error is a different state. Val only.
