# 4.9a: what did the givens-only model learn? Measurements on the three long models, no training

**Status: predictions written before the run (2026-11-12).** Seen before writing them: all results of 4.8a to 4.8e. Nothing of this script was run.

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
