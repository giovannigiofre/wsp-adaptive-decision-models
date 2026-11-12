# 4.8c: two more seeds of the harvest and hindsight arms of 4.8b

**Status: done (2026-11-11). Setup and predictions were committed before the run (`e910e50`).** Seen before writing them: all results of [4.8b](../4.8b/README.md).

**Goal.** In 4.8b (one seed per arm) `hindsight` ended above `harvest` on the hard val band at 4 and 81 sampler steps (+0.063 [+0.012, +0.114], +0.084 [+0.036, +0.132]) and not clearly at 16 (+0.045 [−0.009, +0.099]). Those intervals cover puzzle sampling only. Question: does the difference hold over training seeds?

**Setup.** Exactly the `harvest` and `hindsight` arms of 4.8b (10 rounds of 1,000 steps from the 4.8a state at step 9,546, batch 256, lr 3e-4, `configs/phase4.yaml`), with `diffusion_training.seed` 1 and 2. Run names `harvest_s1`, `harvest_s2`, `hindsight_s1`, `hindsight_s2`; one arm at a time. Analysis: the seed-0 runs of 4.8b are added; per sampler budget, the three seeds are pooled per val puzzle (mean of valid over seeds) and the paired difference `hindsight` − `harvest` gets a 95% bootstrap over puzzles; the three per-seed differences are listed beside it.

**Known limits, before the run.** The seed changes the training batches and masks only: the start model (4.8a, seed 0) is the same in every run, and the first harvest is greedy, so the buffer after round 1 is the same up to MPS non-determinism. The spread measured here is therefore smaller than that of a full rerun from scratch. The control (`givens`) is not repeated, so differences from the control still rest on one control seed. Three seeds give three differences: a range, not an interval. Val only. Outside load on the machine varies; wall-clock is not a clean timing.

## Predictions, written before the run

1. **Hard at 81 steps:** `hindsight` − `harvest` is positive in both new seeds, and the pooled difference is at least +0.04 with an interval above zero. Moderate confidence: the 4.8b difference was the largest of the three budgets.
2. **Hard at 4 steps:** positive pooled difference, +0.02 to +0.08; I would not be surprised if one of the two new seeds has it at zero or below.
3. **Hard at 16 steps:** pooled difference inside ±0.05 with an interval that includes zero or just clears it. The per-round curve of one run moves by about ±0.02, so the sign per seed is not predictable.
4. **Seed spread of one arm** on hard at 16 steps: the three final values of each arm lie within 0.06 of each other.
5. **Buffer:** every run ends above 0.93 of hard train puzzles.
6. **Medium at 4 steps:** `hindsight` is not above `harvest` (4.8b: −0.030 [−0.060, 0.000]); pooled difference between −0.05 and +0.01.

**Negative result that would be reported as such:** pooled `hindsight` − `harvest` with intervals including zero at all three budgets, i.e. the 4.8b advantage was seed noise.

## Command

```
ARM=harvest SEED=1 bash scripts/run_phase4.sh self-a2-seed ~/dllm-cache/venv/bin/python     # also SEED=2, ARM=hindsight
```

Each launch pauses after 100 minutes at the next round end; repeat until it prints "finished". Results: `results/phase4/4.8c/<arm>_s<seed>.csv`, `.json`, `_valid_val.csv`, `_solved_val.csv`. Checkpoints under `~/dllm-cache/phase4/4.8c/`.

```
PYTHONPATH=src python -m phase4.selfloop_pairs --pooled        # results/phase4/4.8c/pooled_pairs.csv
```

## Results (2026-11-11, 15:29 to 21:33)

Run one at a time in the order `hindsight_s1` (97 min), `harvest_s1` (91 min), `hindsight_s2` (88 min), `harvest_s2` (86 min), each in a single launch; evaluation included in the 6 h 4 min total. Apple M5 24 GB, MPS, torch 2.15.0, float32 (MPS is not bit-deterministic). Unrelated jobs loaded the machine for part of the time. No thermal warning at any launch change. The first harvest gave the same buffer in all runs (80,285 puzzles), as expected.

Hard val band (334 puzzles), valid share at step 19,546 per seed; seed 0 is the 4.8b run:

| arm | 4 steps: seed 0 / 1 / 2 | 16 steps: seed 0 / 1 / 2 | 81 steps: seed 0 / 1 / 2 |
|---|---|---|---|
| `harvest` | 0.311 / 0.362 / 0.344 | 0.653 / 0.668 / 0.701 | 0.719 / 0.722 / 0.754 |
| `hindsight` | 0.374 / 0.365 / 0.347 | 0.698 / 0.686 / 0.707 | 0.802 / 0.802 / 0.796 |
| `hindsight` − `harvest`, per seed | +0.063 / +0.003 / +0.003 | +0.045 / +0.018 / +0.006 | +0.084 / +0.081 / +0.042 |

Seeds pooled per puzzle, 95% bootstrap over puzzles of the paired difference (`pooled_pairs.csv`); the control is the single `givens` run of 4.8b:

| a − b, hard band | 4 steps | 16 steps | 81 steps |
|---|---|---|---|
| `hindsight` − `harvest` | +0.023 [−0.006, +0.054] | +0.023 [−0.008, +0.055] | +0.069 [+0.042, +0.095] |
| `hindsight` − `givens` | +0.153 [+0.111, +0.197] | +0.047 [+0.004, +0.091] | +0.112 [+0.067, +0.155] |
| `harvest` − `givens` | +0.130 [+0.086, +0.175] | +0.024 [−0.019, +0.068] | +0.043 [−0.000, +0.085] |

Medium band at 4 steps: `hindsight` − `harvest` −0.013 [−0.029, +0.003]; both arms above the control (+0.049 [+0.018, +0.083] and +0.062 [+0.031, +0.095]). Easy band: 1.000 in every run at 16 and 81 steps. Buffer at the end, hard train puzzles: 0.939 to 0.942 in the four new runs.

## Predictions against results

1. Hard at 81 steps, positive in both new seeds and pooled at least +0.04 with an interval above zero: **held** (+0.081, +0.042; pooled +0.069 [+0.042, +0.095]).
2. Hard at 4 steps, pooled +0.02 to +0.08, one new seed possibly at zero: **held at the bottom of the range, and the interval includes zero** (+0.023 [−0.006, +0.054]). Both new seeds are at +0.003: the +0.063 of 4.8b came from a low `harvest` seed 0 (0.311 against 0.362 and 0.344), not from hindsight.
3. Hard at 16 steps inside ±0.05, interval including zero or just clearing it: **held** (+0.023 [−0.008, +0.055]).
4. Final values of each arm within 0.06 on hard at 16 steps: **held** (`harvest` 0.653 to 0.701, `hindsight` 0.686 to 0.707).
5. Buffer above 0.93 of hard train puzzles: **held** (0.939 to 0.942).
6. Medium at 4 steps, `hindsight` not above `harvest`, between −0.05 and +0.01: **held** (−0.013).

The negative result named before the run did not occur at 81 steps; it did occur at 4 and 16 steps.

## What this shows / does not show

- Shows: over three training seeds from the same start model, hindsight masks add about +0.07 to the harvest loop on the hard band when the sampler takes 81 steps (one cell per step), in every seed, and nothing measurable at 4 or 16 steps. The 4-step advantage of hindsight over harvest read in 4.8b was a low harvest seed. Against the single control run, both harvest arms stay well ahead at 4 steps (+0.13, +0.15) and hindsight at 81 steps (+0.11); at 16 steps harvest is not distinguishable from the control and hindsight is +0.047 with an interval that just clears zero.
- Does not show: seed variance of the control or of the start model (one `givens` run, one 4.8a run), so every difference from the control still rests on one control seed. Why hindsight helps only at one cell per step was not examined (the hindsight states have givens masked, which is what the sampler meets late in an 81-step fill only loosely; not tested). Three seeds are a range, not an interval over seeds. Val only, no test evaluation; no plateau.

## Addendum: a second seed of the givens-only control

**Predictions written before the run (2026-11-12).** Seen: everything above, 4.8d, 4.8e. Every difference from the control in 4.8b and 4.8c rests on one `givens` run. `givens_s1`: the same control with `diffusion_training.seed` 1, 10 rounds from the 4.8a state to step 19,546 (`ARM=givens SEED=1 bash scripts/run_phase4.sh self-a2-seed ~/dllm-cache/venv/bin/python`).

1. Hard val at 16 steps at the end: 0.60 to 0.68 (seed 0: 0.650); at 81 steps 0.64 to 0.73 (0.689); at 4 steps 0.17 to 0.27 (0.210).
2. With the two control seeds pooled, `hindsight` − `givens` on hard keeps an interval above zero at 4 and 81 steps; at 16 steps it stays within ±0.03 of the +0.047 read with one control seed, interval possibly including zero.
