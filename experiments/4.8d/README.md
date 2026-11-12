# 4.8d: the hindsight loop continued to the step count of the supervised plateau

**Status: done (2026-11-12). Setup and predictions were committed before the run (`9f5f5d4`).** Seen before writing them: all results of [4.8b](../4.8b/README.md) and [4.8c](../4.8c/README.md).

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

## Results (2026-11-11 22:14 to 2026-11-12 00:58)

Two launches (22:14 to 23:57, paused after round 25; 23:57 to 00:58), 2 h 44 min for rounds 11 to 34 with the final evaluation. Apple M5 24 GB, MPS, torch 2.15.0, float32, seed 0 (MPS is not bit-deterministic). Unrelated jobs loaded the machine for part of the time; wall-clock is not a clean timing. No thermal warning at either launch change.

Hard val band (334 puzzles), solve rate per round at 16 sampler steps (`hindsight_long.csv`):

| rounds | steps | mean hard at 16 steps | mean hard at 4 steps |
|---|---|---|---|
| 8 to 10 | 17,546 to 19,546 | 0.698 | 0.355 |
| 20 to 22 | 29,546 to 31,546 | 0.744 | 0.434 |
| 32 to 34 | 41,546 to 43,546 | 0.730 | 0.450 |

The highest single round is 0.775 (rounds 25 and 27); rounds 28 to 34 lie between 0.713 and 0.760. Supervised 4.1 over its last three evaluations (steps 42,500 to 43,500): 0.811 at 16 steps (0.799, 0.829, 0.805).

Final model (step 43,546), valid share with 95% bootstrap over puzzles (`hindsight_long_valid_val.csv`):

| band | 4 steps | 16 steps | 81 steps |
|---|---|---|---|
| easy | 1.000 | 1.000 | 1.000 |
| medium | 0.949 [0.922, 0.973] | 0.997 [0.991, 1.000] | 1.000 |
| hard | 0.452 [0.398, 0.509] | 0.731 [0.683, 0.778] | 0.841 [0.802, 0.880] |

Buffer at the end: 0.987 of hard train puzzles, 0.9998 of medium, all easy. New valid grids per round over rounds 30 to 34: 18, 13, 19, 16, 19, out of 468 to 534 unsolved puzzles tried.

## Predictions against results

1. Hard at 16 steps, mean of the last three rounds 0.74 to 0.80, less than 0.07 below supervised: **not held, just below the range** (0.730; 0.075 below the 0.805 of the 4.1 README, 0.081 below the mean of the last three supervised evaluations). It does not pass the supervised plateau, as expected.
2. Hard at 4 steps 0.42 to 0.52, gap to supervised smaller than at 16 steps: **range held** (0.450, final 0.452); **the gap is not smaller** (0.07 at 4 steps against 0.075 at 16; not distinguishable).
3. Hard at 81 steps 0.84 to 0.90: **held at the bottom** (0.841 [0.802, 0.880]).
4. Buffer above 0.97 of hard train puzzles, under 100 new grids per round at the end: **held** (0.987; 13 to 19).
5. Easy 1.000 at 16 steps in every round, medium at 16 steps ending at 0.98 or above: **held** (medium 0.997 at the end, lowest round 0.979).
6. Gain over rounds 11 to 22 larger than over rounds 23 to 34: **held, and the second half has no gain** (+0.046 from rounds 8-10 to rounds 20-22, then −0.014 to rounds 32-34, inside the round-to-round noise of about ±0.02).

The first negative result named before the run occurred in part: hard at 16 steps is not flat from round 10 (it rises by about 0.045 until round 20 to 27), but it is flat from about round 20 and ends below the supervised model. No band fell.

## What this shows / does not show

- Shows: continued to the step count of the supervised plateau, the label-free hindsight loop stops rising on the hard val band at about 0.73 to 0.75 at 16 sampler steps (0.45 at 4, 0.84 at 81), 0.07 to 0.08 below the supervised model at the same step, while its buffer holds 98.7% of the hard train puzzles. With almost every training solution found, the remaining gap is not a lack of solved puzzles in the buffer.
- Does not show: why the gap remains (the 9,546 givens-only steps at the start, the hindsight masking, or the mix of buffer and givens batches were not separated); seed variance (one run; single rounds move by ±0.02 to 0.03, so the plateau level is known to about that); whether a longer run or a learning-rate decay would close the gap; anything on the test set. The supervised 81-step value at this step was not measured like for like.
