# 4.3 Choosing the RL start (hard band, 4 sampler steps; confirmed by the user on 2026-11-10)

**Goal:** apply the rule fixed in the spec to pick where RL starts, and measure there the two quantities that decided things in phase 3: the share of sampled rollouts that are valid and the share of groups of 8 with unequal rewards.

**Command**
```
PYTHONPATH=src python -m phase4.rl_start --size medium
```
- Config `configs/phase4.yaml` (`sizes.medium`), model `~/dllm-cache/phase4/sft/sft/final.pt` (step 43,500).
- 512 train puzzles per band (random sample, seed 0), 8 rollouts each at temperature 1 with the `confidence` order; binary reward (valid grid that keeps the givens).
- Hardware: Apple M5, 24 GB, macOS 27.0.1, torch 2.15.0, MPS, fp32. Wall-clock 3 minutes. Result: `results/phase4/4.3/rl_start.csv`.

| band | sampler steps | greedy solve (these train puzzles) | valid rollouts (T = 1) | groups with unequal rewards | groups all valid | groups none valid |
|---|---|---|---|---|---|---|
| easy | 4 / 8 / 16 | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| medium | 4 | 0.955 | 0.949 | 0.035 | 0.928 | 0.037 |
| medium | 16 | 0.984 | 0.984 | 0.000 | 0.984 | 0.016 |
| hard | 4 | 0.592 | 0.560 | 0.244 | 0.441 | 0.314 |
| hard | 8 | 0.764 | 0.765 | 0.133 | 0.701 | 0.166 |
| hard | 16 | 0.842 | 0.837 | 0.096 | 0.787 | 0.117 |

**Reading against the rule.** With the default sampler (16 steps) every band is above 0.7 (hard: 0.81 on test in 4.2), so preference 1 does not apply and preference 2 does: make the task harder, by larger k or a smaller step budget. Larger k is not available without a new dataset: the generator's yield falls to 0.85 at k = 56 and 0.21 at k = 58 (step 4.0), and a new band would need its own data report. A smaller step budget is available now: **hard band with 4 sampler steps** has a greedy solve rate of 0.535 on test (0.592 on these train puzzles), inside the 0.2 to 0.7 window.

At that start 56% of sampled rollouts are valid, so valid rollouts are common (the regime of phase 3's main runs, not the rare-reward regime of 3.6b). But only 24% of groups have unequal rewards: sampling at temperature 1 changes little because the model is very confident (4.2), so 3 groups in 4 are all valid or all invalid and carry no signal for the group-relative objectives (wd1 and its variants, GRPO).

**Shows:** where the plateau model sits on each band and step budget, and that the only start inside the window without new data is the hard band at 4 sampler steps.
**Does not show:** that RL can improve it. Note what the question becomes with this choice: "can RL teach the model to solve hard puzzles in 4 steps instead of 16", which is closer to learning to commit many cells at once than to learning Sudoku. The train puzzles here were seen in training (0.592 against 0.535 on test at 4 steps).
