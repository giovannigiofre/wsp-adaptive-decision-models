# 2.3b Ablation: positive-only weights (w⁻ = 0), 3 seeds

**Goal:** the wd1 paper's argument for its negative term is that positive-only weighting is unstable. This run checks whether that shows up here.

**Command:** `bash scripts/run_phase2_rl.sh ~/dllm-cache/venv/bin/python`. A single run is `PYTHONPATH=src python -m phase2.rl_wd1 --variant positive_only --tau 1 --seed 0`.

- Config: identical to 2.3a except that w⁻ = 0, so the loss is −Σ w⁺ · log p̂.
- Hardware: same as 2.3a. Seeds 0, 1, 2.
- Wall-clock: 66 s per run for seeds 1 and 2. Seed 0 was resumed from `state.pt` after an interruption, and its JSON records only the 55 s of the resumed session.
- Results:
  - `results/phase2/rl/positive_only_tau1_seed{0,1,2}.{csv,json}`
  - `results/phase2/rl_wd1_vs_positive_only.png`

| | val minimum (step) | val at step 500 | test at step 500 |
|---|---|---|---|
| seed 0 | 0.108 (75) | 0.925 | 0.920 |
| seed 1 | 0.095 (100) | 0.954 | 0.925 |
| seed 2 | 0.121 (150) | 0.954 | 0.950 |
| mean ± std | | 0.944 ± 0.017 | 0.932 ± 0.016 |
| wd1 τ = 1 (2.3a) | no dip | 0.731 ± 0.015 | 0.738 ± 0.005 |

What happens:
- **Collapse, then recovery.** In every seed, val solve first drops from 0.42 to about 0.10 within 75–150 steps and train reward falls to about 0.09. Then, from about step 200, both climb steeply. All three seeds overtake wd1 at about step 400 and end at 0.92–0.95 test.
- **Rollouts stay likely.** During the dip, the log p̂ of both correct and wrong rollouts *rises* (−7 → −5 and −9.5 → −6). The model becomes more confident in whatever it samples, which is the sharpening failure the wd1 paper describes.
- **The skip rule.** About 60–67% of groups are skipped (all rewards equal) in the first 15 steps. The dip drives that share higher still, because at low reward most groups are all-zero.

**Shows:** the early instability that motivates w⁻ is real here: all 3 seeds lose about 75% of their solve rate before recovering. Avoiding that dip is what wd1's negative term buys.
**Does not show:** that positive-only is worse at the end. At 500 steps it is clearly *better* (0.93 vs 0.74, non-overlapping across seeds). This does not reproduce the paper's claim that positive-only ends worse.

Possible reasons, none of them tested:
- Binary rewards on a tiny task, where an all-zero group carries no signal and a single solved rollout gets all of w⁺.
- The weak start is a mid-warm-up snapshot that is one sharp step away from solving the task.
- wd1's unbounded negative term, with no reference/KL, may be pushing probability mass into places that slow progress.

A longer run, a KL term or a smaller lr might change the ranking.
