# 2.3c wd1 temperature sweep (τ ∈ {0.5, 1, 2}, 3 seeds each)

**Goal:** sensitivity of wd1 to the softmax temperature τ used for its group weights.

**Commands**
- All runs: `bash scripts/run_phase2_rl.sh ~/dllm-cache/venv/bin/python`.
- One run: `PYTHONPATH=src python -m phase2.rl_wd1 --variant wd1 --tau 0.5 --seed 0`.
- Plot: `PYTHONPATH=src python -m phase2.plots rl` writes `results/phase2/rl_tau_sweep.png`.

**Setup**
- Config: as in 2.3a; only τ changes. The τ = 1 runs are the same runs as 2.3a.
- Hardware: as in 2.3a. Each run took 61–68 s.
- Seeds: 0, 1, 2.

**Results.** The outputs are `results/phase2/rl/wd1_tau{0.5,1,2}_seed*.{csv,json}` and `results/phase2/rl_summary.csv`.

| τ | val at step 500 | test at step 500 |
|---|---|---|
| 0.5 | 0.725 ± 0.000 | 0.740 ± 0.009 |
| 1 | 0.731 ± 0.015 | 0.738 ± 0.005 |
| 2 | 0.737 ± 0.015 | 0.743 ± 0.009 |

Values are mean ± std over 3 seeds.

Why τ has so little room to matter: rewards are binary and groups of all-equal rewards are skipped. That leaves only about 7 distinct advantage patterns per group of 8, so τ only reshapes how weight is spread among the solved (or unsolved) rollouts.

**Shows:** in this setting, wd1 is insensitive to τ over a 4× range. All three settings land at 0.74 ± 0.01 test.

**Does not show:** τ sensitivity with graded rewards, or with larger groups. The sweep covers only 3 values.
