# 2.3a wd1 RL from the weak SFT checkpoint (τ = 1, 3 seeds)

**Goal:** check whether a hand-written wd1 (weighted policy optimisation with positive and negative softmax weights, no reference model or KL) raises the test solve rate of the weak SFT model.

**Command** (runs 2.3a, 2.3b and 2.3c in sequence, skipping finished runs; each run can be resumed from `state.pt`)
```
bash scripts/run_phase2_rl.sh ~/dllm-cache/venv/bin/python
# one run: PYTHONPATH=src python -m phase2.rl_wd1 --variant wd1 --tau 1 --seed 0
PYTHONPATH=src python -m phase2.plots rl
```

- Config: `configs/phase2.yaml` (`rl:`).
  - Init: `sft_weak.pt` (step 120, val 0.42, test 0.48).
  - 500 steps. Each step uses B = 32 train puzzles × G = 8 rollouts (confidence strategy, T = 1, S = 8). Reward = 1 if the grid is a valid completion, else 0.
  - Advantage A = r − group mean. Groups whose rewards are all equal are skipped. w⁺ = softmax(A/τ), w⁻ = softmax(−A/τ).
  - Loss = mean over kept groups of Σ (w⁻ − w⁺) · log p̂.
  - log p̂ is the one-step estimate: mask every non-given cell, additionally mask givens with p = 0.15, and sum log p over the non-given cells.
  - μ = 4 gradient steps per batch. AdamW lr 1e-5, wd 0, grad-clip 1.0. Dropout is off during RL.
  - Val is evaluated every 25 steps (greedy, confidence, S = 8, 305 puzzles). Test is evaluated at the end (1,194 puzzles).
- Code: `src/phase2/rl_wd1.py`. Sign/weight tests: `tests/test_wd1_sign.py`.
- Hardware: Apple M5 24 GB, macOS 27.0.1, Python 3.12.13, torch 2.15.0, MPS, fp32 (shared with a concurrent LLaDA-8B job). Seeds 0, 1, 2; MPS is not bit-deterministic.
- Wall-clock: 61–67 s per run.
- Results:
  - `results/phase2/rl/wd1_tau1_seed{0,1,2}.{csv,json}`
  - `results/phase2/rl_runs.csv`, `results/phase2/rl_summary.csv`
  - `results/phase2/rl_wd1_vs_positive_only.png`
  - Checkpoints (local only): `~/dllm-cache/phase2/rl/<run>/{best,final}.pt`

| | val solve (step 500) | test solve (step 500) |
|---|---|---|
| weak SFT (start) | 0.42 | 0.482 |
| wd1 τ = 1, mean ± std over 3 seeds | 0.731 ± 0.015 | 0.738 ± 0.005 |

Training is smooth and monotone. Train reward goes from 0.27 to 0.73. Over training, the one-step log p̂ of wrong rollouts falls (−9.5 → −15) much faster than that of correct ones (−7 → −11), so the negative term is doing most of the work. Val is still rising slowly at step 500.

**Shows:** wd1 lifts the weak model from 48% to 74% test solve in about one minute of Mac compute, consistently across 3 seeds.
**Does not show:** what happens after 500 steps, or with a reference/KL term. It also falls well short of the fully trained model (100%), and of positive-only training at the same step count (2.3b).
