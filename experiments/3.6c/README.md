# 3.6c Is it the fixed order? wd1 with the likelihood along the rollout's own path

**Goal:** follow-up to [3.6a](../3.6a/README.md). There, putting the fixed-order (row-major) sequential likelihood in the loss made every arm slower and made wd1 worse than its start. Candidate reason (a) in that write-up: the loss scores a grid along a path (cell 1, 2, ..., 16) that the confidence-order sampler never uses. Two checks.

**Part 1, training.** Same three arms as 3.6a, 3 seeds, 1000 steps, with `rl.likelihood=trajectory`: each cell is scored with the input the sampler actually saw when it committed it (givens plus the cells committed at earlier sampler steps). This is the chain rule along the rollout's own path, one forward pass per sampler step (8), cells committed in the same step treated as independent, and the choice of order treated as fixed (no gradient through it).

| Arm | Likelihood in the loss |
|---|---|
| wd1 (τ = 1) | rollout path |
| wd1 with a floor on the negative term | rollout path; floor at −9, the same value as the sequential arm of 3.6a (not tuned) |
| positive-only (τ = 1) | rollout path |

**Part 2, evaluation only.** Take the final checkpoints of the one-step runs (3.1b/3.1c, 1500 steps) and of the sequential runs (3.6a, 1000 steps) and solve the test set two ways: the usual confidence order, and a fixed row-major order, one cell per step, greedy values. If the sequential loss trained "a path the sampler does not use", those models should be better along the row-major path than along the confidence path.

**Prediction** (written before the runs and before the evaluation):
- Part 1. If the order mismatch is the reason, the rollout-path likelihood should behave like the one-step one. wd1: val at step 1000 of 0.70 or more (sequential 0.333, one-step 0.842). Floored wd1: 0.90 or more (sequential 0.815, one-step 0.982). Positive-only: 0.95 or more, with an early dip below 0.3.
- Part 2. The models trained with the sequential likelihood solve at least 0.15 more of the test set in row-major order than in confidence order (8 steps); the models trained with the one-step likelihood do not (row-major no better than confidence order).

**Command**
```
bash scripts/run_phase3.sh 3.6c ~/dllm-cache/venv/bin/python
PYTHONPATH=src python -m phase3.fixed_order_eval
PYTHONPATH=src python -m phase3.analysis 3.6c
```

- Config: `configs/phase3.yaml` with `rl.likelihood=trajectory`, `rl.max_rl_steps=1000`; floor −9.
- Code: `src/phase3/rl.py` (`trajectory_cell_logp`), `src/phase3/fixed_order_eval.py`. Test that the path likelihood reduces to the one-step one when everything is committed in one step, and to the sequential one when cells are committed one per step in row-major order: `tests/test_objectives_sign.py`.
- Hardware: Apple M5, 24 GB, MPS, float32, torch 2.15.0. Seeds 0–2.
- Results: `results/phase3/3.6c/`.
