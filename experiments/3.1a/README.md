# 3.1a Was step 500 too early? wd1 and positive-only for 2000 steps

**Goal:** phase 2 compared wd1 and positive-only at 500 RL steps, when wd1 was still improving. Run both four times longer and see whether the ranking holds. The result also fixes the standard horizon for the rest of phase 3.

**Prediction** (written before the runs):
- wd1 keeps climbing but slows down as more groups are skipped. It ends at 0.85 ± 0.05 test solve rate and does not catch positive-only.
- Positive-only reaches at least 0.97 by step 1000 and stays at or above 0.95 until step 2000, with no second collapse.
- By the end more than 80% of the groups are skipped in both variants, because all 8 rollouts are valid.

**Rule for the standard horizon** (fixed before the runs): the first evaluation step at which the seed-mean val solve rate of *both* variants is within 0.02 of its value at step 2000, rounded up to a multiple of 500.

**Command**
```
bash scripts/run_phase3.sh 3.1a ~/dllm-cache/venv/bin/python
PYTHONPATH=src python -m phase3.analysis 3.1a
```

- Config: `configs/phase3.yaml` with `rl.max_rl_steps=2000`; wd1 is `beta=1`, positive-only is `beta=0`, both at τ = 1. Seeds 0–4. Everything else as in 3.0. For seeds 0–2 the first 500 steps are the 3.0 runs again.
- Code: `src/phase3/rl.py`, `scripts/run_phase3.sh`.
