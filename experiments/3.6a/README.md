# 3.6a Is the likelihood proxy the problem? wd1 with the sequential likelihood

**Goal:** robustness check added after the phase 3 summary. Every claim about wd1 "pushing down the likelihood of its own valid rollouts" was read through the one-step mean-field estimate, which phase 2 found to be biased low by about 3 nats. Replace it in the loss by the fixed-order sequential likelihood (16 forward passes, the estimate phase 2 used as the reference) and see whether the story changes.

Arms, 3 seeds, **1000 steps** (the sequential likelihood costs about 1 s per step, so the horizon is shorter than the standard 1500; comparisons with the one-step runs are made on the val curves at matched steps):

| Arm | Likelihood in the loss |
|---|---|
| wd1 (τ = 1) | sequential |
| wd1 with a floor on the negative term | sequential; floor at −9, i.e. the one-step floor of −12 shifted by the 3-nat bias |
| positive-only (τ = 1) | sequential |

The diagnostics `logp_valid` / `logp_invalid` are then sequential log-likelihoods; `pred_entropy` is not logged.

**Prediction** (written before the runs):
- wd1 with the sequential likelihood is still slow: val at step 1000 within 0.05 of the one-step wd1 (0.843), and the sequential log-likelihood of its valid rollouts still falls by at least 3 nats over training. The drag belongs to the objective, not to the proxy.
- The floored wd1 reaches a val solve rate of 0.95 or more by step 1000, as its one-step version did.
- Positive-only still collapses early (lowest val below 0.2) and recovers to 0.95 or more by step 1000.

**Command**
```
bash scripts/run_phase3.sh 3.6a ~/dllm-cache/venv/bin/python
PYTHONPATH=src python -m phase3.analysis 3.6a
```

- Config: `configs/phase3.yaml` with `rl.likelihood=sequential`, `rl.max_rl_steps=1000`; floor −9.
- Code: `src/phase3/rl.py` (`sequential_cell_logp`). Test against phase 2's `sequential_loglik`: `tests/test_objectives_sign.py`.
