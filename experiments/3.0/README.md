# 3.0 Generic RL trainer and reproduction of the phase 2 runs

**Goal:** check that the new phase 3 trainer (`src/phase3/rl.py`, pluggable objective and reward, extra diagnostics) gives the same results as the phase 2 script before anything is built on it.

**Prediction** (written before the runs):
- wd1 (β = 1, τ = 1) ends at 0.74 ± 0.01 test solve rate and never dips on val.
- Positive-only (β = 0, τ = 1) ends at 0.93 ± 0.02 after an early collapse to about 0.10–0.15 val near steps 75–150.
- Per-seed curves follow the phase 2 ones closely at first and then drift, because the order of random draws is the same but MPS is not bit-deterministic. In a 6-step smoke run the first three steps matched phase 2 to every printed digit.
- New diagnostics: at τ = 1 the share of w⁺ mass on invalid samples is 0.35–0.55 in the first steps, and the effective sample size of the weights stays above 6 of 8.

**Command**
```
for s in 0 1 2; do
  PYTHONPATH=src python -m phase3.rl --seed $s                                    # wd1
  PYTHONPATH=src python -m phase3.rl --seed $s --set rl.objective_params.beta=0   # positive-only
done
PYTHONPATH=src python -m phase3.analysis repro
```

- Config: `configs/phase3.yaml`, which reuses data, model and sampler from `configs/phase2.yaml`.
  - Init: `sft_weak.pt` (SFT step 120, val 0.42, test 0.48), pinned by sha256 in the config.
  - RL settings identical to 2.3a/2.3b: 500 steps, B = 32 puzzles × G = 8 rollouts (confidence, T = 1, S = 8), binary reward, μ = 4, AdamW lr 1e-5, grad-clip 1.0.
- Code: `src/phase3/{rl,objectives,rewards}.py`. Tests: `tests/test_objectives_sign.py`, `tests/test_rewards.py`.
- Diagnostics added per step: share of w⁺ on invalid samples, effective sample size of w⁺, gradient norm, one-step predictive entropy, and val solve rate at S = 1 next to S = 8.
