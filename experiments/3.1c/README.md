# 3.1c How much negative push is the right amount? β sweep and a floored negative term

**Goal:** wd1 and positive-only are the two ends of one family, loss = Σ (β·w⁻ − w⁺)·log p̂, with β = 1 and β = 0. Sweep β to see how the early collapse and the final solve rate depend on the size of the negative term. Also try a floored negative term, which stops pushing a rollout down once its log p̂ is below a floor. The floor is motivated by the phase 2 logs (wd1 lowered the likelihood of valid rollouts too) and is not from the wd1 paper.

Note from 3.0: every gradient is clipped to norm 1, so β changes the direction of the update, not its size.

**Prediction** (written before the runs), at τ = 1:
- With 2 valid rollouts out of 8, the net coefficient on an invalid rollout is β·0.148 − 0.087, which is negative (a push *up*) for β below about 0.59. So β = 0.25 and β = 0.5 still dip early, less deeply as β grows; β = 1 and β = 2 do not dip.
- The final solve rate decreases with β: β = 0.25 ends within 0.02 of positive-only, β = 0.5 in between, β = 2 ends below β = 1.
- Floor: with the floor at −12 or −15 the estimated likelihood of valid rollouts falls less than under plain wd1, and the final test solve rate is at least as high as plain wd1's (within noise or better).

**Command**
```
bash scripts/run_phase3.sh 3.1c ~/dllm-cache/venv/bin/python
PYTHONPATH=src python -m phase3.analysis 3.1c
```

- Config: `configs/phase3.yaml`; β ∈ {0.25, 0.5, 2} at τ = 1 and at the best τ from 3.1b, plus floor ∈ {−12, −15} at τ = 1, β = 1. Seeds 0–2, standard horizon. The β = 0 and β = 1 cells are the 3.1b runs.
- Code: `src/phase3/objectives.py` (`wd1`, `weighted_loss`). Test of the floor: `tests/test_objectives_sign.py`.
