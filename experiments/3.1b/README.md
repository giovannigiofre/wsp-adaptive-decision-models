# 3.1b What happens when the weights are actually sharp? τ sweep for wd1 and positive-only

**Goal:** in phase 2 (τ ∈ {0.5, 1, 2}) the softmax weights were almost uniform, so τ was never really tested. With 0/1 rewards and m valid rollouts out of G, the share of w⁺ on invalid rollouts is (G − m) / ((G − m) + m·e^(1/τ)): for m = 2 that is 52% at τ = 1, 5% at τ = 0.25 and about 0% at τ = 0.1. If positive-only collapses early because it reinforces its own mistakes, the collapse should shrink as τ falls.

**Prediction** (written before the runs):
- Positive-only: the early dip gets shallower as τ falls. The lowest seed-mean val solve rate is at least 0.30 for τ ≤ 0.1, against 0.11 at τ = 1. The final solve rate is at least 0.97 at every τ.
- wd1: lower τ makes it faster. At τ ≤ 0.1 its final val solve rate is at least 0.03 above τ = 1, but still below positive-only at the same τ. It never dips below the start at any τ.

**Rule for the "best τ"** used by 3.1c and 3.1e (fixed before the runs): the τ in {0.25, 0.1, 0.05} with the highest seed-mean final val solve rate for wd1; values within 0.01 of the best count as ties and the largest such τ is taken.

**Command**
```
bash scripts/run_phase3.sh 3.1b ~/dllm-cache/venv/bin/python
PYTHONPATH=src python -m phase3.analysis 3.1b
```

- Config: `configs/phase3.yaml`; τ ∈ {1, 0.25, 0.1, 0.05} × β ∈ {1 (wd1), 0 (positive-only)}, seeds 0–2, at the standard horizon fixed by 3.1a. Everything else as in 3.0.
- Code: `src/phase3/rl.py`, `src/phase3/objectives.py`.
