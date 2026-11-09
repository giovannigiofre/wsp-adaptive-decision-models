# 3.2 wd1 next to simpler objectives

**Goal:** find out whether what we saw for wd1 and positive-only is specific to them or shared by other objectives. All methods are hand-written and use the same rollouts, the same one-step likelihood estimate and the same budget (rollouts, gradient steps, learning rate).

| Method | Loss | Uses labels |
|---|---|---|
| wd1 (τ = 1) | Σ (w⁻ − w⁺)·log p̂ with softmax weights | no |
| positive-only (τ = 1) | −Σ w⁺·log p̂ | no |
| REINFORCE | −Σ A·log p̂, A = r − group mean (linear instead of softmax weights) | no |
| REINFORCE, normalised | the same with A divided by the group std | no |
| filtered imitation | −mean of log p̂ over the valid rollouts of a group (the τ → 0 limit of positive-only) | no |
| self-training with the SFT loss | masked-diffusion loss on the model's own valid rollouts, all groups | no |
| d1-style GRPO | clipped per-cell ratio against the pre-update policy (ε = 0.5), KL to the frozen start (0.04); adapted from the d1 trainer shipped in the wd1 repository | no |
| supervised reference | masked-diffusion loss on the true solutions of the same puzzles | **yes** |

**Prediction** (written before the runs):
- The methods that only push valid rollouts up (filtered imitation, self-training) do not collapse (lowest val at least 0.35) and end at 0.97 or more. They beat the "contrast" methods (wd1, REINFORCE, GRPO) at this horizon.
- REINFORCE, with or without normalisation, behaves like wd1: no dip, final test solve rate within 0.03 of wd1.
- GRPO: no dip, final within 0.05 of wd1.
- The supervised reference reaches 0.99 within 500 steps and is the fastest of all.

**Command**
```
bash scripts/run_phase3.sh 3.2 ~/dllm-cache/venv/bin/python
PYTHONPATH=src python -m phase3.analysis 3.2
```

- Config: `configs/phase3.yaml` (`rl.objective`, `rl.objective_defaults`), seeds 0–4, standard horizon.
- Code: `src/phase3/objectives.py`, `src/phase3/rl.py`. Sign tests for every objective: `tests/test_objectives_sign.py`.
