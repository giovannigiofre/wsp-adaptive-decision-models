# 3.1f Does a reference policy temper wd1?

**Goal:** phase 2 listed "no KL/reference term" as a possible reason for wd1's lower result. Before implementing anything, the upstream wd1 code was read (github.com/xiaohangt/wd1, commit `1a2e254`, pinned in `configs/phase3.yaml`):

- The loss in `wd1/trainers/rev_grpo_trainer.py` is −Σ softmax(A)·log p̂ + Σ softmax(−A)·log p̂ with A = r − group mean. That is the phase 2 loss with τ = 1. There is no reference or KL term in the loss.
- The positive-only trainer (`rev_grpo_trainer_psr.py`) drops the second sum, as in phase 2.
- The reference policy appears only in the variant `wll_d1_neg_ref` (`rev_grpo_ref_pol_trainer.py`), and there it enters the *sampling*: rollouts are drawn from mixed logits, (1 − β_ref)·current + β_ref·reference, with β_ref = 0.04 by default. The loss is unchanged.

Toy equivalent: `rl.ref_mix` samples the rollouts from (1 − mix)·current + mix·frozen-start logits. Differences from upstream: the reference is the frozen starting checkpoint rather than a LoRA-free base model, and the model is the 0.8M toy.

**Prediction** (written before the runs): mix = 0.04 is indistinguishable from no reference. Mix = 0.2 and 0.5 lower the share of skipped groups late in training, because the rollouts stay more diverse. For all three values the final test solve rate is within 0.03 of wd1 without a reference.

**Command**
```
bash scripts/run_phase3.sh 3.1f ~/dllm-cache/venv/bin/python
PYTHONPATH=src python -m phase3.analysis 3.1f
```

- Config: `configs/phase3.yaml` with `rl.ref_mix` ∈ {0.04, 0.2, 0.5}; wd1 (β = 1, τ = 1), seeds 0–2, standard horizon. The no-reference cell is the 3.1b run.
- Code: `src/phase3/rl.py` (`MixedLogits`).
