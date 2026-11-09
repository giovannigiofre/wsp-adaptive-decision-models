# 3.2b Why did GRPO do better than wd1 and REINFORCE? The KL term to the starting model

**Goal:** follow-up added after 3.2, not in the original spec. In 3.2, d1-style GRPO ended at 0.977 test against 0.884 for REINFORCE and 0.865 for wd1. Its clipping was never active, so the candidate explanation is its KL penalty towards the frozen starting model. This is also the "no KL/reference term" hypothesis from phase 2. (3.1f tests the other way a reference can enter, through the sampling, which is what the upstream wd1 code does.)

Arms, all at 1500 steps, seeds 0–2:

| Arm | What it tests |
|---|---|
| GRPO with KL weight 0 | remove the KL term from GRPO |
| REINFORCE + KL (weight 0.04) | add the same per-cell KL estimate to REINFORCE |
| wd1 + KL (weights 0.01 and 0.04) | add it to wd1. Not part of the wd1 paper or code |

The KL estimate is the one in the d1 GRPO trainer: exp(d) − d − 1 with d = log p_ref − log p for each cell of each rollout, under the same prompt mask. For the weighted objectives it is summed over the cells and rollouts of a kept group and added with weight `rl.ref_kl`.

**Prediction** (written before the runs):
- GRPO without KL falls back to REINFORCE's level: 0.86–0.91 test.
- REINFORCE + KL reaches GRPO's level: 0.95 or more.
- wd1 + KL ends at 0.93 or more for at least one of the two weights, and the estimated log-likelihood of its valid rollouts stays above −10 late in training (plain wd1: about −12).

**Command**
```
bash scripts/run_phase3.sh 3.2b ~/dllm-cache/venv/bin/python
PYTHONPATH=src python -m phase3.analysis 3.2b
```

- Config: `configs/phase3.yaml` (`rl.ref_kl`, `rl.objective_defaults.grpo.kl`).
- Code: `src/phase3/rl.py`, `src/phase3/objectives.py` (`grpo_loss`).
