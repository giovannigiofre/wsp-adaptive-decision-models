# 3.6b A rarer reward and a weaker checker

**Goal:** robustness check added after the phase 3 summary. Two features of the toy favour the objectives that imitate valid rollouts: valid rollouts exist from the start, and a valid grid is always a correct one. Weaken each in turn.

| Part | Change | Start |
|---|---|---|
| rarer reward | start from SFT step 90, val solve rate 0.007 (step 100, used in 3.1d/3.1g, had 0.11) | SFT step 90 |
| weaker checker | reward `lenient`: 1 if rows and columns are valid and the givens kept; the 2×2 boxes are ignored. A Latin square that breaks the boxes is rewarded. Evaluation stays the true solve rate | SFT step 120 (phase 2 start) |

Arms in both parts, 3 seeds, 1500 steps: wd1 (τ = 1), wd1 with floor −12, filtered imitation, self-training with the SFT loss.

**Prediction** (written before the runs):
- Rarer reward: with about 1 valid rollout in 150 at the start, all four still learn. Self-training and filtered imitation dip towards 0 first and end at 0.95 or more; the floored wd1 ends at 0.9 or more; plain wd1 ends between 0.6 and 0.9.
- Weaker checker: the lenient reward is driven to 0.95 or more by every arm, but the true solve rate ends lower than with the strict reward in every arm. The gap is largest for the two imitation arms (at least 5 points below their strict-reward results of 0.997 and 0.999), because they imitate rewarded grids whether or not the boxes are right. No arm's true solve rate falls below its start (0.48).

**Command**
```
bash scripts/run_phase3.sh 3.6b ~/dllm-cache/venv/bin/python
PYTHONPATH=src python -m phase3.analysis 3.6b
```

- Config: `configs/phase3.yaml`; `rl.reward=lenient` for the second part; `--init ~/dllm-cache/phase2/sft/step_000090.pt --tag _sft90` for the first.
- Code: `src/phase3/rewards.py` (`lenient`). Test: `tests/test_rewards.py`.
