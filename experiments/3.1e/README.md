# 3.1e Does a graded reward help?

**Goal:** the 0/1 reward gives no signal when all 8 rollouts of a puzzle are invalid, and treats a grid with one wrong cell like a random grid. Try a graded reward that needs no labels: the share of the 12 units (4 rows, 4 columns, 4 boxes) that contain each digit once. It equals 1 exactly when the grid is valid. Evaluation stays the binary solve rate.

**Prediction** (written before the runs):
- Fewer groups are skipped early on (the rewards within a group are rarely all equal).
- wd1 with the graded reward climbs faster in the first 300 steps and ends within 0.03 of wd1 with the binary reward.
- Positive-only at τ = 1 dips at least as deeply as with the binary reward, because the reward spread within a group is smaller and the weights are even flatter. At the best τ from 3.1b the dip is much smaller. It still ends at 0.95 or more in both cases.

**Command**
```
bash scripts/run_phase3.sh 3.1e ~/dllm-cache/venv/bin/python
PYTHONPATH=src python -m phase3.analysis 3.1e
```

- Config: `configs/phase3.yaml` with `rl.reward=graded`; β ∈ {1, 0} × τ ∈ {1, best τ from 3.1b}, seeds 0–2, standard horizon. The binary-reward cells are the 3.1b runs.
- Code: `src/phase3/rewards.py` (`graded`). Test: `tests/test_rewards.py`.
