# 3.1g Do the better variants survive a different starting model?

**Goal:** follow-up added after 3.1d, not in the original spec. From the phase 2 start (SFT step 120) four variants beat plain wd1 and did not collapse: filtered imitation and self-training with the SFT loss (3.2), wd1 with a floored negative term (3.1c) and wd1 with a small KL penalty to the start (3.2b). 3.1d showed that results from that one start do not carry over: positive-only at τ = 1 goes to 0 from SFT step 100 and collapses late from the model converged on 10 solutions. Rerun the four variants from those two starts.

**Prediction** (written before the runs):
- From SFT step 100 (val 0.11, 1–2% valid rollouts):
  - Filtered imitation and self-training do not collapse, because they give no weight to invalid rollouts, and end at 0.95 or more.
  - wd1 + KL 0.01 ends below plain wd1 (0.88), between 0.6 and 0.88: the KL ties it to a very weak start.
  - wd1 with floor −12 behaves like wd1 early (its rollouts start above the floor) and ends at 0.9 or more.
- From the model converged on 10 solutions (val 0.65, sharp):
  - Filtered imitation and self-training end at 0.9 or more on test with no late collapse.
  - wd1 + KL 0.01 ends within 0.05 of plain wd1 (0.79).
  - wd1 with floor −12 is unstable: many of its rollouts are below the floor from the start, where the objective acts like positive-only at τ = 1. In at least one seed the final val is more than 0.1 below its best.

**Command**
```
bash scripts/run_phase3.sh 3.1g ~/dllm-cache/venv/bin/python
PYTHONPATH=src python -m phase3.analysis 3.1g
```

- Config: `configs/phase3.yaml`; objectives `filtered`, `self_diffusion`, wd1 with `rl.ref_kl=0.01`, wd1 with `floor=-12`; starts SFT step 100 and the 10-solution model; seeds 0–2, 1500 steps. wd1 and positive-only from the same starts are the 3.1d runs.
- Code: as in 3.1c, 3.2 and 3.2b.
