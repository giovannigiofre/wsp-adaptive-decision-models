# 3.3b The confidence token on a frozen denoiser

**Goal:** follow-up added after 3.3, not in the original spec. In 3.3 the confidence token was trained while the grid policy was improving and with wd1's nearly flat weights, and it failed. Here the denoiser is frozen (the wd1-tuned checkpoints from 3.1b, test solve rate 0.84–0.88), so the solve rate cannot change and the token's target is stationary. Only the query vector and the confidence head are trained. The RL arm uses plain REINFORCE with the group-mean baseline on the Brier-shaped reward; the probe arm minimises the expected Brier loss.

**Prediction.** Written before the runs, but after a 40-step smoke test of the code, which showed both arms improving the Brier score from 0.10 to about 0.06–0.07 on val. The numbers below are therefore informed by that.
- probe, last: AUROC at least 0.95 and Brier at most 0.05, better than the 2-parameter recalibration of the implicit confidence (0.067).
- rl, last: AUROC at least 0.90 and Brier at most 0.06, with no collapse onto a single value in at least 2 of 3 seeds.
- first (both modes): AUROC between 0.60 and 0.80, Brier at most 0.02 below the base rate.

**Command**
```
bash scripts/run_phase3.sh 3.3b ~/dllm-cache/venv/bin/python
PYTHONPATH=src python -m phase3.analysis 3.3b
```

- Config: `configs/phase3.yaml` with `rl.objective=reinforce`; `--freeze`; init = `~/dllm-cache/phase3/3.1b/ckpt/wd1_beta1_tau1_seed{s}/final.pt` for seed s; 1500 steps; head lr 1e-3.
- Code: `src/phase3/conf_token.py` (`--freeze`).
