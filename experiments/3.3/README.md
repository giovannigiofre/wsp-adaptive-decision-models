# 3.3 Rewarding honest confidence: a confidence token with a Brier-shaped reward

**Goal:** phase 2 found that wd1 makes the toy model over-confident. RLCR (arXiv:2507.16806) has a model state a confidence q and rewards it with r′ = r − (q − r)², where r is correctness; the expected reward is highest when q equals the true probability of being right. Try a toy version and compare it with simpler ways of getting a calibrated confidence.

Setup: a 17th position with a learned query reads the grid through the denoiser's own layers and outputs one of 4 confidence values (0.125, 0.375, 0.625, 0.875). Grid positions cannot attend to it, so the grid predictions are exactly those of the underlying model.

| | Meaning |
|---|---|
| place = last | q is read from the completed grid: "is the grid I just wrote valid?" |
| place = first | q is read from the puzzle alone: "will I solve this puzzle?" |
| mode = rl | q is sampled, its log-probability joins log p̂, and wd1 uses the reward r′ |
| mode = probe | the grid is trained by wd1 on the plain reward; the confidence head minimises the expected Brier loss directly. This is the exact-gradient reference: no sampling of q |
| implicit, raw | no token: the mean commit confidence of the puzzle, as in phase 2.4 |
| implicit, recalibrated | the same score passed through a 1- or 2-parameter logistic map fitted on val |

Deviations from the spec, decided before any run:
- The grid objective is wd1 at τ = 1, not "the best objective from 3.1/3.2". The best objectives are expected to solve nearly every puzzle, which leaves almost no failures to be calibrated on, and wd1 at τ = 1 is the setting where phase 2 saw over-confidence.
- The no-RL baseline recalibrates the puzzle-level score instead of scaling the logits by a temperature, because scaling the logits would change the unmasking order and so the answers.

**Prediction** (written before the runs):
- rl, last: the token learns to check the grid. AUROC of q is at least 0.90 on test and its Brier score is below that of the raw implicit confidence. Test solve rate stays within 0.03 of wd1 without the token.
- rl, first: AUROC between 0.60 and 0.75, and a Brier score within 0.01 of always stating the base rate. From the puzzle alone there is little to go on except the number of hidden cells.
- probe is at least as good as rl in Brier score for both placements, with a small gap for `last`.
- Recalibration improves the Brier score of the implicit confidence and leaves its AUROC unchanged. The `last` token beats it on both.

**Command**
```
bash scripts/run_phase3.sh 3.3 ~/dllm-cache/venv/bin/python
PYTHONPATH=src python -m phase3.conf_token baseline --ckpt weak=~/dllm-cache/phase2/sft/sft_weak.pt wd1_seed0=... wd1_seed1=... wd1_seed2=...
PYTHONPATH=src python -m phase3.analysis 3.3
```

- Config: `configs/phase3.yaml` (`rl:` for the grid objective, `conf_token:`), seeds 0–2, standard horizon. The new query and head use lr 1e-3; the denoiser keeps 1e-5.
- Code: `src/phase3/conf_token.py`. Tests: `tests/test_conf_token.py`.
