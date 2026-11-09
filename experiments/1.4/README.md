# 1.4 Phase 1 summary and notebook

- **Goal:** write up Phase 1 (setup, speed, best figures, numbers with intervals, limitations, next questions) and provide a notebook that narrates 1.0-1.4 from the committed results without loading the model.
- **Command:**
  - `~/dllm-cache/venv/bin/python -m src.phase1.report_grid`
  - `~/dllm-cache/venv/bin/python -m src.phase1.calibration`
  - `~/dllm-cache/venv/bin/python scripts/build_phase1_notebook.py`
  - `~/dllm-cache/venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/phase1_llada.ipynb`
- **Config:** `configs/phase1.yaml`. The narrative text lives in `notebooks/phase1_narrative.json`.
- **Hardware:** same Mac, CPU only.
- **Software:** torch 2.15.0 (not used), pandas, scipy 1.18.1, scikit-learn, nbconvert. No model weights involved.
- **dtype:** n/a.
- **Seed:** bootstrap seed 0 (`calibration.bootstrap_ci`).
- **Wall-clock:** < 1 min.
- **Results:** `results/phase1/SUMMARY.md`; `notebooks/phase1_llada.ipynb` (executed, ~0.4 MB).

## Outcome
- **Unmasking order:** with confidence-based unmasking, LLaDA fills each block almost left to right (rho 0.64).
- **Strategies:** confidence-ordered unmasking beats random on Countdown, +0.28 [0.11, 0.44] paired. No ranking among confidence, margin and entropy is possible at n = 5-12.
- **Step budget:** fewer steps cost little accuracy here.
- **Calibration:** commit confidence is overconfident, and on Sudoku it is anti-predictive (AUROC 0.10) because of copied zeros.

**Shows:** a reproducible, honest pilot of LLaDA's decoding behaviour on a laptop, with all failures and deviations recorded.
**Does not show:** results at the spec's sample sizes or settings comparable to the LLaDA and d1 papers.
