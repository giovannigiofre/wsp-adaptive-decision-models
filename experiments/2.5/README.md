# 2.5 Write-up and notebook

**Goal:** collect phase 2 into a summary and a runnable narrative notebook.

**Commands**
```
~/dllm-cache/venv/bin/python scripts/build_phase2_notebook.py
~/dllm-cache/venv/bin/jupyter nbconvert --to notebook --execute --inplace notebooks/phase2_sudoku_wd1.ipynb
```

- **Outputs**
  - `results/phase2/SUMMARY.md`: model size, Mac training time, SFT vs RL with seeds, the wd1 vs positive-only ablation, calibration before/after, limitations, and the scope of the conclusions.
  - `notebooks/phase2_sudoku_wd1.ipynb`: one section per step. The notebook mostly loads committed CSV/PNG files. Its one live demo is the weak model decoding a puzzle on CPU, and it is skipped if the local checkpoint is absent.
- **Environment:** Apple M5, Python 3.12.13, torch 2.15.0. The notebook runs on CPU in a few seconds.

**Shows:** every number in the summary can be traced to a committed CSV/JSON file and to the command that produced it.
**Does not show:** anything not already covered by 2.0–2.4.
