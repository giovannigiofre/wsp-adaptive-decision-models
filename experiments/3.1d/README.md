# 3.1d Does the starting model change the story?

**Goal:** phase 2 started RL from one snapshot (SFT step 120, val 0.42) of a model that reaches 0.85 thirty SFT steps later. One suspected reason for positive-only's win is that this snapshot is "one sharpening step away" from solving. Test other starting points:

| Start | What it is |
|---|---|
| SFT step 100 | earlier snapshot, val 0.11 |
| SFT step 120 | the phase 2 start, val 0.42 (runs from 3.1b) |
| SFT step 130 | later snapshot, val 0.57 |
| converged on few solutions | SFT run to convergence on the puzzles of only N of the 200 train solutions. N is chosen in a pilot over {2, 5, 10, 20, 40} as the one whose final val solve rate is in [0.3, 0.6] and closest to 0.45, or the closest to 0.45 if none is in the window. RL then uses puzzles from all 200 solutions with the validity checker only. |

The last row is the realistic case for RL: few labels, a cheap checker, and a model that is weak because it has seen little, not because training was stopped early.

**Prediction** (written before the runs, including the SFT pilot):
- Step 100: positive-only collapses to nearly 0 and ends below 0.5 in at least 2 of 3 seeds. wd1 improves slowly and ends below 0.6.
- Step 130: positive-only dips less than from step 120 (lowest val at least 0.25) and ends at 0.97 or more. wd1 ends around 0.9.
- Converged on few solutions: the ranking flips. Positive-only gains less than 0.2 over its start and shows no collapse-then-recovery pattern, because the model is already sharp and its errors are systematic. wd1 gains more than positive-only.

**Command**
```
bash scripts/run_phase3.sh 3.1d-sft ~/dllm-cache/venv/bin/python     # SFT on N solutions, N in {2, 5, 10, 20, 40}
PYTHONPATH=src python -m phase3.analysis 3.1d-sft                    # picks N, writes chosen_subset.txt
bash scripts/run_phase3.sh 3.1d ~/dllm-cache/venv/bin/python
PYTHONPATH=src python -m phase3.analysis 3.1d
```

- Config: `configs/phase3.yaml`; wd1 (β = 1) and positive-only (β = 0) at τ = 1, seeds 0–2, standard horizon.
- Code: `src/phase3/sft_subset.py` (same model, loss, optimiser and schedule as phase 2 SFT; 5,000 steps), `src/phase3/rl.py --init`.
