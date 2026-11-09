# 3.4a Is there headroom in the unmasking order? Oracle check on the frozen weak denoiser

**Goal:** before training any unmasking policy, find out whether a better order than the `confidence` rule exists at all for the weak denoiser (SFT step 120). One cell is committed per step and its value is always the denoiser's greedy prediction, so only the order can change.

The oracle commits, at each step, a masked cell whose greedy prediction equals the true solution (the most confident such cell). If there is none, the puzzle is lost. It uses the solution, so it is an analysis tool and not a method. Some order reaches its solve rate, so it is a lower bound on what the best order can do.

**Prediction** (written before the run): the oracle solves at least 0.85 of the test puzzles, against 0.56 for `confidence`. If instead it is below 0.65, order is not the bottleneck and track D stops here.

**Command**
```
PYTHONPATH=src python -m phase3.order_policy oracle
```

- Config: `configs/phase3.yaml` (denoiser pinned by sha256). No training, no random draws except for the `random` baseline.
- Code: `src/phase3/order_policy.py`. Tests: `tests/test_order_policy.py`.
- Hardware: Apple M5 24 GB, macOS 27.0.1, Python 3.12.13, torch 2.15.0, MPS, fp32 (the 3.1a runs were using the GPU at the same time). Seed: none needed. Wall-clock: about 3 s.
- Results: `results/phase3/3.4a/oracle.csv`, `oracle_by_hidden_cells.csv`, `oracle_details.json`.

## Outcome

Test solve rate, 1,194 puzzles, one cell per step, 95% bootstrap interval:

| order | solve rate |
|---|---|
| random | 0.144 [0.125, 0.164] |
| entropy | 0.462 [0.433, 0.489] |
| margin | 0.539 [0.510, 0.567] |
| confidence | 0.560 [0.530, 0.588] |
| **oracle** | **0.984 [0.977, 0.991]** |

- **Prediction: right, and the gap is larger than predicted** (0.98 against the predicted 0.85 or more).
- The gap grows with difficulty: at 6 hidden cells `confidence` solves 0.87 and the oracle 0.99; at 12 hidden cells 0.20 against 0.98.
- At the first step the weak denoiser's greedy prediction is right for 4.6 masked cells on average, and the oracle is never stuck at step 0. So a correct next move almost always exists; the `confidence` rule often fails to pick it.

**Shows:** for this weak denoiser, order is a large bottleneck. A policy that chooses cells well could in principle lift the solve rate from 0.56 to about 0.98 without changing the denoiser.
**Does not show:** that such a policy can be learned without the solution. The oracle sees the answer; a learned policy sees only the denoiser's outputs. That is 3.4b.
