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
