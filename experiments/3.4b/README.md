# 3.4b Learning the unmasking order with RL on a frozen denoiser

**Goal:** 3.4a showed that some order solves 0.98 of the test puzzles with the weak denoiser, against 0.56 for the `confidence` rule. Can a small policy trained with RL on the validity reward, without ever seeing a solution, recover part of that gap?

Setup: the denoiser (SFT step 120) is frozen. One cell is committed per step with the denoiser's greedy value. A policy head scores every masked cell and a cell is sampled from the softmax over the scores (greedy at evaluation). Training is REINFORCE with the group-mean baseline, 32 puzzles × 8 episodes per step. The log-probability of an order is exact, so no likelihood proxy is needed.

| Variant | Inputs per cell | Start |
|---|---|---|
| `probs`, zero init | confidence, margin, 1 − entropy | uniform random order (0.14) |
| `probs`, confidence init | the same | exactly the confidence rule (0.56) |
| `hidden`, zero init | the same + the denoiser's final hidden state | uniform random order |
| `hidden`, confidence init | the same | exactly the confidence rule |
| oracle teacher, `probs` and `hidden` | as above | supervised with the solution: a ceiling for what each input set can express, not a method |

**Prediction** (written before any training run, including pilots):
- `probs`, zero init: learns quickly, ends near the confidence rule (0.50–0.60 test), not clearly above it. The three inputs *are* the heuristics, and they probably cannot tell a confident wrong cell from a confident right one.
- `probs`, confidence init: 0.56–0.62, a small gain at most.
- `hidden`: clearly above the confidence rule, 0.65–0.80, far below the oracle's 0.98.
- Oracle teacher: about 0.60 with `probs` and about 0.85 with `hidden`.

**Command**
```
bash scripts/run_phase3.sh 3.4b ~/dllm-cache/venv/bin/python
PYTHONPATH=src python -m phase3.analysis 3.4b
```

- Config: `configs/phase3.yaml` (`order_policy:`). Learning rate chosen on val in a one-seed pilot (see Outcome).
- Code: `src/phase3/order_policy.py`. Tests: `tests/test_order_policy.py`.
