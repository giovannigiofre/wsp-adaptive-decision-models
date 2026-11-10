# 4.1 Model size check and masked diffusion training on 9×9

**Goal:** pick the model size by the rule fixed in the spec, then train that model to a plateau.

## Size check (done)

**Command**
```
bash scripts/run_phase4.sh size python     # size_small then size_medium, 20 minutes each, one after the other
```
- Config: `configs/phase4.yaml` (`sizes.small` = 4 layers, d 128, 809,737 parameters; `sizes.medium` = 6 layers, d 160, 1,876,489 parameters). Batch 256, AdamW lr 3e-4 with 500 warm-up steps then constant (phase 2 used a cosine; here the last step is not known in advance), dropout 0.1, seed 0.
- Evaluation: 1,002 val puzzles, `confidence` unmasking, 16 sampler steps, greedy.
- Hardware: Apple M5 (Mac17,3), 24 GB, macOS 27.0.1, Python 3.12.13, torch 2.15.0, MPS, fp32, nothing else on the GPU.
- Wall-clock: 20 min each (the budget), 40 min in total.
- Results: `results/phase4/sft/size_{small,medium}.{csv,json}`. Checkpoints local, `~/dllm-cache/phase4/sft/`.

| after 20 minutes | steps done | val loss | val solve: all | easy | medium | hard | hidden cells correct |
|---|---|---|---|---|---|---|---|
| 0.8M | 7,489 | 0.0563 | 0.829 | 1.000 | 0.958 | 0.530 | 0.922 |
| 1.86M | 2,971 | 0.0510 | 0.807 | 1.000 | 0.937 | 0.485 | 0.911 |

Paired bootstrap on the val puzzles, 1.86M minus 0.8M solve rate: −0.022 [−0.044, 0.000] overall, −0.045 [−0.108, 0.015] on the hard band.

**Reading against the rule.** The rule says: use 1.86M unless it is not better than 0.8M. The answer is mixed. At equal time the larger model has the lower val loss (by 9%) and a solve rate that is 2 points lower, with an interval that touches zero. At equal steps it is clearly ahead (its step 2,000 matches the small model's step 4,000 on every column), but each step costs 2.5 times as much (403 ms against 160 ms, evaluations included). The 1.86M model was chosen, because val loss is the quantity the plateau rule watches and it is ahead there; this is a judgement on an ambiguous outcome, not a clean pass of the rule. Neither model is at zero on the easy band, so the 6.3M model was not tried.

**Shows:** both sizes learn 9×9 Sudoku from 100k puzzles within minutes; after 20 minutes they solve all easy, about 95% of medium and about half of the hard val puzzles.
**Does not show:** which size is better at convergence (one seed, 20 minutes, neither run had flattened), nor anything about RL.

## Training to the plateau (running)
`SIZE=medium bash scripts/run_phase4.sh sft python`, in launches of 110 minutes that resume from `state.pt`. The run continues the size-check run of the same model (same config and seed), so its first 2,971 steps are that run.
