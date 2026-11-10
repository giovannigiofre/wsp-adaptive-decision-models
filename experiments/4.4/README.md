# 4.4 The phase 3 objectives on 9×9, at their phase 3 settings

**Goal:** run the seven objectives of phase 3 from the 9×9 plateau model and check four phase 3 claims on a harder task.

**Setup (fixed before any run)**
- Start: `~/dllm-cache/phase4/sft/sft/final.pt` (sha256 pinned in `configs/phase4_rl.yaml`), the 4.1 model. No half-trained snapshot.
- Task: hard band only (k 51–56; 33,334 train, 334 val, 667 test puzzles), rollouts and evaluation with **4 sampler steps** (chosen in 4.3 by the rule in the spec). The solve rate at 16 steps is logged next to it (`val_solve_ref`).
- Everything else as in phase 3: 32 puzzles × 8 rollouts per step, temperature 1, `confidence` order, 4 inner steps, lr 1e-5, binary reward, one-step likelihood estimate, 500 steps, seeds 0, 1, 2.
- Arms: wd1 (τ 1); positive-only (β 0); wd1 with floor −75.6 (10th percentile of the one-step log p̂ of invalid rollouts at the start, `results/phase4/4.3/floor.json`); wd1 + KL to the start (0.01); filtered imitation; self-training with the SFT loss; d1-style GRPO.
- Trainer: `src/phase3/rl.py`, which now reads `task: sudoku9` from the config. Regression guard rerun after this change: phase 2 strategies table identical, phase 3 RL log equal up to 3.8e-6 (MPS noise, as before).

**Measured at the start** (4.3, 512 train puzzles × 8 rollouts): 56% of sampled rollouts valid; 24% of groups with unequal rewards; median one-step log p̂ −52 for valid and −62 for invalid rollouts. Greedy solve at 4 steps: val 0.521, test 0.535; at 16 steps: val 0.805.

## Predictions, written before the runs
Each is the phase 3 claim carried over, with what would count as "held".
1. **wd1 improves steadily but slowly, and its log p̂ of valid rollouts falls.** Held if val solve (4 steps) at step 500 is above the start in all three seeds, by less than the best arm, and `logp_valid` is lower at the end than at the start.
2. **Bounding the push down makes wd1 faster.** Held if both the floored and the KL variant end above plain wd1, with a paired interval over test puzzles (seeds pooled) that excludes zero.
3. **Positive-only at τ = 1 puts about half of its positive weight on invalid rollouts and collapses early.** Held if `w_pos_on_invalid` is between 0.3 and 0.6 over the first 50 steps and val solve falls by more than 0.10 below the start within the first 150 steps in at least two seeds. My own expectation, noted here so it can be checked: the weight share will be about 0.4, but the collapse is less likely than on the toy because this start is a plateau model, not a half-trained one.
4. **Training on one's own valid rollouts is the best objective when valid rollouts are common** (they are: 56%). Held if self-training or filtered imitation has the highest mean test solve at step 500.

Extra expectation, not a phase 3 claim: because 3 groups in 4 carry no signal for the group-relative objectives, wd1, its variants and GRPO have about a quarter of the effective batch of the two imitation arms.

**Commands**
```
SEED=0 bash scripts/run_phase4.sh rl44 python     # also SEED=1 and SEED=2; 7 runs each
bash scripts/run_phase4.sh rl44-h python          # horizon check: one wd1 run of 1,000 steps
```
Results: to be filled in.
