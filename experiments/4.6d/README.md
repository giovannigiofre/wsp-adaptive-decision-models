# 4.6d: what does self-training learn? Supervised arms on the same puzzles

**Status: setup and predictions written before the runs.**

**Goal.** Self-training (retrain on the model's own valid rollouts with the diffusion loss) was the best row of 4.4: +0.067 [+0.038, +0.095] on the hard test band at 4 sampler steps. Because every valid rollout equals the stored solution (4.4b), self-training is supervised training on the puzzles the model already solves. Question: is its gain a matter of *which* puzzles are trained on, or just of more diffusion-loss steps on hard train puzzles? Direction 4 of [`docs/RL_DLLM_DIRECTIONS.md`](../../docs/RL_DLLM_DIRECTIONS.md).

**Arms** (same loop, optimiser, batch of 32 puzzles × 8 rollouts and 500 steps as 4.4; the loss is the diffusion loss on the *stored solution*, so these arms use labels and are reference rows, not RL):

| arm | trained on |
|---|---|
| `sup-all` | the solution of every puzzle in the batch |
| `sup-solved` | only puzzles with at least one valid rollout in the batch |
| `sup-unsolved` | only puzzles with no valid rollout in the batch (queued last; run if the night's budget allows) |

Three seeds each, evaluated on the 667 hard test puzzles at 4 and 16 sampler steps, paired with the start model and with the self-training checkpoints of 4.4.

## Predictions, written before the runs

1. **`sup-solved` equals self-training** within the interval (difference from self-training inside ±0.03): it is the same data up to how many copies of a solved puzzle enter the loss.
2. **`sup-all` is at least as good as self-training** (memo). Its extra data are the puzzles the model fails, which are the informative ones; I expect +0.07 to +0.12 over the start.
3. **`sup-unsolved` alone still gains** over the start, less than `sup-all`.
4. **None of these moves the 16-step solve rate by more than ±0.02**: 500 steps at lr 1e-5 is a small amount of training next to the 43,500 steps of 4.1.

If prediction 2 holds, self-training is a label-free approximation of a few hundred more supervised steps, and `sup-all` is the reference row an RL objective has to be compared with on this task.

## Command

```
SEED=0 bash scripts/run_phase4.sh rl46-sup-all ~/dllm-cache/venv/bin/python      # seeds 0, 1, 2
SEED=0 bash scripts/run_phase4.sh rl46-sup-solved ~/dllm-cache/venv/bin/python
```

Config `configs/phase4_rl.yaml`. Checkpoints and logs under `~/dllm-cache/phase4/4.6d/`.
