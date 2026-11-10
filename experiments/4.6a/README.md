# 4.6a: exact-gradient audit of the RL objectives, no training

**Status: setup and predictions written before the run.**

**Goal.** In 4.4 the objectives that only pull valid rollouts up gained 0.05 to 0.07 at 4 sampler steps and plain wd1 lost 0.03. All of them use a one-step estimate of the rollout's likelihood. On this task the reward is binary and every valid rollout equals the stored solution (4.4b), so the quantity RL should raise is J = share of valid rollouts at temperature 1, and the probability of a rollout under the confidence-order sampler can be written exactly (`phase3.rl.exact_trajectory_logp`, checked against enumeration on a tiny board in `tests/test_phase4_exact_logp.py`). Question: which objective's update points along the gradient of J, and where does the wd1 update go? Direction 1 of [`docs/RL_DLLM_DIRECTIONS.md`](../../docs/RL_DLLM_DIRECTIONS.md).

**Setup.** Start model of 4.3 (1.86M parameters), first 256 hard train puzzles, rollouts with 4 sampler steps at temperature 1, confidence order.

- **Reference g\*:** mean of r · ∇ log P(rollout) with the exact likelihood, 64 rollouts per puzzle, in two halves; the cosine between the halves is reported as the noise of the reference.
- **Directions:** for each objective, minus the gradient of its loss, averaged over 40 batches of 32 puzzles × 8 rollouts (same batch layout as 4.4). Reported: cosine with g\*, cosine between the two halves of the batches, norm, batch noise. For wd1, positive-only, floored wd1, filtered and self-training also at inner step 4, after three real AdamW updates on the batch (lr 1e-5, clip 1.0, as in 4.4).
- **Parts of wd1:** pull-up and push-down separately; and the whole wd1 gradient split by cell type (right cells, wrong cells of the first-error step, wrong cells committed later).
- **Likelihood gap:** one-step estimate with the masked prompt (used in training), one-step without it, per-step path likelihood without the order factor, each minus the exact log-likelihood, for valid and invalid rollouts.
- **Finite steps:** the parameters are moved along g\* and along eight objective directions by a step whose predicted gain along g\* is 0.02 and 0.08, and the valid share is re-measured with 128 rollouts per puzzle. This check does not depend on the likelihood formula.
- **Per-cell change** from the start model to two 4.4 checkpoints (plain wd1 seed 0, self-training seed 0): change of the one-step log-probability of the solution's digit and of the written digit, by commit step and cell type.

**Known limits, before the run.** J is piecewise smooth: when the set of cells committed at a step changes, J jumps, and g\* does not see that (the finite steps do). g\* is a Monte Carlo estimate; a cosine with it cannot be read above the cosine between its halves. One model, one seed, train puzzles only.

## Predictions, written before the run

1. **The reference is usable:** cosine between the halves of g\* above 0.8 at 64 rollouts per puzzle. Low confidence: if it is below 0.8 the run is repeated with 256 rollouts per puzzle and that is reported.
2. **Pull-up objectives point along g\*:** self-training, filtered and positive-only have cosine with g\* above 0.5 (memo). My own confidence is lower than the memo's: I expect them positive but would not be surprised by 0.1 to 0.4, since the one-step likelihood differs from the exact one by tens of nats.
3. **Plain wd1 does not:** cosine with g\* below 0.2, and the push-down part alone near zero or negative.
4. **The exact policy gradient from 256 rollouts is noisy but aligned:** `pg_exact` has the highest cosine with g\* of all directions (it estimates the same quantity).
5. **The one-step estimate is further off on invalid rollouts:** mean absolute gap to the exact log-likelihood larger for invalid than for valid rollouts.
6. **Alignment falls inside a batch:** cosine with g\* at inner step 4 below the cosine at inner step 1 for every objective measured at both.
7. **Most of the wd1 gradient sits on cells after the first error:** the "later wrong cells" part has a larger norm than the "first-error cells" part, and the "right cells" part is not close to zero (section 2.3 of the memo conjectures cancellation there; I expect it to be partial).
8. **Finite steps agree in sign with g\*:** along g\* the valid share rises; along self-training it rises; along plain wd1 it rises less than along positive-only or falls.
9. **Per cell:** from the start to the self-training checkpoint the log-probability of the solution's digit rises on first-error cells; to the plain wd1 checkpoint it rises less or falls there, and the written (wrong) digit of later cells falls more.

## Command

```
PYTHONPATH=src python -m phase4.grad_audit --size medium \
  --compare wd1=$HOME/dllm-cache/phase4/4.4/ckpt/wd1_beta1_tau1_seed0/final.pt \
            self_training=$HOME/dllm-cache/phase4/4.4/ckpt/self_diffusion_seed0/final.pt
```

Config `configs/phase4.yaml`. Results: `results/phase4/4.6a/reference.csv`, `directions.csv`, `eps.csv`, `fd.csv`, `cells.csv`.
