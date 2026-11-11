# 4.6a: exact-gradient audit of the RL objectives, no training

**Status: done (2026-11-11). Setup and predictions were committed before the run (`efa8198`).**

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

## Results (2026-11-11, 01:37 to 02:03, 25 min)

Hardware Apple M5 24 GB, MPS, torch 2.15.0, float32, seed 0 (MPS is not bit-deterministic). The detector gate of 4.6b ran beside it for the first 4 minutes. Valid share at the start, J = 0.500. Figure: `results/phase4/4.6a/audit.png`.

**Reference.** Cosine between the two halves of g\* (32 rollouts per puzzle each): 0.805; norm of g\* 0.75. A step along g\* with predicted gain +0.020 gave +0.018 measured; with predicted +0.080 it gave +0.036 (the first-order prediction holds for the small step only).

**Directions** (inner step 1, 40 batches; "halves" = cosine between the mean directions of the even and odd batches, a measure of how well the mean direction itself is determined):

| objective | cosine with g\* | halves | measured change of J, small step | large step |
|---|---|---|---|---|
| g\* itself | 1 | 0.81 | +0.018 | +0.036 |
| exact policy gradient, no baseline | +0.82 | 0.50 | | |
| exact policy gradient, group baseline | +0.21 | 0.14 | +0.016 | +0.031 |
| supervised on the stored solution | +0.00 | 0.75 | +0.014 | +0.040 |
| self-training | −0.01 | 0.70 | +0.011 | +0.037 |
| filtered imitation | +0.06 | 0.63 | +0.009 | +0.033 |
| positive-only (wd1, β = 0) | +0.08 | 0.72 | +0.009 | +0.033 |
| wd1, batch weights, β = 0 | +0.06 | 0.86 | | |
| wd1, batch weights, floored | −0.04 | 0.72 | +0.014 | +0.011 |
| plain wd1 | −0.04 | 0.42 | +0.001 | +0.022 |
| floored wd1 | −0.01 | 0.39 | | |
| wd1 push-down term alone | −0.09 | 0.72 | −0.002 | −0.012 |
| REINFORCE with the one-step likelihood | −0.04 | 0.42 | | |
| GRPO | −0.06 | 0.37 | | |
| pull-up on valid, one-step likelihood without prompt mask | +0.00 | 0.87 | | |
| pull-up on valid, path likelihood without the order factor | −0.05 | 0.93 | | |

All steps have the same length (0.027 and 0.107 in parameter norm). J after a step is measured with 128 rollouts per puzzle on the same 256 puzzles; a difference of two such measurements has a standard error of roughly 0.004, so the small-step column separates "rises" from "does not" only loosely and the large-step column is the one to read.

**Parts of the wd1 direction** (cosine with g\*, norm): right cells +0.07, 2.24; wrong cells of the first-error step +0.09, 0.49; wrong cells committed later −0.15, 2.08. Pull-up part norm 10.2, push-down part 10.0, their sum (plain wd1) 2.7.

**Inner step 4** (after three real updates on the batch): positive-only +0.065 (from +0.079), filtered +0.044 (from +0.060), plain wd1 −0.066 (from −0.037), floored −0.022 (from −0.009), self-training −0.011 (from −0.014).

**Likelihood gap** (estimate minus exact log-likelihood; 5,082 valid and 5,158 invalid rollouts; exact mean −2.9 valid, −7.4 invalid):

| estimate | valid: mean gap, correlation with exact | invalid: mean gap, correlation |
|---|---|---|
| one-step, masked prompt (used in training) | −49.1, 0.03 | −53.2, 0.13 |
| one-step, no prompt mask | −23.4, 0.02 | −31.9, 0.26 |
| path likelihood without the order factor | +2.4, 0.28 | +3.8, 0.66 |

**Per-cell change from the start to two 4.4 checkpoints** (one-step log-probability, answer fully masked, on 2,048 rollouts of the start model; full table in `cells.csv`):

| cell type | cells | plain wd1: solution digit | plain wd1: written digit | self-training: solution digit | self-training: written digit |
|---|---|---|---|---|---|
| valid rollout | 54,069 | −0.06 | (same) | +0.06 | (same) |
| invalid, right, before the first error | 11,717 | +0.01 | (same) | +0.04 | (same) |
| invalid, right, at or after the first error | 28,395 | −0.05 | (same) | +0.08 | (same) |
| invalid, wrong, first-error step | 1,885 | −0.55 | −0.01 | −0.25 | +0.02 |
| invalid, wrong, later | 13,462 | −0.21 | −0.47 | −0.02 | −0.10 |

## Predictions against results

1. Reference usable (halves above 0.8): **held, barely** (0.805). Not repeated with 256 rollouts, by the rule written above.
2. Pull-up objectives point along g\* (above 0.5; my weaker guess 0.1 to 0.4): **not held, on either reading.** Self-training −0.01, filtered +0.06, positive-only +0.08. Their mean directions are well determined (halves 0.63 to 0.72), so this is not batch noise.
3. Plain wd1 below 0.2, push-down near zero or negative: **held** (−0.04, −0.09), but this no longer separates wd1 from the others: every objective that uses a one-step likelihood is within ±0.09.
4. The exact policy gradient has the highest cosine: **held** (0.82, at the noise ceiling of g\*). With the group baseline it is 0.21 and poorly determined (halves 0.14).
5. One-step estimate further off on invalid rollouts: **held, by a small margin** (53 against 49 nats). Not predicted: the estimate used in training has correlation 0.03 to 0.13 with the exact log-likelihood inside each class.
6. Alignment falls from inner step 1 to 4: **held for four of five in sign**, all changes 0.03 or smaller; undecided at this noise level.
7. Most of the wd1 gradient sits after the first error, right cells not cancelled: **held** (norms 2.08 against 0.49; right cells 2.24).
8. Finite steps: **held.** Up along g\* and along self-training; plain wd1 below positive-only (+0.022 against +0.033); push-down alone lowers J (−0.012).
9. Per cell: **first part not held.** The solution's digit on first-error cells becomes *less* likely under the self-training checkpoint too (−0.25), less so than under plain wd1 (−0.55). Second part held: plain wd1 lowers the written wrong digit of later cells more (−0.47 against −0.10).

## What this shows / does not show

- Shows: on this model the objectives of 4.4 do not follow the score-function gradient of the solve rate. The pull-up objectives are nearly orthogonal to it and still raise the valid share by as much as a step of the same length along it (+0.033 to +0.040 against +0.036); the exact gradient predicts none of that gain (first-order prediction about zero). The wd1 push-down term alone lowers the valid share, and the part of the wd1 direction that is negatively aligned is the one on wrong cells committed after the first error. The one-step likelihood used in training is about 50 nats from the exact one and carries almost no information about it.
- Does not show: *why* directions orthogonal to g\* raise J. Two readings fit and were not separated: (a) J is piecewise smooth and the gain comes through changes of the unmasking order, which g\* does not contain by construction (the limit stated before the run); (b) g\* is one ascent direction among many at this step length and the supervised-like directions are others. It does not show that following the exact gradient in training would beat self-training (no training was run), nor anything beyond one model, one seed, 256 train puzzles and the start point. The cosine with g\* cannot be read above 0.8. The per-cell table uses the one-step likelihood from a fully masked answer, which is not the state in which those cells were committed.
