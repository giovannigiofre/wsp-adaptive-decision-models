# 4.5a Factorial on floored wd1: sampled unmasking order × weights over the batch

**Status:** setup and predictions, written before the runs. Results are added below when the runs finish.

**Goal:** 4.4 found that the group-relative objectives get no signal from about 3 groups in 4 (all rollouts valid or all invalid) and at best undo the damage of plain wd1. Two ways to get signal from more groups, from `docs/RL_DLLM_SURVEY.md` (proposals 2a and 3, both existing ideas applied here), tested together:

- **Sampled order.** In rollouts the cells to commit at each sampler step are drawn without replacement with weights `confidence^(1/τ_o)`, τ_o = 0.05, instead of the top k. 4.4b measured that this turns 0.26 → 0.54 of groups into groups with unequal rewards at the same valid share. The loss is unchanged (one-step likelihood; the order is not in the gradient). Evaluation stays greedy with the fixed order.
- **Weights over the batch.** The wd1 softmax is taken over all 256 rollouts of the batch instead of within each group of 8 (`phase3.objectives.wd1_batch`). An all-valid group is then pulled up and an all-invalid group pushed down (down to the floor), so no group is skipped.

A note on the second factor. The survey called it "batch-mean baseline". With wd1's softmax weights a change of baseline alone does nothing (a softmax ignores a constant shift), so what is changed is the set the softmax runs over. This was fixed before the runs.

| arm | order | weights | runs |
|---|---|---|---|
| floored wd1 (4.4) | fixed | group | reused from 4.4 |
| + sampled order | τ_o = 0.05 | group | 3 seeds |
| + batch weights | fixed | batch | 3 seeds |
| + both | τ_o = 0.05 | batch | 3 seeds |

All arms: wd1, β = 1, τ = 1, floor −75.6, start and every other setting as in 4.4 (`configs/phase4_rl.yaml`: hard band, 4 sampler steps, 32 puzzles × 8 rollouts, 4 inner steps, lr 1e-5, 500 steps). With `order_temp: 0` and `weights_over: group` the code path and random stream are those of 4.4.

**Command**
```
for arm in ord batch both; do for s in 0 1 2; do SEED=$s bash scripts/run_phase4.sh rl45a-$arm ~/dllm-cache/venv/bin/python & done; wait; done
PYTHONPATH=src python -m phase4.rl_analysis compact 4.5a && python -m phase4.rl_analysis evaluate 4.5a && python -m phase4.rl_analysis report 4.5a
```
Hardware: Apple M5, 24 GB, macOS 27.0.1, torch 2.15.0, MPS, fp32. Logs and checkpoints in `~/dllm-cache/phase4/4.5a/`.

**Expected time.** A 20-step trial of the "both" arm took 2.6 s per step with one other job running, because every group now goes through the gradient pass (4.4: about 1 in 4). With three seeds in parallel: about 20 minutes for the sampled-order arm and 40 to 50 minutes for each batch arm, so about 1 h 50 min in total, more than the 1 h announced. The three arms are launched one after the other, each under an hour.

## Predictions (written before the runs)

The 20-step trial was seen before writing these (equal-reward groups about 0.5 with sampled order; nothing about solve rates).

1. **Sampled order uses more groups:** the share of skipped groups in the sampled-order arm is below 0.6 over the run (4.4 floored arm: 0.70 to 0.83). Confidence: high, it follows from 4.4b unless training moves it.
2. **Sampled order helps:** the sampled-order arm ends above the floored arm of 4.4 on hard test at 4 steps, paired interval excluding zero (the survey's prediction). Confidence: low. Against it: the rollouts use a noisier order than the greedy sampler that is evaluated, and in 4.4 the arms with negatives only undid damage.
3. **Batch weights help:** the batch arm ends above the floored arm of 4.4, paired interval excluding zero (the survey's prediction). Sharper: it also ends above filtered imitation of 4.4 (+0.051 over the start); if it only matches it, the push-down of all-invalid groups adds nothing. Confidence: medium for the first part (all-valid groups are pulled up, which is close to self-training, the best arm of 4.4), low for the second.
4. **The two overlap:** the gain of "both" over the floored arm is smaller than the sum of the two single gains.
5. **No collapse:** no arm ends below the start (0.535) by more than its interval, and `logp_valid` does not fall as it did for plain wd1 (−57 → −68).

Deciding measurement: hard test solve at 4 sampler steps at step 500, paired over the 667 test puzzles with seeds pooled, as in 4.4; 16 steps reported next to it.
