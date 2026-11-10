# RL on a small masked diffusion LM: where to go one step further

Written 2026-11-11, after 4.4, 4.4b, 4.5a and 4.5b. It continues `docs/RL_DLLM_SURVEY.md` and does not repeat it. No experiment was run for this memo and no code was changed.

Every statement is marked by its source:
**[P]** stated in a paper that was opened (how far it was read is in section 7),
**[M]** measured in this repo (experiment id given),
**[C]** conjecture or derivation of mine, not checked.

## Summary

| rank | direction (section 3 number) | question | new code | compute (3 seeds side by side) | what decides it |
|---|---|---|---|---|---|
| 1 | Exact-gradient audit (1) | Which objective's update points along the true policy gradient on this model, and how far off is the one-step estimate? | exact trajectory likelihood, audit script, 2 tests | audit 10 to 20 min, no training; then 2 to 3 arms, about 1.5 h | cosine of each objective's update with the exact gradient |
| 2 | Learned detector of wrong cells (5) | 4.5b: repair is easy, detection is the bottleneck. Can a detector be learned, how good must it be, and can it be learned from the validity reward alone? | degraded-oracle curve, probe, detector head, fix round in rollouts | gate 20 to 30 min, no denoiser training; head with labels about 1 h; RL arms about 1.5 h | AUROC inside failed grids, and solve rate at equal forward passes against `plain` |
| 3 | Regime map for the push-down term (2) | Under which conditions does wd1's negative term help, as its paper reports, and under which does it hurt, as here? | dense 9×9 reward, 2 config keys | 6 to 8 arms, 4 to 5 h in three launches | paired test difference, wd1 minus positive-only, per regime |
| 4 | Where the push-down should land (3) | Does a negative term help if it is put only on the first wrong cells, in the state where they were chosen? | per-cell negative weights on the trajectory likelihood | 3 arms, about 1.5 h | localised minus cascade-only negatives |
| 5 | What self-training learns (4) | Is the +0.067 of self-training selection of solvable puzzles, more SFT, or path-specific? | puzzle filter for the `supervised` objective | 3 arms, about 1.5 h | supervised on solved, on unsolved, on all |

Two rankings are possible and they differ. By headroom, the detector is first: the gap between the best real detector and the oracle is 0.16 to 0.37 in solve rate **[M, 4.5b]**, against +0.067 for the best RL arm **[M, 4.4]**. By what is open in the literature and by the link to wd1, the audit is first: learned correctors and quality heads are published (Token-Critic, PRISM, RemeDi), a gradient audit was not found. The table follows the second, because the brief asks for a research step and the wd1 connection; the first night (section 5) runs the cheap gates of both.

Recommended programme (section 5): direction 1 as the main line, the no-training gate of direction 5 in the same night, direction 2 as the batch that needs almost no new code, direction 4 as filler. Fallback if the audit shows nothing: directions 2 and 4 alone already give a self-contained account of the disagreement with wd1.

The one-line version of the main idea **[C]**: on this task the reward is binary and the solution is unique, so the true objective is `π(x* | q)` and its gradient can be computed to any precision for a 1.86M-parameter model. Papers argue about which surrogate likelihood is less biased; here the bias can be measured instead of bounded.

## 1. Literature beyond the survey

Only what the survey did not cover or what was re-read for this memo.

**Estimators and where they are biased.**
- SPG (2510.09541) **[P]**: lower bound (ELBO) for positive-advantage samples, upper bound (EUBO) for negative ones, because "minimizing the ELBO for negatively-rewarded samples does not guarantee a reduction in the true log-likelihood". Its ablation without negatives gives Sudoku 68.8 against 82.4 with the ELBO for both signs. Rollout temperature 0.3 on Sudoku, 3-shot. This is the published statement closest to our problem: a surrogate that is fine for pulling up can be wrong for pushing down.
- GDSD (2605.29398) **[P]**: says wd1's explicit penalisation "can trigger training instability", with no mechanism given; reports LLaDA Sudoku 53.9 for wd1 against 89.4 for its own method. So wd1's negative term is already reported as fragile by a later paper, on the 8B model.
- Kunde et al. (2603.12554) **[P]**: denoising as a finite-horizon MDP, per-step policy gradient with intermediate advantages `r(x0) − r(greedy completion of the state)`. The unmasking set is "treat[ed] as fixed when differentiating"; the paper calls this "an empirical design choice". Its only gradient-fidelity experiment compares a subset of steps against all 128 steps, not against a true gradient. Remasking models are named as future work.
- DiFFPO (2510.02212) **[P]**: off-policy surrogate with a two-stage likelihood approximation and an importance correction. In the survey already; relevant here because it is the one method that corrects for the proxy rather than replacing it.
- DUEL (2603.01367) **[P]**: for a deterministic unmasking rule that depends only on the revealed sequence, the exact likelihood of a sample is the product of the token probabilities along the replayed path, one forward pass per step. Stated limits: "Stochastic policies are not point masses, and remasking samplers revisit revealed tokens." It does not use the likelihood for RL and does not list that as future work; it notes that one RL paper uses "per-trajectory token likelihoods of the kind DUEL computes".
- Discrete Tilt Matching (2604.18739) **[P]**: likelihood-free weighted cross-entropy with control variates, gains on LLaDA Sudoku and Countdown. A route that avoids sequence likelihoods altogether.

**Positive and negative samples (autoregressive LLMs, the theory carries over in part).**
- Davis and Recht (2510.13651) **[P]**: with binary rewards, the common algorithms are stochastic gradient ascent on a monotone transform of the probability of a correct answer; rejection sampling corresponds to the logarithm, GRPO to the arcsine of the square root.
- RAFT, Xiong et al. (2504.11343) **[P]**: training only on positive samples is competitive with GRPO; GRPO's edge comes from dropping prompts where every sample is wrong.
- Arnal et al. (2506.20520) **[P]**: asymmetric REINFORCE. With off-policy data the update should weight positives more than negatives; an improvement guarantee holds when the baseline is below the expected reward.
- Zhu et al. (2506.01347) **[P]**: splits the policy gradient into positive and negative sample reinforcement. Negative-only training improves pass@k and keeps diversity; positive-only raises pass@1 and cuts diversity.
- LENS (2510.08696) **[P]**: a confidence-weighted penalty for groups where every sample is wrong. BCPG-NSA (2505.14403) **[P]**: step-level credit inside negative samples, because a wrong answer contains correct steps.

**Credit per token in dLLM RL.**
- DACA-GRPO (2605.16342) **[P]**: per-token weights from denoising progress, and a stratified-masking likelihood against the mean-field bias of one-step estimates.
- DTCA (2610.35201) **[P]**: token credit from the confidence gain a token causes on the still-masked positions.
- DiSPO (2602.06462) **[P]**: resamples masked positions at intermediate states and updates the newly filled tokens.
- Mask-Aware Policy Gradients (2607.15200): Plackett-Luce order in the policy. In the survey; not re-opened for this memo.

**Error propagation and correction** (context for direction 5): Kim et al. (2604.10567) on early decisions and a planner; RemeDi (2509.23653), remasking learned by SFT then RL on an 8B model; PRISM (2510.01384), a per-token quality head without RL; ME-DLM (2605.09603), edit-based refinement; Informed Correctors (2407.21243); Token-Critic (2209.04439); ReMDM (2503.00307). All **[P]**, abstract pages only.

**What this adds up to for us [C].** Three separate strands each explain a part of 4.4: the surrogate is worse for negatives than for positives (SPG), positives should dominate when updates are off-policy (Arnal et al.), and positive-only training is a sound algorithm of its own under binary rewards (Davis and Recht, RAFT). None of them measures the size of these effects against a known gradient, and none is on a model small enough to do so.

## 2. wd1 against our implementation

### 2.1 What the paper claims

Source: arXiv 2507.08838 (v2, 14 Feb 2026; read in full as HTML) and the upstream code at commit `1a2e254`. All **[P]**.

- **Loss.** `(1/G) Σ_i (−w⁺_i + w⁻_i) · log π_θ(o_i | q)`, with `w⁺ = softmax(ψ·A)`, `w⁻ = softmax(−ψ·A)`, `A = r − group mean` (no division by the standard deviation), ψ = 1. Upstream: `is_weights = torch.softmax(advantages, dim=0)`, `n_is_weights = torch.softmax(-advantages, dim=0)`.
- **Derivation.** The `w⁺` term is derived: it is the weighted-likelihood projection onto the optimum of a reverse-KL-regularised objective, `π* ∝ π_old^{λ/(λ+β)} · π_ref^{β/(λ+β)} · exp(A/(λ+β))`. **The `w⁻` term is not derived from that objective.** It is motivated separately as unlearning of low-advantage samples and described as an extension of NegGrad to masked diffusion.
- **Error claim.** The ratio-based d1 objective has error `C·exp(ε + ε′)` in the likelihood-approximation error; wd1 has "only linear approximation error" `C′·ε`, where ε bounds the gap between the proxy and `log π`.
- **Likelihood.** The d1 one-step estimate: answer fully masked, prompt masked with probability 0.15, mask re-drawn at each gradient step.
- **Ablation of the negative term** (wd1-P against wd1, two generation lengths): Sudoku 6.69 / 6.84 against 25.2 / 24.2; Countdown 13.67 / 4.69 against 51.2 / 46.1; GSM8K 65.66 / 78.17 against 80.8 / 82.3; MATH500 29.4 / 22.8 against 34.4 / 39.0. A mix `−λ·w⁺ + (1−λ)·w⁻` gives Sudoku 25.63 at λ = 0.5, 11.77 at 0.4, 14.11 at 0.6. The stated argument: when all rewards of a group are low, more positive weight "would paradoxically increase the log-likelihood of negative samples".
- **Stated limitation.** "wd1 may lose effectiveness when all generations within a sampled group receive identical rewards". No share of such groups is reported; the code logs `zero_std_ratio`.
- **Settings.** LLaDA-8B-Instruct, LoRA rank 128, 4-bit, lr 3e-6, weight decay 0.1, gradient clip 0.2, G = 6, 8 to 12 inner steps, 128 diffusion steps, 4×4 text Sudoku. **The Sudoku reward is the fraction of empty cells filled correctly**, so it is dense and uses the solution. Accuracy before RL is about 6.7%.

### 2.2 What `src/phase3/objectives.py` does

`wd1(r, tau, beta, floor)` returns `pos = softmax(A/τ)`, `neg = β·softmax(−A/τ)` per group, and `weighted_loss` is the mean over kept groups of `Σ_i (neg_i − pos_i)·log p̂(o_i)`. With τ = 1, β = 1 and no floor this is the upstream loss up to a constant factor (sum over the group, mean over groups; upstream sums over a per-device batch that is one group). Groups with equal rewards are skipped; upstream they contribute a zero net weight, which is the same thing. The estimator is the same one-step construction (`rl.one_step_logprobs`, prompt mask 0.15). **I found no difference in the formula.** The differences are in what the formula is applied to.

### 2.3 A property of the loss on this task [C]

With the one-step estimate the answer is fully masked, so every rollout of a group is scored from the same input (up to its prompt mask) and `log p̂(o_i) = Σ_c log p_c(o_ic)`. In a group with unequal rewards `Σ_i (neg_i − pos_i) = 0`. Put together: on any cell where all rollouts of the group carry the same digit, the terms cancel. In 4.4b every valid rollout was the stored solution **[M]**, and a failed grid has about 16 wrong cells out of 51 to 56 empty ones **[M]**. So in a mixed group wd1 gives no net gradient on roughly 70% of the cells, and on the rest it does one thing: for each invalid rollout and each cell where it differs from the solution, push its digit down and pull the solution's digit up by the same amount.

Two consequences, both to be checked:
- About 88% of those cells were committed after the first error **[M, 4.4b]**: they are digits that fit an earlier mistake. They are scored in the fully masked state, where that mistake is not visible. The push-down therefore lands mostly on cells that were not the cause of the failure, in a state the sampler was not in when it chose them.
- The cancellation is exact only if the group shares its prompt mask. `rl.py` draws the mask per rollout, so the cancelled cells leave mask noise in the gradient.

Filtered imitation and self-training do not have this structure: they pull up all cells of the solution.

### 2.4 Differences, by how likely I think they matter, each with a cheap test

| # | difference | wd1 paper | here | why it could flip the sign of the negative term **[C]** | cheap test |
|---|---|---|---|---|---|
| 1 | Reward | dense, fraction of correct cells, uses labels **[P]** | binary validity, no labels **[M]** | A dense reward ranks invalid rollouts by how wrong they are and makes nearly every group informative; ours gives 72% equal groups and flat weights inside the rest | `cell_accuracy` reward for 9×9; wd1 and positive-only with it (direction 2) |
| 2 | Start regime | about 7% accuracy **[P]** | 56% valid rollouts, a plateau model **[M]** | The paper's own argument for the negative term is about groups where every reward is low. Phase 3.6b agrees: from 1 valid rollout in 700, floored wd1 reached 0.99 and the imitation arms 0.00 **[M]** | same objectives at 2 sampler steps, where valid rollouts are rare (direction 2) |
| 3 | Unique solution | text answers, many wrong and partly right forms | all valid rollouts identical **[M]** | section 2.3: the net gradient sits on cascade cells | per-cell split of the wd1 gradient into first-error, cascade and agreeing cells (audit, direction 1) |
| 4 | Off-policy inner steps | 8 to 12, LoRA at lr 3e-6 **[P]** | 4, full fine-tune at lr 1e-5 **[M]** | No importance ratio in either. Arnal et al. **[P]**: off-policy updates should favour positives. A tiny model moves further per step | `inner_steps: 1` arm |
| 5 | Update size | LoRA, weight decay 0.1, clip 0.2, lr 3e-6 **[P]** | all weights, no decay, clip 1.0, lr 1e-5 **[M]** | The gradient of `log p` with respect to a logit stays near 1 as `p → 0`, while the pull-up gradient vanishes as `p → 1`; without a tight clip the push-down keeps going. Floor and KL fixed the damage but gave no gain **[M, 4.4]** | one wd1 arm with lr 3e-6 and clip 0.2 |
| 6 | State mismatch of the proxy | 128 steps, block 32 **[P]** | 4 steps, 14 cells per step **[M]** | Cells of later steps are scored without the cells they were conditioned on. Phase 3.6c: the trajectory likelihood made every arm worse on 4×4, cause unknown **[M]** | ε and gradient cosine against the exact likelihood (direction 1) |
| 7 | Post-error actions | not discussed | no remasking, binary reward **[M]** | After the first wrong commit the reward is 0 whatever follows, so the true advantage of those actions is zero and their terms are noise | variance of the gradient split by before and after the first error (audit) |
| 8 | Group | G = 6, sum **[P]** | G = 8, mean **[M]** | scale only | none needed |

One point could not be settled: a summary of the upstream repository said the weights are normalised over all groups; the trainer code and the appendix say per group. I go with the code. If the summary were right, upstream wd1 would be our `wd1_batch`, which in 4.5a was the better variant (+0.030 [+0.012, +0.050] over floored wd1) **[M]**.

## 3. Directions

Numbered in the order they were written; the ranking is in the summary table. Time basis: a 500-step run with 3 seeds side by side took 30 minutes for group-weight arms and 48 to 49 minutes for arms where every group is in the gradient pass **[M, 4.5a]**; an evaluation on the 667 hard test puzzles takes 1 to 2 minutes. All estimates below are derived from these, none was measured.

### Direction 1. Exact-gradient audit of the objectives

**Question.** On the plateau model, how well does the expected update of each objective (self-training, filtered, positive-only, wd1, floored wd1, GRPO) align with the true gradient of the solve probability, and how much of the misalignment comes from the one-step likelihood, from the negative term and from the inner steps?

**Why it is open, and how close prior work is.** SPG, DACA-GRPO, GDSD and wd1 all argue from bounds or from end-task accuracy that one surrogate is less biased than another. The closest numerical check found is Kunde et al., which compares against the full-step update of the same surrogate. DUEL gives the exact likelihood and stops before RL. We did not find a comparison of dLLM policy-gradient surrogates against a ground-truth gradient (scope in section 6). What makes it possible here **[C]**:
- Reward is binary and the solution unique, so `J(θ) = E_q[π_θ(x* | q)]`.
- With rollouts at temperature 1 the unbiased gradient estimate is `r · ∇log P(trajectory)` and needs the exact trajectory probability. `trajectory_cell_logp` is not that: the sampler ranks cells by the probability of the *sampled* digit, so the commit set depends on digits that were drawn and thrown away. For one step with commit set `C`, values `v` and `m = min_{i∈C} p_i(v_i)`: `P = Π_{i∈C} p_i(v_i) · Π_{j∉C} Σ_{c: p_j(c) < m} p_j(c)`. The second product is the missing factor. It costs nothing extra: same forward pass. (My derivation; ties ignored; the last step has no second factor.)
- With that, 4 forward passes per rollout give an unbiased gradient sample, and averaging many rollouts on a fixed set of puzzles gives the reference gradient to any precision the time allows. The model has 1.86M parameters, so storing and comparing full gradients is trivial.
- A cross-check that needs no sampling: an order rule that scores a cell by its maximum probability (not by the sampled digit) is deterministic given the state, so DUEL applies and `π(x* | q)` is one replay along the solution. Greedy evaluation is identical under both rules. At temperature 1 it commits low-probability draws that the current rule defers, so its valid share will be lower; that has to be measured before using it for training.

**Link to wd1.** It tests the paper's central technical claim, that the weighted-likelihood form has only linear error in ε, by measuring ε and the resulting gradient error on a model where both are available. It also measures whether the underived `w⁻` term adds bias, variance or neither.

**Experiment.**
- Files: `src/phase3/rl.py` (new `exact_trajectory_logp` next to `trajectory_cell_logp`), `src/phase4/grad_audit.py` (new), `tests/` (two tests: the formula against brute-force enumeration on a 4×4 puzzle with 3 to 4 empty cells; the max-probability rule equals the current rule under greedy decoding).
- Audit, no training: 256 hard train puzzles, fixed. Reference gradient `g*` from 64 rollouts per puzzle with the exact likelihood, split in two halves to report the reference's own noise (cosine between halves). For each objective, the mean update direction over 20 batches of 32 × 8 at inner step 1, and again at inner step 4 after three real updates.
- Metrics: cosine with `g*`; norm; variance across batches; ε as the gap between `log p̂` and the exact log-likelihood, separately for valid and invalid rollouts; the wd1 gradient split by cell type (agreeing, first-error step, after first error); cosine of the push-down part alone with `g*`.
- Models: the start checkpoint, and the 4.4 step-500 checkpoints of plain wd1 and self-training.
- Training arms, only if the audit separates the objectives: (a) exact on-policy policy gradient on valid rollouts (`r · ∇log P`, one inner step), (b) the same with a group-mean baseline, so that invalid rollouts enter with their exact likelihood, (c) self-training as reference. Seeds 0 to 2, 500 steps, 4-step and 16-step test solve, paired intervals as in 4.4.

**Wall-clock.** Audit: 256 × 64 rollouts with gradient is about 16,000 four-pass rollouts, the size of 60 RL steps; with the per-objective batches I expect 10 to 20 minutes per checkpoint. Training arms: three arms of 30 to 50 minutes, one launch under 2 hours.

**Predictions [C].** (1) Self-training and filtered imitation have cosine above 0.5 with `g*`; plain wd1 is below 0.2, and its push-down part alone is within noise of zero or negative. (2) ε is larger in absolute value for invalid rollouts than for valid ones. (3) The cosine of every one-step objective drops between inner step 1 and inner step 4. (4) More than half of the variance of the wd1 gradient comes from cells committed after the first error. Each is wrong if the number comes out on the other side.

**Interesting either way.** If the predictions hold, 4.4 has a mechanism with numbers attached, and the exact arms show whether a correct negative term is worth anything. If wd1's update is well aligned with `g*` and still loses, then the true gradient is not the right thing to follow at this step size, which points to direction 2's regime explanation and to Davis and Recht's point that the algorithms differ by a transform of the objective, not by bias.

**Main risk.** The objective is piecewise smooth: the top-k choice makes `P(trajectory)` jump when a probability crosses the threshold, and the score-function gradient does not see the jumps. The brute-force test on 4×4 must compare against a finite difference of `J`, not only against the formula. Second risk: phase 3.6c, where the path likelihood made training worse for an unknown reason; the audit may find that reason or hit it.

### Direction 2. A regime map for the push-down term

**Question.** Is there a region of (success rate at the start, reward density, off-policy steps, update size) where wd1 with its negative term beats positive-only on this model, as in the wd1 ablation?

**Why open.** The wd1 paper reports the ablation at one operating point per task, all with low start accuracy and, for Sudoku, a dense reward **[P]**. SPG reports the same direction on the 8B model **[P]**. We measured the opposite at 56% valid and binary reward **[M, 4.4]**, and the paper's direction at 1 valid in 700 on 4×4 **[M, 3.6b]**. Nobody, including us, has moved one factor at a time. RAFT and Arnal et al. give reasons to expect a crossover **[P]**.

**Link to wd1.** A direct reconciliation: differences 1, 2, 4 and 5 of section 2.4, one arm each.

**Experiment.** Config `configs/phase4_rl.yaml`, trainer `src/phase3/rl.py`, new reward `cell_accuracy` in `src/phase3/rewards.py` (fraction of empty cells equal to the solution; uses labels, to be said in every table). Arms, 3 seeds, 500 steps:

| arm | changes against 4.4 | compare with |
|---|---|---|
| R1 | `wd1_batch`, β = 0 | batch weights of 4.5a: separates pull-up from push-down, the open point of 4.5a |
| R2 | plain wd1, 1 inner step | plain wd1 of 4.4 (note: 4 times fewer updates; read the per-update trend, not only the end point) |
| R3 | plain wd1, lr 3e-6, clip 0.2, weight decay 0.1 | plain wd1 of 4.4 |
| R4, R5 | dense reward: wd1 and positive-only | each other; this is the paper's ablation |
| R6, R7, R8 | 2 sampler steps in rollouts and evaluation: wd1, positive-only, filtered | each other; first measure the start solve rate at 2 steps (2 minutes) and go ahead only if it is between 0.02 and 0.30 |

Metrics: test solve at the trained step count and at 16, paired differences with bootstrap intervals; `logp_valid`, `logp_invalid`, skipped share, `w_pos_on_invalid`.

**Wall-clock.** R1 about 50 min, R2 15 to 30, R3 30, R4 and R5 about 50 each (nearly every group has unequal rewards), R6 to R8 about 25 each. Total 4 to 5 hours in three launches.

**Predictions [C].** R1 equals batch weights with β = 1 within the interval (the push-down of all-invalid groups adds nothing). R2 and R3 do not fall below the start. With the dense reward, wd1 beats positive-only (the paper's direction) because positive-only then puts weight on almost-right invalid grids. At 2 sampler steps, wd1 minus positive-only is larger than at 4 steps.

**Interesting either way.** A sign flip in any one arm names the condition that separates our result from the paper's. No flip in any arm means the disagreement is about the model class or the task (unique solution, section 2.3), which is a stronger statement than 4.4 can make now.

**Main risk.** Four factors, one arm each, no interactions. It is a screening design; whatever flips needs a second run before it is believed. R2 confounds on-policy with fewer updates.

### Direction 3. Where the push-down should land

**Question.** Does a negative term help when it is applied only to the cells of the first-error step, scored in the state in which they were committed?

**Why open.** DACA-GRPO, DTCA and DiSPO weight tokens by heuristics of denoising progress or confidence gain, and BCPG-NSA does step-level credit in wrong answers of autoregressive models **[P]**. None can check its credit against the truth. Here the first wrong cell of every failed rollout is known from the solution **[M, 4.4b]**, and after it the true advantage is zero (difference 7).

**Link to wd1.** It asks whether the negative term fails because of what it is (unlearning) or because of where the one-step estimate puts it.

**Experiment.** Positive part as in filtered imitation. Negative part on the trajectory likelihood with a per-cell mask: (a) first-error step only, (b) cells after the first error only, (c) all cells. β from {0.25, 1}. Uses labels to locate the error, so it is an oracle study, not a method. Before any training, in the audit: AUROC of published credit scores (confidence at commit, confidence gain on the remaining cells) for finding the first-error cells; 4.2 measured 0.58 for commit confidence on wrong cells **[M]**. Files: `src/phase3/objectives.py` (per-cell weights), `src/phase3/rl.py`.

**Wall-clock.** Three arms at about 30 min, 1.5 hours; the AUROC is part of the audit.

**Prediction [C].** (a) is above (b) with a paired interval excluding zero, and (b) is at or below filtered imitation. Whether (a) beats filtered imitation I cannot call; the first-error cells are 1.9 per failed grid, a small signal.

**Interesting either way.** If (a) wins, a label-free detector of first errors becomes the thing to build, and its required AUROC is known. If not even oracle credit helps, the push-down has no use on this task with this sampler, and direction 5 is the only place left for it.

**Main risk.** The trajectory likelihood again (3.6c). Run it after direction 1.

### Direction 4. What self-training learns

**Question.** Self-training gains +0.067 at 4 steps while its targets are the stored solutions of puzzles the model already solves **[M, 4.4, 4.4b]**. Is the gain from continued SFT on any hard puzzle, from the selection of solvable puzzles, or from something tied to the model's own path?

**Why open.** RAFT and Davis and Recht explain why imitation of one's own successes is a valid algorithm **[P]**; they do not say what it adds over more supervised training when the successes equal the labels. The `supervised` objective exists in `rl.py` and was not run in phase 4.

**Experiment.** `supervised` on (a) all sampled puzzles, (b) only puzzles with at least one valid rollout in the group, (c) only puzzles with none. Same loss, same schedule, 3 seeds. (b) is self-training with labels in place of rollouts; if (b) and self-training differ, something is wrong in one of them. File: `src/phase3/rl.py`, a puzzle filter in the `supervised` branch.

**Wall-clock.** Three arms at about 30 min.

**Prediction [C].** (a) is at least as good as self-training, and (c) is the best per puzzle seen, because it trains on what the model cannot do. If so, the +0.067 is the effect of 2,000 more SFT updates at lr 1e-5 on the hard band at a plateau, and the right baseline for every RL arm in phase 4 is (a), not the start.

**Interesting either way.** If (a) wins: an honest baseline that the report needs. If self-training beats (a): selection by solvability acts as a curriculum, and the transform view of Davis and Recht has a measurable instance.

**Main risk.** Low. It can only lower the headline number of 4.4, which is the point.

### Direction 5. A learned detector of wrong cells

**What 4.5b measured [M].** With exactly the wrong cells masked and re-predicted, the start model solves 0.907 of the hard test puzzles in 4 forward passes and 1.000 from 8 on (plain sampler: 0.535 and 0.736). No available score finds the wrong cells: inside failed grids the AUROC is 0.41 to 0.49 for commit confidence, 0.55 to 0.61 for the probability with the cell masked, 0.55 to 0.59 for the rule conflict, which flags 14 to 25% of the wrong cells. Running-confidence remasking gains +0.013 to +0.024 with intervals touching zero; fill-then-fix with either real detector is below plain.

**Question.** Three in order. (a) How good must a detector be: what solve rate does a fix round give at a given recall and false-positive rate? (b) Can a detector of that quality be learned by this model from its own failed grids, with labels from the solution? (c) Can it be learned without labels, from the validity reward after the fix round?

**How close prior work is.** Close on the mechanism, so nothing here is a new method. All from abstract pages **[P]**: Token-Critic trains a second transformer to tell sampled from real tokens and uses it to choose what to resample. PRISM fine-tunes a per-token quality head for remasking, without RL or a verifier, and lists Sudoku among its tasks. Informed Correctors learn a corrector for masked diffusion. RemeDi adds a confidence stream trained by supervised remasking and then by RL, on an 8B model. ReMDM and MDPO's running-confidence rule are training-free; our `rcr` arm is of that family and gained about 0.02. Kunde et al. name remasking models as future work for per-step policy gradients. What I did not see in those abstracts: an oracle reference that separates detection from repair, the required detector quality as a curve, or a comparison of a label-trained and a reward-trained detector where the wrong cells are known. No search was run specifically for this direction after the 4.5b message; the scope is the remasking queries of section 6.

**Why the available scores fail [C].** A wrong cell never repeats a visible digit and the cells after the first error fit it **[M, 4.4b]**, so a failed grid is locally consistent around its wrong cells. Re-predicting a random quarter of the cells keeps most of the wrong cluster visible, and the model agrees with it. The oracle works because it removes the whole cluster at once. A learned detector therefore has to find a set, not score cells one by one; whether a 1.86M-parameter model can do that from the final grid is the open point, and it may be about as hard as solving.

**Link to wd1.** (1) A remask decision is an explicit action with an exact likelihood (a product of per-cell Bernoulli probabilities), so a weighted-likelihood objective on the detector has no one-step proxy and ε = 0. This is a place to test wd1's positive and negative weights free of the estimator question of direction 1. (2) "This cell is wrong" turns the push-down of a digit into a pull-up of an action, and with a fix round the actions after an error do change the reward (difference 7 of section 2.4 no longer holds). (3) At the start, successes are rare (the rule detector is below plain), the regime where the negative term helped in 3.6b **[M]** and where the wd1 paper argues for it **[P]**.

**Experiment.**
- Gate, no training of the denoiser, in `src/phase4/eraser.py`: (i) degraded oracle: mask each wrong cell with probability ρ and each right filled cell with probability φ, ρ in {0.5, 0.7, 0.9, 1}, φ in {0, 0.1, 0.3, 0.6}, solve rate at 4 and 8 passes; (ii) a probe (logistic, then a two-layer head) on the frozen final-layer features of the filled grid, trained on the failed grids of the hard train band, AUROC on failed test grids; (iii) the same probe on the features of a second pass with the conflict-flagged cells masked.
- Label-trained detector: a one-logit-per-cell head, trunk trained too, binary cross-entropy against "cell differs from the solution" on the model's own 4-step grids, with the SFT loss kept on so the denoiser does not drift. Arms: head only (frozen trunk), head and trunk. Evaluation: `fix_learned` in the 4.5b table at equal forward passes.
- Reward-trained detector, label-free: from each filled grid draw 8 remask sets from the head, run one fix round, reward = validity. Objectives on the exact action likelihood: filtered imitation of successful sets, positive-only, wd1. Initialise the head at the conflict flag.
- Seeds 0 to 2 for the trained arms; paired intervals against `plain` at the same number of passes; easy and medium bands checked for damage.

**Wall-clock.** Gate: the whole of 4.5b took 6 minutes **[M]**; the grid of 16 settings and the probes I estimate at 20 to 30 minutes. Label-trained head: about the cost of an RL run, two arms in about 1 hour. Reward-trained: three arms, about 1.5 hours.

**Predictions [C].** (1) The fix round tolerates false positives much better than missed cells: at ρ = 1, φ = 0.3 it stays above 0.80 at 4 passes; at ρ = 0.7, φ = 0 it falls below 0.70. Low confidence. (2) The probe on frozen features stays below AUROC 0.75 inside failed grids; with the trunk trained it exceeds 0.85. Low confidence in the second half. (3) If (2) holds, `fix_learned` beats `plain` at 8 passes with an interval excluding zero and does not at 4, where a fix round costs a quarter of the budget. (4) The reward-trained detector ends below the label-trained one, and among its objectives wd1 is not below positive-only.

**Interesting either way.** A learnable detector closes more of the 4-to-16-step gap than any RL arm so far, and prediction 1 says how good it has to be. If no head of this size learns it, that is a capacity statement about detection against repair on a model that repairs perfectly, and the remaining route is search over remask sets, which costs passes. Prediction 4 is the one that says something about wd1.

**Main risk.** A detector trained on the grids of the current model is off-distribution as soon as the denoiser or the step count changes. The reward-trained version has a large action space (a subset of about 53 cells) and may see no success at all without the conflict initialisation. And the result may simply reproduce PRISM at toy scale; it should be reported as that.

## 4. Free diagnostics on existing checkpoints

No training, part of the audit script, a few minutes: for the 4.4 step-500 checkpoints of plain wd1 and self-training, the change of the per-cell one-step log-probability of the solution's digit against the start, split by cell type from 4.4b (committed at step 0, 1, 2, 3; first-error cells; cascade cells). It tells where plain wd1 lost its 11 nats on valid rollouts (−57 to −68 **[M]**).

## 5. Recommended programme

**Main line: direction 1, with the gate of direction 5 beside it. Fallback: directions 2 and 4**, which need little new code and stand on their own.

First batch, one night, every launch under 2 hours, at most 3 jobs side by side:

| launch | content | estimate |
|---|---|---|
| L0 | tests; start solve rate at 2 sampler steps; audit on the start checkpoint (direction 1 without training arms, section 4 diagnostics) | 30 min |
| L0b | gate of direction 5: degraded-oracle curve and probes | 30 min |
| L1 | R1, R2, R3 | 1 h 45 min |
| L2 | R4, R5 | 1 h 40 min |
| L3 | R6, R7, R8, only if the 2-step start rate is in range | 1 h 15 min |
| L4 | direction 4 (a) and (b) | 1 h |

Total about 6 h 40 min. L4 is the one to drop if the night is short. The 4.4 launch overran its estimate by 60% **[M]**, so each launch here is planned with that margin under 2 hours.

Decisions:
- **Audit separates the objectives** (prediction 1 of direction 1 holds): run the exact arms next, then direction 3.
- **Audit shows wd1 well aligned with `g*`**: drop direction 3; the story is direction 2.
- **Reference gradient too noisy** (cosine between halves below 0.8): quadruple the rollouts once; if still noisy, switch to the max-probability rule, where the reference is exact.
- **Gate of direction 5, probe AUROC above 0.75 inside failed grids, or the degraded oracle tolerant of false positives**: train the label-based head next; it has the largest headroom on the table. Then the reward-trained arms.
- **Probe near chance and the fix round intolerant of errors**: one label-based run with the trunk trained, then stop; report detection as not learnable at this size.
- **R4 beats R5**: the dense reward is the difference with the paper. Run the dense reward at 2 steps to see whether the two factors add.
- **R6 beats R7 and R4 does not beat R5**: the start regime is the difference. Add a third success rate.
- **No arm flips**: section 2.3 is the remaining explanation; the audit's per-cell split tests it.
- **Direction 4 (a) at or above self-training**: use (a) as the reference row in every later table.

## 6. Looked for and not found

Scope: about fifteen web searches in November 2026 plus the reference lists of the papers opened. Not a systematic review.

- A follow-up to wd1 that analyses its negative term mechanistically. GDSD and SPG say it is unstable or biased and replace it.
- RL that uses DUEL-style exact likelihoods as the policy likelihood.
- A numerical comparison of the one-step or ELBO surrogate gradient against an exact or large-sample policy gradient for a masked diffusion model.
- Credit assignment for dLLM RL that uses a known first error, or the observation that post-error actions have zero advantage without remasking.
- RL on a small Sudoku masked diffusion model trained from scratch with a binary, label-free reward.
- For direction 5, from the abstracts of the remasking papers only: an oracle-detector reference, a detector-quality-to-solve-rate curve, or a label-trained against reward-trained detector comparison. The full texts of PRISM, Token-Critic, Informed Correctors and RemeDi were not read, so any of these may be in them.

## 7. Not verified

- Read in full: wd1 (HTML) and its code. Read as the first 100,000 characters of the HTML: SPG, GDSD, Kunde et al., DUEL; their later sections and appendices were not seen. All other papers: abstract page only.
- Venues come from search results, not from proceedings pages: wd1 at ICLR 2026, SPG at ICLR 2026, RemeDi at ICLR 2026, DUEL at ICML 2026, Zhu et al. and Arnal et al. at NeurIPS 2025.
- Not opened, known from search snippets only, so not used as evidence: 2607.15200 (not re-opened), 2602.19619, 2508.13021, 2511.04647, 2607.24306, 2601.11214, BGPO 2510.11683, a CTRL preprint from Columbia. NegGrad and reward-weighted regression are cited by wd1; their own papers were not opened.
- The survey gives 2509.06949 for TraceRL; a search returned a different paper (2601.11214) for a similar description. Not resolved.
- The Sudoku reward used by SPG was not found in the part read.
- Whether upstream wd1 normalises weights per group or over groups (section 2.4).
- Everything marked **[C]**, in particular the trajectory-probability formula of direction 1 and the cancellation argument of section 2.3. Both have a test listed.

## References

Opened for this memo:
- wd1: Tang, Dolga, Yoon, Bogunovic. https://arxiv.org/abs/2507.08838 ; code https://github.com/xiaohangt/wd1 (commit 1a2e254)
- SPG: Wang et al. https://arxiv.org/abs/2510.09541
- GDSD: Tang et al. https://arxiv.org/abs/2605.29398
- Kunde et al., per-step policy gradient for dLLMs. https://arxiv.org/abs/2603.12554
- DUEL: Turok, Schiff, De Sa, Kuleshov. https://arxiv.org/abs/2603.01367
- DiFFPO: Zhao et al. https://arxiv.org/abs/2510.02212
- Discrete Tilt Matching: Chen et al. https://arxiv.org/abs/2604.18739
- DACA-GRPO: Monsefi et al. https://arxiv.org/abs/2605.16342
- DTCA: Chen et al. https://arxiv.org/abs/2610.35201
- DiSPO: Oba, Furuta, Okazaki. https://arxiv.org/abs/2602.06462
- Davis, Recht. https://arxiv.org/abs/2510.13651
- RAFT: Xiong et al. https://arxiv.org/abs/2504.11343
- Arnal et al., asymmetric REINFORCE. https://arxiv.org/abs/2506.20520
- Zhu et al., negative sample reinforcement. https://arxiv.org/abs/2506.01347
- LENS: Feng et al. https://arxiv.org/abs/2510.08696
- BCPG-NSA: Yang et al. https://arxiv.org/abs/2505.14403
- Kim et al., early decisions and planner. https://arxiv.org/abs/2604.10567
- RemeDi: Huang et al. https://arxiv.org/abs/2509.23653
- PRISM: Kim et al. https://arxiv.org/abs/2510.01384
- ME-DLM: Ren et al. https://arxiv.org/abs/2605.09603
- Informed Correctors: Zhao et al. https://arxiv.org/abs/2407.21243
- Token-Critic: Lezama et al. https://arxiv.org/abs/2209.04439
- ReMDM: Wang, Schiff, Sahoo, Kuleshov. https://arxiv.org/abs/2503.00307
- Abstracts re-checked, discussed in the survey: MDPO https://arxiv.org/abs/2508.13148 ; IGPO https://arxiv.org/abs/2509.10396 ; SEPO https://arxiv.org/abs/2502.01384 ; DDPP https://arxiv.org/abs/2410.08134 ; MMaDA https://arxiv.org/abs/2505.15809 ; AWR https://arxiv.org/abs/1910.00177 ; PAPL https://arxiv.org/abs/2509.23405 ; P2 https://arxiv.org/abs/2502.03540

Named but not opened (see section 7): https://arxiv.org/abs/2607.15200 ; https://arxiv.org/abs/2601.11214 ; https://arxiv.org/abs/2510.11683
