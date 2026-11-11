# RL for masked diffusion language models: survey and proposals for phase 4

Written 2026-11-10, while step 4.4 was still running. This is a reading note for a learning project, not a claim of novelty: every proposal below is an existing idea applied to this repo's setup, phrased as "worth testing here".

**How to read the sources.** Each paper has its arXiv id. Unless marked otherwise, the arXiv abstract page or HTML version was opened. Three marks are used:
- *(abstract)*: only the abstract was read.
- *(HTML, first 100k chars)*: the HTML page was read up to 100,000 characters; later sections and appendices were not.
- *(not verified)*: not opened; known only through another paper, a search result, or memory. Treat as a pointer, not as a fact.

Numbers are copied from the papers as reported there. They come from LLaDA-8B-scale models with LoRA on text-formatted tasks, so they do not transfer to a 1.9M-parameter model on a grid.

## 1. Our setup in one paragraph

A 1,876,489-parameter masked diffusion transformer trained on 9×9 Sudoku to a plateau (step 4.1). RL starts on the hard band (51 to 56 hidden cells) with **4 sampler steps**, where the greedy solve rate is 0.521 on val and 0.535 on test, against 0.805 on val with 16 steps (4.3, 4.4). Reward is binary (valid grid that keeps the givens). Each RL step uses 32 puzzles × 8 rollouts at temperature 1 with the deterministic `confidence` order, and the one-step likelihood estimate of d1. Four measured facts shape the proposals:

1. **Order matters and confidence is over-confident** (4.2): random order solves 0.128 of hard test puzzles at 16 steps, confidence 0.814, an oracle order 667 of 667. Wrong cells are committed at mean confidence 0.975; cell AUROC 0.58.
2. **Step count matters** (4.2): hard band 0.003 / 0.535 / 0.736 / 0.814 / 0.871 at 1 / 4 / 8 / 16 / 81 steps.
3. **Three groups in four carry no group-relative signal** (4.3): 24.4% of groups of 8 have unequal rewards; 44% are all valid, 31% all invalid.
4. **Pushing down invalid rollouts without a bound hurt on 4×4** (phase 3): a floor or a KL term fixed it; the one-step likelihood trained better than the sequential and trajectory estimates, for reasons not found.

Preliminary and uncommitted: at the time of writing the 4.4 logs held only the first ~200 steps of plain wd1 (seeds 0 to 2): val solve 0.491 / 0.512 / 0.509 against a start of 0.521, 75 to 88% of groups skipped, `logp_valid` falling from about −52 to between −61 and −66. This is not a result; 4.4 will say what it is.

## 2. Survey

### 2.1 Background

| Paper | What it is | Why it matters here |
|---|---|---|
| MDLM, Sahoo et al., [2406.07524](https://arxiv.org/abs/2406.07524) | Masked diffusion objective as a weighted mixture of masked-LM losses | Our SFT loss (`diffusion_loss`, 1/t weighting) |
| SEDD, Lou et al., [2310.16834](https://arxiv.org/abs/2310.16834) | Score-entropy discrete diffusion | Background for SEPO below |
| LLaDA, Nie et al., [2502.09992](https://arxiv.org/abs/2502.09992) | 8B masked diffusion LM, low-confidence remasking sampler | Base model of almost every paper in 2.2 |
| DDPO, Black et al., [2305.13301](https://arxiv.org/abs/2305.13301) | Denoising as a multi-step MDP; policy gradient reported better than reward-weighted likelihood (images) | The "each sampler step is an action" view that 2.2's trajectory methods follow |
| DPOK, [2305.16381](https://arxiv.org/abs/2305.16381) | Policy gradient plus KL regularisation to the pretrained model (images) | Precedent for our KL arm |
| AWR, Peng et al., [1910.00177](https://arxiv.org/abs/1910.00177) | Advantage-weighted regression | wd1's positive term is of this family |
| RWR, Peters and Schaal 2007 | Reward-weighted regression | *(not verified, from memory)* |

### 2.2 RL fine-tuning of masked diffusion LMs

Method families, by how they estimate the likelihood of a sampled answer:

**(a) One-step mean-field proxy.**
- **d1 / diffu-GRPO**, Zhao, Gupta, Zheng, Grover, [2504.12216](https://arxiv.org/abs/2504.12216) *(HTML, first 100k chars)*. GRPO with a per-token log-probability from one forward pass on a fully masked answer, with the prompt randomly masked (p = 0.15) as regulariser. Sudoku there is 4×4 in text, reward = share of empty cells filled correctly; LLaDA 11.7 / 6.7 / 5.5 and d1 22.1 / 16.7 / 9.5 at lengths 128 / 256 / 512. *For us:* this is the `onestep` likelihood and the GRPO arm of 4.4. Note the dense reward; ours is binary.
- **wd1**, Tang, Dolga, Yoon, Bogunovic, [2507.08838](https://arxiv.org/abs/2507.08838). Removes the policy ratio: weighted log-likelihood that raises high-advantage samples and lowers low-advantage ones. Reports wd1++ at 44.2% MATH500 and 84.5% GSM8K in 20 RL steps. *For us:* the main arm of phases 3 and 4. Its push-down term is the one that needed a floor on 4×4.

**(b) Better likelihood or ELBO estimates, still one action per sequence.**
- **LLaDA 1.5 / VRPO**, [2505.19223](https://arxiv.org/abs/2505.19223). Preference optimisation with variance reduction for the ELBO (budget allocation over timesteps, antithetic sampling). *For us:* our model is small enough that many ELBO samples are cheap, but phase 3 found the more exact estimates trained worse.
- **GDPO**, [2510.08554](https://arxiv.org/abs/2510.08554). Semi-deterministic Monte Carlo ELBO; states that the one-step estimate is severely biased.
- **SPG**, Wang et al., [2510.09541](https://arxiv.org/abs/2510.09541) *(HTML, first 100k chars)*. Lower bound (ELBO) for positive-advantage samples, upper bound (EUBO, Rényi) for negative ones, mixed with ω = 0.5; block-wise masking. Sudoku (length 256): SPG 94.0, UniGRPO 67.0, wd1 32.1, d1 32.5. Ablation: without the negative loss 68.8 against 90.0 (Sudoku average); ELBO on negatives 82.4 and an early plateau. Rollout temperature 0.3 on Sudoku. *For us:* direct evidence, on their setup, that negatives carry information and that how they are bounded matters. Their rollout temperature is lower than ours.
- **ESPO**, Ou et al., [2512.03759](https://arxiv.org/abs/2512.03759). Whole sequence as one action, ELBO as proxy, per-token ratio normalisation, robust KL. Reports +20 to +40 points on Countdown.
- **StableDRL**, [2603.06743](https://arxiv.org/abs/2603.06743) *(HTML, first 100k chars)*. Attributes GRPO collapse to noisy ratios; unconditional clipping and self-normalised weights. Its table (Sudoku average): d1 23.0, wd1 29.2, UniGRPO 63.0, ESPO 86.0, SPG 90.0, StableDRL 91.5.
- **RSPO**, [2605.10218](https://arxiv.org/abs/2605.10218). Uses the advantage as a regression target for the relative log-ratio.

**(c) Likelihood-free: match a reward-tilted target.**
- **DMPO**, Zhu et al., [2510.08233](https://arxiv.org/abs/2510.08233). Weighted denoising cross-entropy towards a reward-tilted distribution, with a weight baseline. *For us:* close to our "self-training with SFT loss" arm, with soft weights instead of a filter.
- **GDSD**, Tang et al. (wd1 authors), [2605.29398](https://arxiv.org/abs/2605.29398) *(HTML, first 100k chars)*. Distils the denoiser towards an advantage-guided teacher by squared logit matching at masked states. No separate negative penalty; the paper says wd1's penalty can destabilise training, and that a forward-KL version recovers wd1 and DMPO. Sudoku average: ESPO 86.0, GDSD 89.4, with its extra component (TLC) 91.0. *For us:* the wd1 authors themselves moved away from the unbounded push down, which agrees with phase 3.
- **LFPO**, [2603.01563](https://arxiv.org/abs/2603.01563). Likelihood-free alignment on denoising logits, aimed at fewer steps.

**(d) Per-step or trajectory policy gradient (the sampler steps are the actions).**
- **MDPO**, He, Renz, Cao, Geiger, [2508.13148](https://arxiv.org/abs/2508.13148). RL along the inference schedule with a reward at each intermediate step, PPO clipping, KL 0.08. Names "over-denoising" (a correct intermediate answer later changed to a wrong one) and a training-free Running Confidence Remasking. Countdown 70.7 / 71.1 against 40.1 / 58.9 for diffu-GRPO. No Sudoku. *For us:* the closest in spirit to "learn to solve in 4 steps": it trains on the states the sampler really visits.
- **AGRPO**, Zhan, [2510.04019](https://arxiv.org/abs/2510.04019). Policy gradient over sampled denoising steps (m = 128, k = 24); unmasking positions treated as exogenous, random unmasking in training; dense reward. 4×4 Sudoku 95.2 / 95.2 against SPG 94.0 / 93.1, d1 32.5 / 29.3, wd1 32.1 / 22.5.
- **DCoLT**, Huang et al., [2505.10446](https://arxiv.org/abs/2505.10446). Outcome reward over the whole reverse chain; for LLaDA, an Unmasking Policy Module with a Plackett-Luce distribution over positions, trained together with the tokens. Reports +9.8 on GSM8K.
- **MAPG**, Raajesh, Shah, Klivans, Krähenbühl, [2607.15200](https://arxiv.org/abs/2607.15200) *(HTML, first 100k chars)*. Two-stage action (which positions, then which tokens). The unmask set is sampled from a Plackett-Luce over confidence scores (position temperature 0.5); the gradient has a token term and a masking term from the same forward passes. Sudoku 84.5 / 94.9 / 96.2 against SPG 82.9 / 94.0 / 93.1; on GSM8K the token-only variant gets 78.0 against 81.0. *For us:* the most direct template for proposal 2. No masking-term-only ablation was seen in the part read.
- **TraceRL / TraDo**, [2509.06949](https://arxiv.org/abs/2509.06949). Trajectory-aware RL with a diffusion value model.
- **CJ-GRPO**, Yang et al., [2509.23924](https://arxiv.org/abs/2509.23924). Makes the optimisation trajectory consistent with the rollout trajectory; also targets fewer steps.
- **d-TreeRPO**, [2512.09675](https://arxiv.org/abs/2512.09675). Tree-structured rollouts, bottom-up advantages, time-scheduled self-distillation; argues that higher confidence shrinks the gap between the one-step probability and the order-expected probability. *For us:* a possible reason the one-step proxy did well in phase 3 (our models are very confident), not tested.
- **Continuous-time RL**, Zhang et al., [2607.14522](https://arxiv.org/abs/2607.14522). Policy gradients in the CTMC view, intermediate advantages, trajectory subsampling.
- **STP**, [2602.08905](https://arxiv.org/abs/2602.08905). Spatio-temporal pruning of the trajectory for cheaper RL.

**(e) Helping exploration when groups fail.**
- **IGPO**, [2509.10396](https://arxiv.org/abs/2509.10396). Inpaints parts of a ground-truth trace into the canvas so that all-fail groups produce some successes. *For us:* addresses the 31% all-invalid groups; our givens already play this role, and more could be revealed.
- **CanvasAnneal**, [2609.13060](https://arxiv.org/abs/2609.13060). Teacher traces in the initial canvas, annealed away.

**Survey.** "Reinforcement Learning-driven Diffusion Models: Language, Vision, and Beyond", [2610.21539](https://arxiv.org/abs/2610.21539). It lists further work that was *not opened*: BGPO 2510.11683, DiffuCoder 2506.20639, d2 2509.21474, DiFFPO 2510.02212, ERILS 2608.01717, ELBO-KTO 2510.23658, SAPO 2510.01544, RFG 2509.25604, DiRL 2512.22234, DARE 2604.04215. UniGRPO (from MMaDA), StepMerge and GSPO were seen only as baselines or components inside other papers *(not verified)*.

### 2.3 Unmasking order, planners and self-correction

- **Kim et al.**, [2502.06768](https://arxiv.org/abs/2502.06768) *(HTML, first 100k chars)*. A 6M-parameter MDM on Sudoku: random order 6.88%, top probability 18.51%, top margin 89.49%; on a hard set margin gets 49.88%. *For us:* closest published small-model Sudoku result. It disagrees with 4.2, where margin and confidence are equal within 0.009. Likely reasons: their sampler and data differ, and our comparison is deterministic top-k (see Mohamud et al. below, where the confidence rule is 18.71 stochastic and 92.63 deterministic).
- **Ye et al. (MGDM)**, [2410.14157](https://arxiv.org/abs/2410.14157). Discrete diffusion at 100% on Sudoku against 20.7% autoregressive.
- **Adaptive order policies**, Mohamud, Hasan, Ravanelli, Bengio, [2606.00295](https://arxiv.org/abs/2606.00295). A ~50K-parameter MLP on confidence and hidden state, trained with supervision (no RL), on a frozen or jointly trained denoiser. 9×9 Sudoku, 6M model, 20 steps: learned 93.75, margin 91.53, high confidence 18.71 (stochastic) or 92.63 (deterministic), oracle 100. *For us:* same task size, same finding as 4.2 (oracle order solves everything), and a small gain from a learned order. Agrees with phase 3.4b that the hidden state is the useful feature.
- **Hong et al.**, [2510.05725](https://arxiv.org/abs/2510.05725) *(HTML, first 100k chars)*. Order policy as a KL-regularised MDP around the max-confidence order, frozen LLaDA. Sudoku 0.817 against confidence 0.705, margin 0.713, random 0.616.
- **Jazbec et al.**, [2512.09106](https://arxiv.org/abs/2512.09106) *(HTML, first 100k chars)*. Frozen dLLM, one-layer policy on confidences with per-token Bernoulli decisions, GRPO, reward multiplied by a step-count term. Better than heuristics at low step counts in full diffusion mode. No Sudoku.
- **dUltra**, [2512.21446](https://arxiv.org/abs/2512.21446). Planner head trained jointly with the base model by GRPO, initialised by imitating a heuristic; clipping advantages at 0 stabilised training. GSM8K 81.29% at 21.2 function evaluations.
- **SAS, "Scheduling Thoughts"**, [2606.23567](https://arxiv.org/abs/2606.23567) *(HTML, first 100k chars)*. Order head on a frozen denoiser, GRPO, reward = teacher-forced log-likelihood of the label along the order. Sudoku (SMDM-1B): confidence 82.0, learned order 91.8, then SFT of the denoiser along the learned order 97.5 (human order 97.8). *For us:* the second stage (train the denoiser on the states its own order visits) is the idea behind proposal 4.
- **Where to unmask**, [2602.09501](https://arxiv.org/abs/2602.09501). An oracle margin order gives 0.995 on Sudoku with LLaDA against 0.105 to 0.110 for heuristics; a supervised ranking planner does not beat margin on Sudoku (0.085). *For us:* a learned order is not guaranteed to help; same shape as our 3.4b transfer failure.
- **PAPL**, [2509.23405](https://arxiv.org/abs/2509.23405). Planner-aware ELBO: train on the masks the planner produces instead of uniform masks.
- **P2**, [2502.03540](https://arxiv.org/abs/2502.03540), and **ReMDM**, [2503.00307](https://arxiv.org/abs/2503.00307). Samplers that may remask committed tokens.
- **PRISM**, Kim et al., [2510.01384](https://arxiv.org/abs/2510.01384) *(abstract)*. Plug-in per-token quality head trained with a self-correction loss, no RL, no verifier; flags tokens for remasking; evaluated on Sudoku among others (numbers not seen).
- **RemeDi**, [2509.23653](https://arxiv.org/abs/2509.23653) *(not verified, search summary)*. Confidence head with remasking, SFT then RL.
- **Self-aware Markov models**, [2603.16661](https://arxiv.org/abs/2603.16661). Train on the model's own outputs, remask, learned stopping; about 95% on Sudoku-Extreme.
- **Self-distillation through time**, Deschenaux and Gulcehre, [2410.21035](https://arxiv.org/abs/2410.21035) *(abstract)*. Distils a many-step discrete diffusion sampler into a few-step one (32 to 64 times fewer steps). *For us:* precedent for proposal 4b.
- Reflective Masking 2606.16700 *(not verified)*.

### 2.4 Other reward-optimisation families (not used in the proposals)

SEPO, Zekri and Boullé, [2502.01384](https://arxiv.org/abs/2502.01384) *(abstract)*: policy gradient for score-entropy models. DRAKES, [2410.13643](https://arxiv.org/abs/2410.13643): backpropagation through Gumbel-softmax samples with KL to the pretrained model; needs a differentiable reward, ours is not. SVDD, [2408.08252](https://arxiv.org/abs/2408.08252): value-guided decoding without fine-tuning. DDPP, [2410.08134](https://arxiv.org/abs/2410.08134), and RTB, [2405.20971](https://arxiv.org/abs/2405.20971): posterior-sampling objectives (GFlowNet view). TR2-D2, [2509.25171](https://arxiv.org/abs/2509.25171): tree search plus a replay buffer. MDNS, [2508.10684](https://arxiv.org/abs/2508.10684); SQDF, [2512.04559](https://arxiv.org/abs/2512.04559). Twisted SMC 2505.22524 and VGAS 2608.26585 *(not verified)*.

Equal-reward groups in GRPO, autoregressive LLMs only: "Gradient starvation in binary-reward GRPO", [2605.07689](https://arxiv.org/abs/2605.07689): replaces the group-normalised advantage with a sign advantage A = 2r − 1 so that all-correct and all-wrong groups still give a gradient; reports GSM8K 73.8% against 28.4%. AVSPO, NGRPO and RL-ZVP address the same problem *(not verified)*.

### 2.5 What the literature says about our four facts

1. **Order.** Three independent sources find an oracle order near 100% on Sudoku with a fixed denoiser (2602.09501, 2606.00295, our 4.2). Learned orders give small gains over a good heuristic (2606.00295: +2.2 points; 2510.05725: +10 on LLaDA) or none (2602.09501). Joint training of order and tokens by RL exists (DCoLT, MAPG, dUltra) but was not found on a small from-scratch model.
2. **Few steps.** MDPO, CJ-GRPO, LFPO, Jazbec et al. and dUltra all treat the step count as part of the problem. The shared idea: optimise on the states the few-step sampler visits, not on uniformly masked states.
3. **Equal-reward groups.** For dLLMs only the all-fail case was found (IGPO, CanvasAnneal). The all-success case, 44% of our groups, was found only for autoregressive LLMs (2605.07689).
4. **Pushing down negatives.** The field has moved from "unlearn negatives" (wd1) to bounded forms: an upper bound (SPG), clipping (StableDRL, dUltra clips advantages at 0), or no explicit penalty (GDSD). SPG's ablation still says that dropping negatives entirely costs a lot. This matches phase 3: keep the negatives, bound them.

## 3. Proposals, ranked by information per compute hour

**Baseline for all of them:** the best arm of 4.4 at step 500 (mean of seeds 0 to 2, val and test solve at 4 steps, and at 16 steps as a side check), with a paired bootstrap interval over test puzzles as in 4.2 and 4.4. 4.4 was not finished when this was written, so the baseline is named by rule, not by arm. If no 4.4 arm beats the start (0.521 val, 0.535 test), the baseline is the start itself, and that is a result to report.

**Cost unit:** one arm = 3 seeds × 7 min = 21 run-minutes (about 7 to 10 minutes of wall-clock if the three seeds run at once, which is the limit of 3 MPS jobs). Arms that add forward passes cost more; estimates are given and are guesses until timed on one 25-step run.

### Proposal 1. Three measurements before any training (about 15 minutes, no gradient step)

*Idea.* Three questions decide which of the later proposals can work at all, and each needs only rollouts from the start checkpoint.

- **1a. Do wrong cells follow an earlier mistake?** For each failed 4-step and 16-step rollout on the hard test band, find the first step with a wrong cell; count wrong cells committed at that step and after it. Also split first errors into "wrong on its own" and "two cells of the same step that conflict with each other" (same digit in one row, column or box, committed together). This tests the reading left open in 4.2.
- **1b. Does order noise create unequal-reward groups?** Rerun the 4.3 measurement with a Plackett-Luce order (Gumbel noise added to log confidence, temperature τ_o in {0.25, 0.5, 1, 2}) and report valid share, share of groups with unequal rewards, all-valid and all-invalid shares.
- **1c. Are valid rollouts just the dataset solution?** Share of valid rollouts on train puzzles that equal the stored solution. If it is 1.0 (unique solutions), then self-training and filtered imitation on train puzzles are SFT on a subset of data the model has already plateaued on, with uniform masks; that would predict a small gain for those two arms in 4.4.

*Where.* A new script next to `src/phase4/rl_start.py` and `src/phase4/sampler_eval.py`; 1b needs a `gumbel` option in `src/phase2/sampler.py::select_topk` (also needed by proposal 2).

*Prior work.* 1a is the "over-denoising" and error-propagation analysis of MDPO (2508.13148) and the motivation of PRISM (2510.01384), done on our model. 1b is the position sampling of MAPG (2607.15200) and DCoLT (2505.10446), measured before being trained.

*Predictions.* 1a: more than half of wrong cells are committed after the first-error step at 16 steps (if not, the "consistent with a wrong grid" reading is wrong and errors are independent). At 4 steps, a large share of first errors are same-step conflicts (if not, few-step failure is not mainly a joint-commit problem and proposal 4 loses its reason). 1b: some τ_o raises unequal-reward groups from 0.24 to above 0.40 while keeping the valid share above 0.35 (if no τ_o does, proposal 2 loses its exploration argument). 1c: the share is above 0.99.

*Cost.* About 15 minutes total, no training. *Risk.* None beyond the time; the only way to waste it is to skip it.

### Proposal 2. Sample the unmasking order in rollouts, and put the order in the gradient

*Idea.* Today all 8 rollouts of a puzzle use the same deterministic top-k order; they differ only in sampled digits, and the model is so confident that they rarely differ. Sample the order instead: at each sampler step draw the cells to commit from a Plackett-Luce distribution over confidence (Gumbel top-k, temperature τ_o). Two variants, in this order:
- **2a, order as exploration only.** Sampled order in rollouts, objective unchanged (order treated as fixed in the likelihood, as `trajectory_cell_logp` already does, or the one-step proxy).
- **2b, order in the gradient.** Add the log-probability of the sampled order, which depends on the denoiser through its confidences, to the weighted objective. The denoiser is then rewarded for being confident on cells that were good to commit early, which is the calibration problem of 4.2 addressed through the reward.

Evaluation stays greedy and deterministic, as now.

*Where.* `src/phase2/sampler.py::select_topk` and `sample` (Gumbel option, return the within-step rank of each committed cell); `src/phase3/rl.py`, a new `order_logp` next to `trajectory_cell_logp`, added to the log-likelihood used by `weighted_loss`; a config switch `rl.order: fixed|sampled|sampled+grad` and `rl.order_temp` in `configs/phase4_rl.yaml`. Unit test: with τ_o near 0 the sampled order equals the deterministic one and the phase 3 regression guard still passes; the order log-probability sums to 1 over all orders on a 3-cell toy.

*Prior work.* This is MAPG (2607.15200) applied to our setup, with the difference that MAPG uses its own trajectory approximation and a ratio objective on LLaDA-8B, and we would use the floored wd1 weights and exact 4-step trajectories (4 forward passes are affordable here). DCoLT (2505.10446) and RemeDi learn a separate order module; here there is no new module, the order comes from the denoiser's confidence. 2a alone is close to AGRPO's treatment of positions as exogenous (2510.04019). Phase 3.4b trained an order head on a frozen denoiser; this is the complementary case (no head, denoiser trained).

*Prediction.* (i) The share of skipped groups falls from about 0.75 to below 0.6 (from 1b). (ii) 2a ends above the best 4.4 group-relative arm on test solve at 4 steps, paired interval excluding zero. (iii) 2b ends above 2a, and the mean commit confidence of wrong cells (4.2 calibration, 16 steps) falls below 0.975 after 2b but not after 2a. Deciding measurement: test solve at 4 steps at step 500, three seeds, plus the 4.2 calibration table on the final checkpoints. If (ii) fails while (i) holds, more unequal groups was not the bottleneck.

*Cost.* 2a: 21 run-minutes. 2b needs the trajectory likelihood (4 forward passes with gradient per inner step instead of 1): estimated 2 to 3 times, so 45 to 60 run-minutes. Together about 1.3 hours.

*Risks.* Phase 3 found that the trajectory likelihood trained worse than the one-step proxy, cause unknown; 2b inherits that. Rollouts with a noisy order are worse on average (random order: 0.029 at 4 steps), so training may improve the noisy sampler and not the greedy one that is evaluated. The order term can also be gamed by lowering confidence everywhere; watch mean confidence and the 16-step solve rate.

### Proposal 3. Use the equal-reward groups: fixed baseline instead of the group mean, with the floor

*Idea.* The group mean as baseline gives zero advantage to every rollout in an all-valid or all-invalid group, 75% of the batch. Replace it with a baseline that does not depend on the group: either a constant (sign advantage, A = +1 for valid, −1 for invalid) or the running mean of the batch reward (about 0.56 at the start). All-valid groups then pull up, as in self-training; all-invalid groups push down, which phase 3 says must be bounded, so the floor of 4.4 (−75.6) stays on.

*Where.* `src/phase3/objectives.py::advantages` (new `baseline: group|batch|sign` option) and the `keep` filter in `src/phase3/rl.py` that skips equal-reward groups. Unit test: with `baseline: group` the weights are unchanged; with `sign`, an all-valid group gets equal positive weights.

*Prior work.* The sign advantage of "Gradient starvation in binary-reward GRPO" (2605.07689), which is for autoregressive LLMs and GRPO; here it is applied to wd1 weights on a masked diffusion model, with the floor from phase 3. With positive weights only it reduces to filtered imitation, an existing 4.4 arm, so the new information is in the bounded negatives of all-invalid groups.

*Prediction.* Floored wd1 with a batch baseline ends above floored wd1 with the group baseline (paired interval excluding zero), because the effective batch is about 4 times larger. Sharper version: it also ends above filtered imitation; if it only matches filtered imitation, the negatives from all-invalid groups add nothing here, against SPG's ablation on LLaDA. Deciding measurement: test solve at 4 steps at step 500; diagnostic `logp_valid` should not fall below the floored arm's.

*Cost.* 2 arms (batch, sign) × 21 = 42 run-minutes.

*Risk.* The all-invalid groups are probably the puzzles the model cannot solve in 4 steps at all; pushing their samples down, even with a floor, removes probability without saying where to put it. Phase 3's positive-only collapse came from weights spread over invalid rollouts; check `w_pos_on_invalid` stays at 0 with these baselines.

### Proposal 4. Train on the states the 4-step sampler visits, with better targets

*Idea.* The imitation arms of 4.4 apply the SFT loss with uniformly random masks to the model's own valid 4-step rollouts. Two things are off for the question "solve in 4 steps": the masked states are not the ones the 4-step confidence sampler visits, and the targets come only from puzzles the model already solves in 4 steps. Two changes, testable separately:
- **4a, path-matched states.** For each kept rollout, build the training inputs from its own trajectory: the 4 partially filled grids the sampler saw (cells committed before step s shown, the rest masked), and take the cross-entropy on all still-masked cells against the final grid.
- **4b, slow teacher.** Sample the rollouts with 16 steps (valid share 0.84 at T = 1 against 0.56 with 4 steps, from 4.3), keep the valid ones as targets, and train the 4-step states of 4a towards them. This is step distillation with a validity filter.

A control that uses labels and no RL at all belongs here: 4a with the dataset solution as target on all train puzzles (planner-aware SFT). If the control matches 4b, sampling and filtering added nothing over the labels.

*Where.* The self-training branch of `src/phase3/rl.py` (where `diffusion_loss` is called on the model's own rollouts) and a new `path_loss` beside `src/phase2/train_diffusion.py::diffusion_loss`; `sample` already returns `commit_step`, which is enough to rebuild the visited states. Config: `rl.self_states: random|path`, `rl.teacher_steps: 4|16`.

*Prior work.* 4a is PAPL (2509.23405) and the second stage of SAS (2606.23567), which reports 91.8 to 97.5 on Sudoku from SFT along a learned order, with our own heuristic order instead of a learned one and filtered own samples instead of labels. 4b is self-distillation through time (2410.21035) with a reward filter; d-TreeRPO (2512.09675), CJ-GRPO (2509.23924) and LFPO (2603.01563) contain related few-step ideas. MDPO (2508.13148) trains on visited states with a policy gradient; this is the imitation version.

*Prediction.* 4a ends above uniform-mask self-training (the 4.4 arm) on test solve at 4 steps. 4b ends above 4a and closes at least a third of the gap between 4 and 16 steps on test (0.535 to 0.814, so above 0.63), while the 16-step solve rate does not fall by more than 0.02. If the labelled control equals 4b within the interval, report that RL-style sampling was not needed. If 1a shows few same-step conflicts, expect 4a to fail.

*Cost.* 4a: 21 run-minutes (4 forward passes per inner step with gradient may raise it; time one 25-step run first). 4b: 16-step rollouts roughly quadruple the sampling cost; estimate 40 to 60 run-minutes. Control: 21. Total about 1.5 to 1.7 hours.

*Risk.* The gap between 4 and 16 steps may be a capacity limit of a 1.9M model (predicting 13 or more cells jointly from factorised outputs), not a training-distribution problem; then nothing here moves it. Training only on path states may degrade the model on uniform masks, visible as a drop at 16 and 81 steps.

### Proposal 5. A different bound on the push down (only if 4.4 shows the floor or KL arm unstable)

*Idea.* The floor is our own fix. The literature has others for the same problem: (i) SPG's mixture for negative-advantage samples, log(ω·p + (1−ω)·stopgrad-reference) style upper bound, which saturates instead of cutting; (ii) GDSD's squared logit matching towards an advantage-shifted copy of the model's own logits, with no negative likelihood term at all; (iii) advantage clipping at 0 (dUltra), which is the positive-only arm with group-relative weights.

*Where.* `src/phase3/objectives.py` (new entries in `KIND` and `WEIGHTED`), reusing `MixedLogits` and `rl.ref_mix` in `src/phase3/rl.py`, which already implement a mixture with the reference model.

*Prior work.* SPG (2510.09541) and GDSD (2605.29398) directly; the difference is only the scale and that we compare them against a floor and a KL term on the same runs.

*Prediction.* On this setup the three bounded forms (floor, mixture, logit matching) end within each other's intervals, and all above plain wd1. A clear winner would be the surprise. Deciding measurement: test solve at step 500 and the lowest val solve along the run (stability).

*Cost.* 2 arms × 21 = 42 run-minutes; GDSD needs one extra forward pass, small.

*Risk.* Low information if 4.4 already shows floor and KL equal and stable; that is why it is ranked fifth.

### Proposal 6. Self-correction at inference (no RL; a check on what RL is competing with)

*Idea.* If 1a shows that failures start from one early wrong commit, undoing commits may be cheaper than retraining. Two steps: a training-free remasking rule (Running Confidence Remasking from MDPO: a committed cell can be masked again when its current confidence is among the lowest), then, only if that helps, a small quality head on the frozen denoiser in the style of PRISM, which extends the confidence token of phase 3.3b.

*Where.* A `remask` option in `src/phase2/sampler.py::sample`; the head would extend `src/phase3/conf_token.py`.

*Prior work.* MDPO's RCR (2508.13148), ReMDM (2503.00307), P2 (2502.03540), PRISM (2510.01384) applied as they are.

*Prediction.* With cell AUROC 0.58 at commit time, remasking by confidence does not help at 16 steps (within ±0.02 of 0.814 on hard test) unless confidence becomes informative after later commits; measure the AUROC of the *final-state* confidence of committed cells first. If that AUROC is above 0.8, RCR at 16 steps gains more than 0.03.

*Cost.* Training-free part: about 10 minutes. Head: about 20 run-minutes.

*Risk.* At 4 steps there is no step budget to spend on remasking, so this speaks to the 16-step setting, not to the 4.4 question.

### A factorial design for proposals 2 and 3

Order (fixed / sampled, 2a) × baseline (group mean / batch mean), all with floored wd1, one-step likelihood, 500 steps, seeds 0 to 2. The fixed-order, group-mean cell is the floored arm of 4.4 and can be reused if the code path is bit-compatible (check with the regression guard), leaving 3 new arms = 63 run-minutes. It answers whether the two ways of getting signal out of more groups add up or overlap: sampled order turns equal groups into unequal ones, the batch baseline uses equal groups as they are. Expected, to be checked: the two effects overlap (sub-additive), because both draw on the same 75% of the batch. Adding objective (floored wd1 / path-matched self-training) as a third factor doubles this to 7 new arms, about 2.5 run-hours; that crosses the 2-hour line of `CLAUDE.md` and needs the user's OK.

## 4. Recommended first experiment

Run **proposal 1** (15 minutes, no training) as soon as 4.4 has finished and its results are written. It costs almost nothing and each of its three numbers switches a later proposal on or off:
- 1b decides whether the factorial of proposals 2a and 3 is worth its 63 minutes (run it if some τ_o lifts unequal-reward groups above 0.40 with a valid share above 0.35; otherwise run only the baseline arm of proposal 3).
- 1a decides between proposal 4 (same-step conflicts dominate at 4 steps) and proposal 6 (single early errors dominate at 16 steps).
- 1c says how to read the two imitation arms of 4.4.

Then the 2 × 2 factorial, then proposal 4 with its labelled control. Write the predictions above into the experiment README before each run, as in 4.4.

## 5. Looked for and not found

- An RL fine-tuning study on a small (about 2M parameters) from-scratch masked diffusion model on 9×9 Sudoku. Small-model Sudoku papers found are supervised (2502.06768, 2606.00295, 2410.14157); RL papers use LLaDA-scale models and mostly 4×4 Sudoku in text.
- An analysis showing that over-confident wrong commits follow an earlier error. MDPO's over-denoising and PRISM's motivation are related but not this measurement.
- A masking-term-only ablation in MAPG (not in the first 100k characters).
- Trajectory-balance or GFlowNet objectives applied to dLLM reasoning tasks with binary reward (RTB and DDPP exist as general methods).
- Work on equal-reward groups specific to dLLMs beyond the all-fail case (IGPO, CanvasAnneal). The all-success case was found only for autoregressive LLMs.
- A paper explaining why a one-step proxy can train better than a more exact trajectory likelihood (our phase 3 finding). GDPO and StableDRL say the opposite, that the one-step estimate is biased or inaccurate; d-TreeRPO's remark on confidence is the only hint found.
- A comparison of RL against plain planner-aware SFT when labels are available, which is the control in proposal 4.

## 6. Not verified

- Not opened at all: RWR (Peters and Schaal 2007); UniGRPO, StepMerge, GSPO; the ten ids listed under the survey 2610.21539; Reflective Masking 2606.16700; twisted SMC 2505.22524; VGAS 2608.26585; AVSPO, NGRPO, RL-ZVP; RemeDi 2509.23653 (search summary only). The Gumbel top-k fact used in proposal 2 (adding Gumbel noise and taking the top k gives an ordered sample without replacement from a Plackett-Luce distribution; Kool et al. 2019) is from memory.
- Abstract only: PRISM 2510.01384, self-distillation through time 2410.21035, SEPO 2502.01384.
- Read only to 100,000 characters of the HTML: d1, SPG, GDSD, StableDRL, MAPG, Jazbec et al., SAS, Hong et al., Kim et al. Appendix details and later ablations of these may contradict a statement above.
- Venue labels (ICLR, NeurIPS, ICML, ACL, COLM) were seen on arXiv pages or in search results and were not checked against proceedings; they are left out of this file for that reason.
- The 4.4 numbers in section 1 are from partial logs and are not results.
- All cost estimates beyond "21 run-minutes per arm" are guesses until one short run is timed.
