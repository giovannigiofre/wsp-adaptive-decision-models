# Making the checker loop sample-efficient: literature and design memo

Written 2026-11-12, while 4.8d was still running. **No experiment was run for this memo and no code was changed.** Nothing here is claimed as new: **every proposal is an existing idea applied to this repo's setup.** It continues `docs/SELF_LEARNING_FROM_CHECKER.md` (called "the earlier memo" below) and reuses what that memo verified.

Marks: **[P]** stated in a paper whose page was opened for this memo (or for the earlier memo, then marked "earlier memo"); **[M]** measured in this repo (experiment id or file given); **[C]** conjecture or arithmetic of mine, not checked. Section 6 lists what could not be verified.

Signal codes: **F** = full-grid checker only (valid / not valid). **R** = the rules applied to cells or partial grids (conflict mask, candidate elimination). Label-free, but it is rule knowledge beyond the one bit and must be declared. **L** = solution labels.

The question: from random weights, with puzzles and a full-grid checker only, how to reach a given solve rate with the fewest training steps, rollouts and checker calls.

## Short answer

1. **Labels are no longer the scarce resource [M].** After 10 rounds the buffer holds a valid grid for 97,970 of 100,002 train puzzles (94% of hard) from 159,845 rollouts (`hindsight_long.csv`, rounds 1 to 10). At equal steps the loop is 0.03 below the supervised model (0.698 against 0.731, 4.8b). No better label source can gain more than that gap at 16 sampler steps.
2. **Training steps are the cost, and most of the lost steps are before the first harvest [M].** On the hard band the givens loss lags the supervised loss by about a factor of 3 in steps (0.18 against 0.43 at step 2,000; 0.32 against 0.60 at step 5,000; `a1.csv`, `sft.csv`). The loop waits 9,546 steps before it uses the checker at all, although the givens model already solves 0.99 of easy and 0.68 of medium val puzzles at step 1,000.
3. **So the first experiment is a schedule change, not a new objective:** start the harvest at step 1,000 instead of 9,546 (E1 below).
4. **What can go beyond the supervised curve** is either something that would also speed up supervised training (sampling priority, augmentation, architecture) or the checker used at inference (retry and search). The second is a different quantity and must be reported as "model plus checker".
5. **Per-cell signal:** the rule-conflict mask is too weak as measured (recall 0.14 to 0.25, 4.5b). One label-free variant is untested and cheap to measure: dead-end detection by candidate elimination on the partial grid (E3). Worth a measurement, not a training run.

## 1. Where the cost is now [M unless marked]

- **Steps.** 4.8a: 9,546 steps (60 min). 4.8b: 10,000 more (about 100 min per arm). About 0.41 s per training step (4.1).
- **Rollouts and checker calls.** 159,845 in the 10 rounds, of which 100,002 in round 1. At 16 passes each that is 2.56M inference passes, the same count as the 2.56M training passes of 10,000 steps at batch 256. In wall-clock the harvest plus evaluation is about 30% of the 5,968 s of the seed-0 hindsight run **[C, from the elapsed column]**.
- **Yield.** 61% of rollouts accepted. From round 6 on, under 1,000 new grids per round.
- **Ceiling.** With all labels the model reaches 0.805 hard at 16 steps after 43,500 steps (4.1). Its val loss is below its train loss through the whole run (`sft.csv`), so it is not short of data.
- **4.8d, unfinished log read on 2026-11-12, not a result:** rounds 20 to 30 move between 0.73 and 0.78 on hard at 16 steps with the buffer at 0.985 of hard train puzzles.

Reading **[C]**: rollouts and checker calls are cheap and nearly exhausted. An idea helps only if it (a) brings labels in earlier, (b) makes each training step count more, or (c) spends forward passes at inference.

## 2. What "efficiency" should mean here

Three metrics, all to a fixed target on the hard val band at 16 sampler steps, greedy, no checker at inference:

- **E-steps:** training steps until the mean of three consecutive evaluations is at least 0.65, and 0.75. One evaluation moves by about ±0.02 between rounds **[M, 4.8b]**, so a single crossing is not a result.
- **E-passes:** total model forward passes to the same point, training passes (batch × steps) and rollout passes (rollouts × sampler steps) reported as two numbers, plus the count of checker calls. Rule calls on partial grids are counted separately from full-grid calls.
- **E-clock:** wall-clock, secondary. Outside load made timings unclean in 4.8b **[M]**.

Baselines, no rerun: supervised (`sft.csv`), givens-only control, harvest and hindsight (three seeds, 4.8b and 4.8c). Current first crossings of 0.65 **[M, single evaluations, 1,000-step grid]**: supervised by step 7,000; hindsight 13,546; harvest 14,546; givens 19,546. For 0.75: supervised about 18,000; hindsight first at 30,546 and not sustained in the unfinished 4.8d log.

Two rules for the comparison. (i) Report 4 and 81 sampler steps beside 16, and all three bands. (ii) Resolution: 334 hard val puzzles give about ±0.05; differences of 0.03 cannot be read. The generator makes puzzles in seconds (4.0 **[M]**), so a fresh hard evaluation set of about 3,000 puzzles (about ±0.017 **[C]**) is the cheapest improvement to every experiment below.

## 3. Ideas from the literature, weighed against the measurements

Costs assume 0.41 s per step and 2.5 min per 20,000 rollouts at 16 steps **[M, earlier memo]**.

### 3.1 Search, then distil

| idea | what it is | signal | why it could help here | cost | what makes it fail |
|---|---|---|---|---|---|
| Expert iteration, AlphaZero (earlier memo **[P]**); Gumbel AlphaZero (Danihelka et al., ICLR 2022, *search summary only*); Ranked Reward, Laterre et al. 2018 **[P]** | A slow search produces better targets than the raw policy; the network imitates them. Ranked Reward adapts self-play to single-player problems. | F, or R if the search prunes | The loop already has a weak expert: the same model with more passes. A stronger one raises yield per round. | Tree search with a value net: too heavy and no value signal. Best-of-K: K × 16 passes per puzzle. | Yield is already 94 to 98%; more labels are bounded by the 0.03 gap. |
| Self-improvement for neural CO, Pirnay and Grimm, TMLR 2024 **[P]** | Sample several solutions per instance, keep the best, train by imitation; no expert data, no policy gradient. | F | The published form of the harvest loop, with sampling without replacement to get distinct attempts. | As harvest. | Greedy harvest tries each unsolved puzzle once per round; whether distinct attempts add yield is unmeasured. |
| Cross-entropy method (de Boer et al. 2005, *not opened*) | Sample, keep the elite, refit by maximum likelihood. | F | With a binary reward the elite is "valid", so CEM is the harvest loop **[C]**. Nothing to add. | none | n/a |
| Checker-guided inference: MDM-VGB, Jeon, Vuong, Tao 2026 **[P]**; FK steering, Singhal et al. 2025 **[P]**; Ma et al. 2025 **[P]**; ReMDM (earlier memo) | VGB adds verifier-guided remasking to the unmasking walk and reports 100% on Sudoku for its momentum variant against 99.3% best-of-N at budget 16; its Sudoku process verifier is "no constraint yet violated", which the authors note can be far from the true value. FK steering resamples particles by intermediate reward. | R (partial-grid check) | Raises solve rate at a fixed training budget for extra passes; rollback also saves the passes of doomed rollouts. | No training; passes only. | 4.5b: remasking flagged cells gains nothing because flags cover 14 to 25% of wrong cells **[M]**. Needs the dead-end measurement of E3 first. |

### 3.2 Imitating own successes, replay, relabelling

| idea | what it is | signal | why here | cost | failure |
|---|---|---|---|---|---|
| RWR, AWR, SIL, RAFT (earlier memo **[P]**) | Train only on own successful rollouts. | F | Already the loop; pull-up works, push-down does not (4.4, 4.6c **[M]**). | none | n/a |
| Hindsight relabelling, HER, GCSL (earlier memo **[P]**) | Reuse an outcome as the label of another task. | F | Already an arm: +0.069 at 81 steps, nothing measurable at 4 or 16 (4.8c **[M]**). | none | n/a |
| Prioritized Experience Replay, Schaul et al., ICLR 2016 **[P]**; Prioritized Level Replay, Jiang et al. 2020 **[P]** | Replay items in proportion to their learning potential instead of uniformly; PLR does it per level and gets an emergent curriculum. | F | Buffer rows are drawn uniformly, so two thirds of the label gradient goes to easy and medium puzzles that are at 0.98 to 1.00 **[M]**. Shifting it to the hard band costs nothing. | zero extra | Easy or medium falls; gain inside noise. 4.6d found that *which* hard puzzles carry the loss does not matter at 4 steps **[M]**. |
| Off-policy reuse: DiFFPO, Zhao et al. 2025 **[P]** | Off-policy RL on a surrogate policy with importance correction for masked dLLMs. | F | The buffer already reuses every valid grid forever with a plain likelihood loss; no correction is needed because a valid grid is the label **[M, 4.4b]**. | none | n/a |

### 3.3 Curriculum

| idea | what it is | signal | why here | cost | failure |
|---|---|---|---|---|---|
| Self-paced learning, Kumar, Packer, Koller, NIPS 2010 **[P]**; reverse curriculum, Polu et al. (earlier memo **[P]**) | Train on easy samples first and widen the set. | F | The harvest is already self-paced: round 1 accepts 0.997 / 0.929 / 0.483 by band **[M]**. What is *not* self-paced is the 9,546-step wait before round 1. | Earlier start costs more rollouts, fewer steps. | Early buffers are easy-only and take 75% of the batch. |
| Fewest blanks first, as an explicit schedule | Order puzzles by k. | F | Not needed: the supervised model has easy at 0.98 by step 500 and the givens model by step 900 **[M]**. There is no cold start to bridge. | none | n/a |
| Simplified variants: Ren and Liu 2026 **[P, abstract; the variant detail is from a search summary]** | Augment HRM training with easier versions of each puzzle; with two other changes, Sudoku-Extreme 54.5% to 96.9%. | L there | Hindsight masks and the givens loss already produce such variants. | none | n/a |

### 3.4 Symmetries

| idea | what it is | signal | why here | cost | failure |
|---|---|---|---|---|---|
| Augmentation in training: HRM, Wang et al. 2025 **[P]** (1,000 puzzles, band and digit permutations); TRM, Jolicoeur-Martineau 2025 **[P]** (1,000 puzzles × 1,000 rule-preserving shuffles; 87.4% against 55.0% on Sudoku-Extreme) | Each solved puzzle becomes many valid ones at zero checker cost. | F plus declared rule knowledge | Those papers use it because they have 1,000 puzzles. Here there are 100,002 distinct grids, each already drawn under a random symmetry (4.0 **[M]**), and no sign of overfitting (val loss below train loss **[M]**). Expected gain near zero **[C]**. | zero | Null result, predicted. |
| Equivariant architecture: SE-RRM, Freinschlag et al. 2026 **[P]** | Build digit-permutation equivariance into the layers instead of augmenting. | n/a | A model change; outside "one factor against the existing loop". | new model | n/a |
| Symmetric copies at inference: POMO, Kwon et al., NeurIPS 2020 **[P]**; Sym-NCO, Kim et al., NeurIPS 2022 **[P]** | Run the model on several equivalent views of one instance and keep the best. | F | Gives K different greedy attempts per puzzle, checked by the validity bit: best-of-K without sampling noise. | K × 16 passes; minutes on val | The views fail on the same puzzles (errors tied to the puzzle, not the view). |

### 3.5 Losses computed from the rules

| idea | what it is | signal | why here | cost | failure |
|---|---|---|---|---|---|
| Semantic loss (earlier memo **[P]**); pseudo-semantic loss, Ahmed, Chang, Van den Broeck 2023 **[P]**; neurosymbolic diffusion, van Krieken et al., NeurIPS 2025 **[P]** | Maximise the probability that the output satisfies the constraint; the 2023 paper approximates it around a sample and tests on Sudoku. | R | A gradient on blank cells of unsolved puzzles, which the givens loss never touches. | Exact form intractable on 9×9; local forms cheap. | See next row. |
| Differentiable penalties: ConsFormer, Xu et al. 2025 **[P, full text]**; RUN-CSP, Toenshoff et al. 2019 **[P, abstract]**; ANYCSP, Tönshoff et al. 2022 **[P]**; Erdős goes neural, Karalias and Loukas 2020 **[P]**; DRNets, Chen et al. 2019 **[P]**; IC/DC, Hong et al. 2024 **[P]** | Train from constraint violations of the model's own soft outputs, no labels. ConsFormer: 100% on in-distribution Sudoku, 65.9 to 77.7% on harder puzzles, 6 to 10 GPU hours. | R | Label-free and dense. | Negligible per step. | The unit-sum penalty is zero for uniform marginals, and also for the true marginals **[C]**, so it adds little once cross-entropy works; ConsFormer needs near-one-hot outputs for it to bite **[P]**. The model learns peer elimination within 500 to 900 steps anyway **[M]**. |

### 3.6 Objectives that sample in proportion to reward, and RL for masked diffusion since wd1

- **GFlowNets, trajectory balance** (TB, NeurIPS 2022 **[P]**; "Let the Flows Tell", NeurIPS 2023 **[P]**): learn to sample objects in proportion to reward, with better credit assignment over long trajectories. Signal F. With one valid grid per puzzle the target is a point mass, so diversity, the method's strength, has no use here, and the loss on zero-reward trajectories is a push-down, which hurt in 4.4 and 4.6c **[M]**. Drop **[C]**.
- **Since wd1:** DMPO (Zhu et al. 2025) and Discrete Tilt Matching (Chen et al. 2026) both train with a weighted cross-entropy toward a reward-tilted target, without sequence likelihoods **[P]**; with a binary reward that is the harvest loss up to weights **[C]**. A continuous-time framework (Zhang et al. 2026) allows intermediate rewards along the denoising path **[P]**. Self-Aware Scheduling (Xu et al. 2026) trains only the unmasking order with GRPO on a frozen denoiser: Sudoku 82.0% to 91.8% on a 1B model **[P]**. DARE (Yang et al. 2026) is a framework, not a method **[P]**. All fine-tune pretrained models of 1B to 8B parameters; none starts from random weights. Nothing here suggests an objective better than plain imitation of valid grids for this task.
- **Iterative-reasoning models on Sudoku:** Drozdova et al. 2026 report 99.90% on Sudoku-Extreme with a looped denoiser and conclude that diffusion's contribution is a training curriculum **[P, abstract]**; Drozdova 2026 uses continuous diffusion as a stochastic search that stops when the constraints hold and finds it sample-inefficient **[P]**. Sudoku-Bench (Seely et al. 2025) tests LLMs on Sudoku variants and is not about training **[P]**. These say the ceiling of 0.805 is a property of this model and budget, not of the task; changing the architecture is a different project.

## 4. The per-cell signal, weighed

The oracle version (which cells differ from the solution) is label use (earlier memo). The label-free version marks cells that share a digit with a peer. What it can tell: that the grid is wrong, and that at least one cell of each flagged pair is wrong. What it cannot tell: which one, and anything about wrong cells that clash with nobody. Measured on failed final grids, it finds 25% of the wrong cells at 4 steps and 14% at 16, and 61 to 66% of flagged cells are wrong; remasking the flagged cells does not beat the plain sampler; fill-then-fix needs recall near 0.9 (4.5b, 4.6b **[M]**). The reason is structural: a failed grid is one or two early wrong cells followed by 14 to 20 cells that fit them (4.4b **[M]**), so a model trained to be consistent hides its error until the last cells, and the visible clash is usually far from the cause. As a training signal the mask could only act as a push-down on flagged cells, the kind of term that lost in every arm of 4.4 and 4.6c **[M]**. As a logit mask it blocks direct clashes, which are rare (8% of first errors at 4 steps, 4.4b **[M]**). One variant is untested: candidate elimination on the *partial* grid during sampling. A masked cell with no candidate left, or a unit with no place for a digit, proves that a committed cell is wrong before any clash is visible. It still does not say which cell; choosing which commit to undo is backtracking, and with full propagation and backtracking the sampler becomes the repo's own solver (the rules alone solve 67.9% of hard puzzles by singles, 4.0 **[M]**). My view: it is worth one measurement of how soon after the first error a dead end becomes detectable (E3), because that number decides whether cheap rollback is possible. It is not worth a training run before that, and any result that uses it belongs in a "model plus rules" column, not beside the checker-only results.

## 5. Ranked shortlist

Each isolates one factor against the existing hindsight loop. Predictions are **[C]**.

**E1. Start the harvest at step 1,000 instead of 9,546 (run first).** Hindsight arm unchanged, from the 4.8a state at step 1,000 (`step_001000.pt` is in the cache; if it lacks the optimiser state, 1,000 steps cost 7 min), 13 rounds, about 110 min. Compare with the 4.8b and 4.8c runs at equal total steps. *Prediction:* hard at 16 steps at least 0.58 at total step 9,546 (givens-only: 0.485; supervised: 0.67), and E-steps for 0.65 between 9,000 and 11,000 (now 13,546). E-passes for rollouts rise, perhaps to 250,000 rollouts. *Negative result:* hard at step 13,546 within ±0.03 of 0.653, meaning the givens phase was not lost time; or slower, because early batches are 75% easy grids.

**E2. Priority to the hard band when drawing buffer rows.** From the 4.8b state at round 10, 10 rounds with 60% of buffer rows from hard puzzles; the control is rounds 11 to 20 of 4.8d, already run. About 80 min. *Prediction:* +0.02 to +0.04 on hard at 16 steps, medium not below 0.975. *Negative result:* difference inside ±0.03, or a band falls. Needs the larger evaluation set of section 2 to be readable.

**E3. Dead-end detection, no training.** On the 4.8b hindsight model, record for each failed 81-step rollout the commit index of the first wrong cell (val solutions, measurement only) and of the first rule-detectable dead end (empty candidate set, missing place, or clash). Minutes. *Prediction:* the median lag is over 10 commits, too late for cheap rollback. *If instead* the lag is 3 commits or fewer in most failed grids, rollback-and-retry becomes the next experiment, in a "model plus rules" column.

**E4. Best-of-K over symmetric copies, no training.** K = 8 equivalent views of each hard val puzzle, 16 steps each, accept any valid grid; compare with one 81-step rollout and with K sampled rollouts at about the same number of passes. Minutes. *Prediction:* +0.05 to +0.10 over a single 16-step rollout, about equal to sampled best-of-K. *Negative result:* under +0.03, meaning failures belong to the puzzle and not to the view.

**Why E1 first:** it attacks the largest measured loss (the factor of 3 between the givens and supervised curves over the first 9,546 steps), it changes one number, it fits one launch, and E-steps to 0.65 already separates the arms. E3 and E4 are cheaper and can run in a gap, but they inform inference, not training efficiency.

**Dropped:** GFlowNets and trajectory balance; any push-down policy gradient; exact semantic loss; tree search with a value network; an explicit fewest-blanks curriculum; CEM as a separate method. **Deferred with a predicted null:** symmetry augmentation in training (only interesting if the puzzle set is cut to about 1,000), the unit-sum constraint penalty, architecture changes.

## 6. Not verified

- **Not opened:** de Boer, Kroese, Mannor, Rubinstein, cross-entropy tutorial (PDF unreadable; the description of CEM is from memory). RUN-CSP full text (404; "unsupervised" is from the abstract, the loss is not).
- **Search summaries only:** Gumbel AlphaZero (Danihelka et al., ICLR 2022); Combinatorial Adjoint Matching (ICML 2026, not used above); the "simplified variants" detail of Ren and Liu; TRM's 5M / 7M parameter split.
- **Author name:** the arXiv pages of 2201.13259 and 2305.17010 were returned with an author "Esmeralda S. Whitammer" where I expected Nikolay Malkin. The reference list gives the arXiv ids and omits that name; check before quoting.
- **Abstract pages only** except ConsFormer, HRM, TRM and MDM-VGB (HTML full text, read in part through a summarising tool; MDM-VGB only the first half). Venues as shown on arXiv.
- **4.8d numbers** are from an unfinished log and may change. Totals in section 1 (159,845 rollouts, 97,970 grids, 30% wall-clock share) are my sums over the log.
- **[C] throughout**, in particular: that an early harvest shortens E-steps; that the true marginals satisfy the unit-sum penalty; that weighted-cross-entropy RL reduces to the harvest loss here; all predictions of section 5.
- "Nothing found" statements rest on eight web searches, not a systematic review.

## References

Opened for this memo:
- Jeon, Vuong, Tao, 2026. VGB for Masked Diffusion Model. https://arxiv.org/abs/2606.28301
- Jolicoeur-Martineau, 2025. Less is More: Recursive Reasoning with Tiny Networks (TRM). https://arxiv.org/abs/2510.04871
- Wang et al., 2025. Hierarchical Reasoning Model. https://arxiv.org/abs/2506.21734
- Ren, Liu, 2026. Are Your Reasoning Models Reasoning or Guessing? https://arxiv.org/abs/2601.10679
- Freinschlag, Bertram, Kobler, Mayr, Klambauer, 2026. Symbol-Equivariant Recurrent Reasoning Models. https://arxiv.org/abs/2603.02193
- Seely, Imajuku, Zhao, Cetin, Jones, 2025. Sudoku-Bench. https://arxiv.org/abs/2505.16135
- Drozdova, 2026. Can Continuous-Time Diffusion Models Generate and Solve Globally Constrained Discrete Problems? https://arxiv.org/abs/2601.20363
- Drozdova et al., 2026. Diffusion as a Training Curriculum for Timestep-Free Iterative Reasoning (NeurIPS 2026 workshop). https://arxiv.org/abs/2609.01449
- Trajectory balance: Improved credit assignment in GFlowNets, NeurIPS 2022. https://arxiv.org/abs/2201.13259
- Zhang, Dai, et al., Let the Flows Tell, NeurIPS 2023. https://arxiv.org/abs/2305.17010
- Schaul, Quan, Antonoglou, Silver, Prioritized Experience Replay, ICLR 2016. https://arxiv.org/abs/1511.05952
- Jiang, Grefenstette, Rocktäschel, 2020. Prioritized Level Replay. https://arxiv.org/abs/2010.03934
- Kumar, Packer, Koller, Self-Paced Learning for Latent Variable Models, NIPS 2010. https://papers.nips.cc/paper_files/paper/2010/hash/e57c6b956a6521b28495f2886ca0977a-Abstract.html
- Pirnay, Grimm, Self-Improvement for Neural Combinatorial Optimization, TMLR 2024. https://arxiv.org/abs/2403.15180
- Laterre et al., Ranked Reward, NeurIPS 2018 Deep RL workshop. https://arxiv.org/abs/1807.01672
- Kwon et al., POMO, NeurIPS 2020. https://arxiv.org/abs/2010.16011
- Kim, Park, Park, Sym-NCO, NeurIPS 2022. https://arxiv.org/abs/2205.13209
- Xu, Li, Sanner, Khalil, 2025. ConsFormer. https://arxiv.org/abs/2502.15794
- Toenshoff, Ritzert, Wolf, Grohe, 2019. Graph Neural Networks for Maximum Constraint Satisfaction (RUN-CSP). https://arxiv.org/abs/1909.08387
- Tönshoff, Kisin, Lindner, Grohe, 2022. One Model, Any CSP (ANYCSP). https://arxiv.org/abs/2208.10227
- Karalias, Loukas, 2020. Erdős Goes Neural. https://arxiv.org/abs/2006.10643
- Chen, Bai, Zhao, Ament, Gregoire, Gomes, 2019. Deep Reasoning Networks. https://arxiv.org/abs/1906.00855
- Hong et al., 2024. Unsupervised Training of Diffusion Models for Feasible Solution Generation in NCO (IC/DC). https://arxiv.org/abs/2411.00003
- Ahmed, Chang, Van den Broeck, 2023. A Pseudo-Semantic Loss for Autoregressive Models with Logical Constraints. https://arxiv.org/abs/2312.03905
- van Krieken, Minervini, Ponti, Vergari, Neurosymbolic Diffusion Models, NeurIPS 2025. https://arxiv.org/abs/2505.13138
- Yang, Ishay, Lee, Recurrent Transformer for CSPs, ICLR 2023 (abstract; reopened). https://arxiv.org/abs/2307.04895
- Singhal et al., 2025. A General Framework for Inference-time Scaling and Steering of Diffusion Models. https://arxiv.org/abs/2501.06848
- Ma et al., 2025. Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps. https://arxiv.org/abs/2501.09732
- Zhu et al., 2025. Distribution Matching Policy Optimization (DMPO). https://arxiv.org/abs/2510.08233
- Chen, Wang, Potaptchik, Kim, Albergo, 2026. Discrete Tilt Matching. https://arxiv.org/abs/2604.18739
- Zhao, Liang, Tang, Yao, Kallus, 2025. DiFFPO. https://arxiv.org/abs/2510.02212
- Zhang, Sheng, Yao, Tang, 2026. A Continuous-Time RL Framework for Fine-Tuning Discrete Diffusion Models. https://arxiv.org/abs/2607.14522
- Xu, Liu, Agrawal, Chen, Huang, 2026. Scheduling Thoughts (Self-Aware Scheduling). https://arxiv.org/abs/2606.23567
- Yang et al., 2026. DARE. https://arxiv.org/abs/2604.04215

Found through search, not opened as a paper:
- Danihelka, Guez, Schrittwieser, Silver, Policy Improvement by Planning with Gumbel, ICLR 2022. https://iclr.cc/virtual/2022/poster/6418
- de Boer, Kroese, Mannor, Rubinstein, A Tutorial on the Cross-Entropy Method, Annals of Operations Research, 2005 (citation from memory). https://link.springer.com/article/10.1007/s10479-005-5724-z

Reused from `docs/SELF_LEARNING_FROM_CHECKER.md` (see its verification notes): AlphaZero 1712.01815, Expert Iteration 1705.08439, Polu et al. 2202.01344, Self-Imitation Learning 1806.05635, RWR, AWR 1910.00177, RAFT 2504.11343, HER 1707.01495, GCSL 1912.06088, semantic loss 1711.11157, ReMDM 2503.00307, d1 2504.12216, wd1 2507.08838.
