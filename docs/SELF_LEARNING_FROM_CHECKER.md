# Can the model teach itself Sudoku from a checker? Literature and design memo

Written 2026-11-11, after 4.6d. **No experiment was run for this memo and no code was changed.** This is a reading and design note for a learning project, not a paper: **every proposal below is an existing idea applied to this repo's setup**, and nothing here is claimed as new. It continues `docs/RL_DLLM_SURVEY.md` and `docs/RL_DLLM_DIRECTIONS.md` and reuses what they verified.

Marks: **[P]** stated in a paper whose arXiv abstract page (or the page named) was opened for this memo or for the two earlier memos; **[M]** measured in this repo (experiment id given); **[C]** conjecture or arithmetic of mine, not checked. Section 8 lists what could not be verified.

The question: given many puzzles and zero solutions, plus (a) a checker that says whether a completed grid is right and (b) a tool that says exactly which cells are wrong, can the 1.86M-parameter masked diffusion model teach itself Sudoku, how long would it take, and what is a well-designed system for it?

## Short answer

1. **Yes, very probably, and for an unglamorous reason [C].** A valid completed grid of a unique-solution puzzle *is* its label (4.4b: every valid rollout equals the stored solution **[M]**). Once the model solves any puzzles at all, the checker turns its own outputs into labelled data, and re-masking those grids gives training data for harder puzzles. After the cold start, "learning alone" is supervised training on self-found grids.
2. **The whole difficulty is the cold start.** With a validity bit only, an untrained model on a puzzle with k blanks is right with probability 9^−k: non-zero in practice only for k ≤ 3 **[C]**. The dataset starts at k = 30.
3. **There is a cold start that needs no tool at all [C]:** the givens of a puzzle are correct labels for their own cells. Hide some givens and train the model to predict them (ordinary masked training). This is the recommended first experiment.
4. **Tool (b) is not "zero solutions".** A wrong-cell oracle for a unique-solution puzzle reveals the solution in at most 8 queries per cell. Used on the denoiser it is labels by another name, and 4.6d says more labels do not help at 4 steps **[M]**. Its one honest use is to train an *error detector* that is then used with no oracle at inference.
5. **Time [C]:** easy band alone in about 1 to 2 hours; level with the 5 h supervised model in roughly 6 to 11 hours of training, in launches under 2 h. Wide uncertainty; section 6.

## 1. What the two tools give

### 1.1 Information content

A subtlety first: for a puzzle with a unique solution, the solution is a function of the puzzle. A label therefore adds zero Shannon information given the puzzle and unlimited computation. What a label saves is *computation* (search). The numbers below are description lengths, the right measure for a learner that cannot search.

| signal | needs the solution? | nominal size | what the learner gets |
|---|---|---|---|
| full label | yes | log2 9 = 3.17 bits per blank; about 170 bits for a hard puzzle (53.5 blanks **[M, 4.0]**) | everything |
| (i) validity bit on a completed grid | **no**: computable from the rules and the givens | at most 1 bit per rollout, H(p) for valid share p | untrained model at k = 53: p = 9^−53, about 10^−49 bits. At the plateau model (p = 0.56 **[M, 4.3]**): 0.99 bits, but a valid grid is the full label, so the real yield is "the label, with probability p" |
| (ii) rule conflicts (cells that share a digit with a peer) | **no** | computable by the learner | no information beyond the rules. On failed grids it flags 25% of wrong cells at 4 steps (14% at 16), and 61 to 66% of flagged cells are wrong **[M, 4.5b]** |
| (iii) wrong-cell oracle | **yes**: "wrong" means "differs from the unique solution" | 1 bit per cell and query; H(1/9) = 0.50 bits for a uniform guess | by elimination the oracle identifies a cell in at most 8 queries, 4.9 on average for uniform guessing **[C]**: it is a label oracle at a constant-factor cost |

Two consequences. First, tool (a) is weaker than it looks at the start and stronger than it looks later: worthless when p is about 0, a free label source when p is not small. Second, tool (b) cannot be built without the solution or a solver, so a system that uses it has labels, paid for in queries. On a puzzle with several solutions "wrong cell" is not even defined.

### 1.2 What "zero solutions" can honestly mean

| variant | learner has | honest description |
|---|---|---|
| V1 | puzzles + validity bit | label-free. If the checker is code, the learner also has V2, because validity is computed from the same rules |
| V2 | puzzles + rule conflicts | label-free, but the rules plus compute already make labels: a backtracking search with the conflict check is the repo's own generator (100,002 solved grids in 93 s **[M, 4.0]**). "No solutions" then means "no solutions *stored*", and the question becomes whether the model's own sampling can replace the search |
| V3 | V1 + wrong-cell oracle during training | labels at up to 8 queries per cell. Must be reported as label use |
| V4 | V3 + oracle at inference | not a solver: the 4.5b `fix_oracle` row (0.907 at 4 passes, 1.000 at 8 **[M]**). An upper reference, never a result |

Sub-variants that matter: whether the learner may train on cells of the *givens* (free and legitimate, section 3.2); whether it may use Sudoku symmetries (digit relabelling, row and column permutations) to multiply a found grid, which is rule knowledge and must be declared; whether puzzles with small k are supplied (the supplier then holds solutions, the learner does not).

## 2. State of the art

### 2.1 Learning from a verifier, and why search is usually needed

- **AlphaZero**, Silver et al., 2017, [1712.01815](https://arxiv.org/abs/1712.01815) **[P]**: from the rules and random play to top-level chess, shogi and Go. **Expert Iteration (ExIt)**, Anthony, Tian, Barber, 2017, [1705.08439](https://arxiv.org/abs/1705.08439) **[P]**: tree search plans, a network generalises the plans; reported better than REINFORCE on Hex. The pattern: a slow *expert* (search guided by the network) produces better targets than the network alone, and the network imitates them. Without the expert, a verifier only confirms what the policy already does. Search is where improvement comes from.
- **Polu et al.**, 2022, [2202.01344](https://arxiv.org/abs/2202.01344) **[P]**: expert iteration for theorem proving; with statements of varied difficulty it "can solve a progressively harder curriculum without ground-truth proofs". The closest published analogue of our question: a verifier (the proof checker), no labels, a difficulty curriculum.
- **Absolute Zero**, Zhao et al., 2025, [2505.03335](https://arxiv.org/abs/2505.03335) **[P]**: one model proposes tasks and solves them, a code executor verifies, no external data. **Self-Verified Distillation**, Lee and Liang, 2026, [2605.26132](https://arxiv.org/abs/2605.26132) **[P]**: unlabeled questions, own solutions filtered by self-verification, train on the kept ones. Both start from a pretrained LLM, so they have no cold start.
- **NeuroSAT**, Selsam et al., 2018, [1802.03685](https://arxiv.org/abs/1802.03685) **[P]**: a network trained only to predict one bit (satisfiable or not) acquires solving ability. Relevant as evidence that a single bit per instance can carry a solver, given enough instances; it is a classifier with labels for that bit, not RL.
- **Our analogue of the expert [C]:** the same network run with more sampler steps (0.535 at 4 steps, 0.814 at 16, 0.871 at 81 on hard test **[M, 4.2]**), or with remasking. Distilling many steps into few is proposal 4b of the survey (self-distillation through time, [2410.21035](https://arxiv.org/abs/2410.21035) **[P, abstract]**).

### 2.2 Self-training, reward-weighted regression and the cold start

- **RWR**, Peters and Schaal, ICML 2007 **[P, ICML abstract page via search]**; **AWR**, [1910.00177](https://arxiv.org/abs/1910.00177) **[P]**; **Self-Imitation Learning**, Oh et al., 2018, [1806.05635](https://arxiv.org/abs/1806.05635) **[P]**: replay and imitate one's own good episodes. **RAFT**, [2504.11343](https://arxiv.org/abs/2504.11343), and **Davis and Recht**, [2510.13651](https://arxiv.org/abs/2510.13651) **[P]**: with binary rewards, training only on one's own successes is a sound algorithm (gradient ascent on the log of the success probability).
- All of these have gradient proportional to the success rate. At zero successes nothing is learned. This repo saw both sides: imitation arms learned nothing at 1 valid rollout in 700 (3.6b **[M]**) and were the best arms at 56% valid (4.4 **[M]**).
- **DAgger**, Ross et al., 2010, [1011.0686](https://arxiv.org/abs/1011.0686) **[P]**, and **scheduled sampling**, Bengio et al., 2015, [1506.03099](https://arxiv.org/abs/1506.03099) **[P]**: train on the states the learner itself visits. Background for section 2.6.

### 2.3 Curricula and hindsight relabelling

- **Curriculum learning** (Bengio et al., ICML 2009, *not opened*). **Reverse curriculum generation**, Florensa et al., 2017, [1707.05300](https://arxiv.org/abs/1707.05300) **[P]**: start near the goal state and move the start backward. On Sudoku "near the goal" is "few blanks".
- **Hindsight Experience Replay**, Andrychowicz et al., 2017, [1707.01495](https://arxiv.org/abs/1707.01495) **[P]**: relabel a failed episode with the goal it actually reached. **GCSL**, Ghosh et al., 2019, [1912.06088](https://arxiv.org/abs/1912.06088) **[P]**: goal-reaching from scratch by relabelling and plain supervised learning, no value function.
- **The analogue here [C].** Any complete valid grid the model produces is a solution of every puzzle obtained by blanking cells of it. So one valid grid, found on an easy puzzle, yields training puzzles of any difficulty with their label. Solved grids of easy and hard puzzles come from the same distribution (the band is a property of the mask, not of the grid), so grids harvested on easy puzzles are exactly the raw material the supervised run used. Caveat: a puzzle made by blanking at random may have several solutions; the diffusion loss then learns the average over them, as it does for any masked state.
- **Ambient Diffusion**, Daras et al., 2023, [2305.19256](https://arxiv.org/abs/2305.19256) **[P]**: learn a clean distribution from samples with most entries missing, by corrupting further and predicting what was hidden. The image-domain precedent for "train on the givens" (section 3.2).

### 2.4 Sudoku-specific neural work

| work | supervision | result as reported |
|---|---|---|
| Recurrent Relational Networks, Palm, Paquet, Winther, 2017, [1711.08028](https://arxiv.org/abs/1711.08028) **[P]** | solutions | 96.6% of the hardest puzzles |
| SATNet, Wang et al., 2019, [1905.12149](https://arxiv.org/abs/1905.12149) **[P]** | puzzle and solution pairs, no rules given | learns 9×9 Sudoku "solely from examples" |
| Chang et al., NeurIPS 2020, [2312.11522](https://arxiv.org/abs/2312.11522) **[P, search summary]** | | visual-Sudoku SATNet fails (0%) without intermediate labels |
| Recurrent Transformer, Yang, Ishay, Lee, 2023, [2307.04895](https://arxiv.org/abs/2307.04895) **[P]** | labels, plus constraint knowledge for semi-supervised learning | abstract only; no Sudoku number seen |
| Semantic loss, Xu et al., 2017, [1711.11157](https://arxiv.org/abs/1711.11157) **[P]** | a loss on how well outputs satisfy logical constraints | general method |
| **ConsFormer**, Xu et al., 2025, [2502.15794](https://arxiv.org/abs/2502.15794) **[P]** | **none**: differentiable approximations of the constraints, no labels, no RL | Sudoku among the tasks; the published label-free route closest to V2 |
| DDReasoner, Zhang et al., 2025, [2508.16524](https://arxiv.org/abs/2508.16524) **[P]** | stage 1 "basic reasoning", then PPO on a diffusion reasoner with a rule-based reward | Sudoku among the tasks; the closest "diffusion + rule reward", not from scratch |
| Neuro-Symbolic Sudoku Solver, Hathidara and Pandey, 2023, [2307.00653](https://arxiv.org/abs/2307.00653) **[P]** | RL on a Neural Logic Machine | 100% for 3 to 10 empty cells: an RL solver that only reaches tiny k |
| Ye et al., "Beyond Autoregression", 2024, [2410.14157](https://arxiv.org/abs/2410.14157) **[P]** | solutions | discrete diffusion 100% against 20.7% autoregressive; "subgoal imbalance" as the lens |
| Kim et al., "Train for the Worst, Plan for the Best", 2025, [2502.06768](https://arxiv.org/abs/2502.06768) **[P]** | solutions | adaptive order: under 7% to about 90% |
| Mohamud et al., [2606.00295](https://arxiv.org/abs/2606.00295) **[P, survey]** | solutions | learned order 93.75, oracle order 100 |
| IRED, Du, Mao, Tenenbaum, 2024, [2406.11179](https://arxiv.org/abs/2406.11179) **[P]** | solutions | energy minimisation with more steps for harder puzzles |
| HRM, Wang et al., 2025, [2506.21734](https://arxiv.org/abs/2506.21734) **[P]** | 1,000 labelled samples | 27M parameters, "nearly perfect" on complex Sudoku |
| TRM, Jolicoeur-Martineau, 2025, [2510.04871](https://arxiv.org/abs/2510.04871) **[P for title and 7M parameters]** | labels | 87.4% on Sudoku-Extreme against 55.0% for HRM: *secondary sources only* |
| Self-Aware Markov Models, Kornhardt et al., 2026, [2603.16661](https://arxiv.org/abs/2603.16661) **[P]** | labels | trained on its own outputs, can remask; 95% on Sudoku-Extreme |

Reading: every strong small-model Sudoku result found uses solutions. Label-free work exists through differentiable constraints (ConsFormer, semantic loss), not through a binary checker with RL. A search for "RL from a rule checker only, from scratch, with a curriculum, on 9×9" found nothing that fits all four; CLR-DRNets (Bai, Chen, Gomes, CP 2021) was returned as "trained without label supervision, with a curriculum and restarts" but could not be opened.

### 2.5 RL for masked and discrete diffusion

Verified in the two earlier memos and reused: d1 ([2504.12216](https://arxiv.org/abs/2504.12216)), wd1 ([2507.08838](https://arxiv.org/abs/2507.08838)), SPG ([2510.09541](https://arxiv.org/abs/2510.09541)), GDSD ([2605.29398](https://arxiv.org/abs/2605.29398)), MDPO ([2508.13148](https://arxiv.org/abs/2508.13148)), DDPO ([2305.13301](https://arxiv.org/abs/2305.13301)), SEPO ([2502.01384](https://arxiv.org/abs/2502.01384)), DDPP ([2410.08134](https://arxiv.org/abs/2410.08134)), RTB ([2405.20971](https://arxiv.org/abs/2405.20971)), IGPO ([2509.10396](https://arxiv.org/abs/2509.10396)), CanvasAnneal ([2609.13060](https://arxiv.org/abs/2609.13060)). New here: GFlowNets, Bengio et al., 2021, [2106.04399](https://arxiv.org/abs/2106.04399) **[P]** (sample in proportion to reward; with one valid grid per puzzle this collapses to "find the solution").

All of them **fine-tune a pretrained model** (mostly LLaDA-8B) **[P]**. None starts from random weights, so none faces the cold start. IGPO and CanvasAnneal reveal parts of a ground-truth trace to make failing groups succeed: a curriculum that uses labels.

### 2.6 Remasking, self-correction, and training with error locations

- **Samplers that revise:** ReMDM ([2503.00307](https://arxiv.org/abs/2503.00307)), P2 ([2502.03540](https://arxiv.org/abs/2502.03540)), MDPO's running-confidence remasking **[P, earlier memos]**. Predictor-corrector sampling for discrete diffusion (Campbell et al., 2022, [2205.14987](https://arxiv.org/abs/2205.14987)) and corrector sampling in Discrete Flow Matching (Gat et al., 2024, [2407.15595](https://arxiv.org/abs/2407.15595)): both papers exist **[P]**, but their abstracts do not mention correctors; that attribution is from memory. Informed Correctors ([2407.21243](https://arxiv.org/abs/2407.21243)) **[P, earlier memo]**. GIDD, von Rütte et al., 2025, [2503.04482](https://arxiv.org/abs/2503.04482) **[P]**: mixing uniform noise into masking gives the model "the ability to correct its own mistakes".
- **Learned detectors:** Token-Critic ([2209.04439](https://arxiv.org/abs/2209.04439)), RemeDi ([2509.23653](https://arxiv.org/abs/2509.23653), SFT then RL, 8B model) **[P, earlier memos]**. **PRISM**, Kim et al., [2510.01384](https://arxiv.org/abs/2510.01384) **[P]**: per-token quality scores from a self-correction loss, in the same forward pass, no RL and no verifier, Sudoku among its tasks. **DDPD**, Liu et al., 2024, [2410.06264](https://arxiv.org/abs/2410.06264) **[P]**: a separate planner picks which positions are corrupted and should be denoised next. **SCOPE / D3IM**, Yu et al., 2026, [2606.01026](https://arxiv.org/abs/2606.01026) **[P]**: names "preservation bias" (the model repeats its own wrong committed tokens) and post-trains on simulated sampler states. That is the published name for what 4.5b measured: no score from the model finds its own wrong cells.
- **Training with error-location supervision:** **ELECTRA**, Clark et al., 2020, [2003.10555](https://arxiv.org/abs/2003.10555) **[P]**: a small generator replaces tokens, a discriminator learns which were replaced. **Levenshtein Transformer**, Gu, Wang, Zhao, 2019, [1905.11006](https://arxiv.org/abs/1905.11006) **[P]**, and **Edit Flows**, Havasi et al., 2025, [2506.09018](https://arxiv.org/abs/2506.09018) **[P]**: edit operations as the model's actions. **Mask-Predict**, Ghazvininejad et al., 2019, [1904.09324](https://arxiv.org/abs/1904.09324) **[P]**: regenerate low-confidence tokens. **SUNDAE**, Savinov et al., 2021, [2112.06749](https://arxiv.org/abs/2112.06749) **[P]**: unroll the denoiser on its own samples and train it to fix them.
- **RL on a remasking policy:** RemeDi is the one found. A search for a 2026 Sudoku paper that trains a remasking policy by RL found none; the nearest (2603.16661) is supervised and says so.

## 3. Can they learn on their own?

### 3.1 What is known, what is conjecture

Known: (1) verifier-only learning works when improvement comes from search or from a base that already succeeds sometimes (AlphaZero, ExIt, Polu et al., Absolute Zero) **[P]**. (2) Imitating one's own successes is sound under binary rewards (RAFT, Davis and Recht) **[P]** and here gains +0.05 to +0.07 at 4 steps, the same as labels (4.4, 4.6d) **[M]**. (3) Constraint knowledge alone can train a Sudoku solver without labels (ConsFormer) **[P]**. (4) Valid rollouts are the labels (4.4b) **[M]**.

Conjecture: that this particular small model gets from random weights to the supervised level with a binary checker; how long that takes; whether the model's own sampler is enough of an "expert" for the hard band. No published instance of exactly this was found.

One expert caveat: this is a statement about Sudoku, not about "these models". Sudoku has a perfect cheap verifier, a unique answer, and a difficulty knob. Where a valid output is not the unique label (text, code with partial tests), the label-manufacturing argument weakens.

### 3.2 Cold-start arithmetic

Assumptions: untrained model = uniform over 9 digits per blank; one RL step = 32 puzzles × 8 rollouts = 256 rollouts, about 4 s **[M, 4.4]**.

| blanks k | P(valid) = 9^−k | valid rollouts per step | per hour (900 steps) |
|---|---|---|---|
| 1 | 0.11 | 28 | 25,600 |
| 2 | 0.012 | 3.2 | 2,840 |
| 3 | 1.4 × 10^−3 | 0.35 | 316 |
| 4 | 1.5 × 10^−4 | 0.04 | 35 |
| 5 | 1.7 × 10^−5 | 0.004 | 4 |
| 30 (easiest stored) | 2.4 × 10^−29 | 0 | 0 |

A complete random grid is valid with probability 6.7 × 10^21 / 9^81, about 3 × 10^−56. So on the stored bands (k 30 to 56) a validity bit teaches an untrained model nothing, ever **[C]**.

For a partly trained model with per-cell accuracy a at commit time, P(valid) is about a^k (the first error decides, 4.4b **[M]**). A valid share of 0.05 on an easy puzzle (k = 35) needs a = 0.92; a share of 0.5 needs a = 0.980 at k = 35 and 0.987 at k = 53.5. The supervised run crossed this fast: easy val solve 0.49 at step 400 and 0.98 at step 500, under 4 minutes; hard 0.08 at step 500 and 0.33 at step 1,500 (`results/phase4/sft/sft.csv` **[M]**).

Three ways to make the reward non-zero, cheapest first:

1. **Train on the givens [C].** A puzzle with k blanks has 81 − k cells whose digit is known. Keep the blanks masked, mask a random share of the givens, predict them. No tool, no solution; the learner sees puzzles with k + m masked cells and labels for m of them. The 100,002 puzzles hold about 3.6M such cells (mean 45 blanks **[M, 4.0]**). What it does not cover: states with fewer than k masked cells (late sampler steps), and any bias from the uniqueness filter on which cells are givens. Precedent: masked-LM training; Ambient Diffusion.
2. **A rule mask on the logits (V2) [C].** Forbid digits that conflict with a visible peer. With many sampler steps and confidence order, even a uniform model then performs naked-single propagation, which alone solves 99.4% of easy, 76.6% of medium and 10.5% of hard test puzzles **[M, 4.0 rule baseline]**. This is expert iteration with a rule-based expert. It works, and it puts the solver in the sampler: less "on its own".
3. **Supplied small-k puzzles (reverse curriculum).** Start at k = 1 to 3 and raise k when the valid share passes 0.5. Needs a supplier who holds solutions.

After the cold start the curriculum is automatic: 99.4% of easy puzzles need only naked singles, so they fall first; every solved puzzle adds a full grid to a buffer; hindsight masks on buffer grids give medium and hard training puzzles.

## 4. Design for this repo, one stage at a time

Common rules: launches under 2 h (plan with the 60% overrun margin seen in 4.4), at most 3 MPS jobs, seeds fixed, predictions committed in `experiments/<id>/README.md` before the run, paired bootstrap on the 667 hard test puzzles, **all three bands reported** so a hard-band gain is not bought on the others. Reference: the 4.1 supervised model at equal training wall-clock, read from `results/phase4/sft/sft.csv` (no rerun). Evaluation uses only the validity check (`sudoku9.validity`), which equals "is the solution" on unique puzzles.

### Stage A: from scratch, no stored solutions (V1, sub-variant "givens allowed")

**A1, cold start on the givens. One launch, 60 min of training.**
- *Objective.* The loss of `src/phase2/train_diffusion.py::diffusion_loss` with one change: targets are the givens. Blanks always masked; each given masked with probability t ~ U(0.05, 1); cross-entropy on masked givens, weighted 1/t. The stored `solutions` tensor is never read (enforce it: pass a `PuzzleSet` whose solutions are zeros, and test that).
- *Reuse.* Model, optimiser, checkpointing and `band_eval` of `src/phase4/train.py`; `src/phase2/sampler.py::sample`.
- *Compute.* About 0.41 s per step **[M, 4.1]**: 8,700 steps in 60 min, plus 5 min of evaluation.
- *Predictions [C].* Valid share on easy val puzzles at 16 sampler steps between 0.05 and 0.6; at 81 steps higher than at 16; hard below 0.01. Low confidence in the upper bound.
- *Done when* the easy valid share is at least 0.05 at some step budget (5% of 33,334 easy train puzzles is 1,667 grids, enough to start A2).
- *Negative result.* Below 0.01 at every budget: givens alone do not teach filling blanks. Report it, then try route 2 or 3 of section 3.2, one at a time.

**A2, harvest and hindsight. Launches of 110 min.**
- *Loop.* (1) Roll out on a block of train puzzles (16 steps; 1,002 puzzles take 7.3 s **[M, 4.1]**, so 20,000 take about 2.5 min), keep valid grids in a buffer keyed by puzzle. (2) Train the diffusion loss on buffer grids for about 1,500 steps (10 min), half the batch with the puzzle's own mask (as `diffusion_loss` does), half with hindsight masks: uniform over all 81 cells at t ~ U(0.05, 1). (3) Keep a share of A1 loss on unsolved puzzles. Repeat.
- *Reuse.* The `self_diffusion` branch of `src/phase3/rl.py` is this objective on a per-batch basis; A2 is its buffered form, closer to `src/phase4/train.py` with a growing dataset.
- *Arms.* Hindsight on; hindsight off (own masks only). The second is the control that says whether relabelling, not just self-training, carries the hard band.
- *Predictions [C].* After the first launch the buffer holds over 90% of easy train puzzles; hard val solve at 16 steps is above 0.3 with hindsight and below 0.1 without.
- *Done when* hard val solve at 16 steps passes 0.2 (the lower bound of the 4.3 window).
- *Negative result.* Buffer stalls below 50% of easy, or hindsight on equals hindsight off.

**A3, equal wall-clock comparison.** Continue A2 to 5 h of total training time. Compare with the supervised model at 1, 2, 3 and 5 h. *Prediction [C]:* at 5 h, hard val at 16 steps is below the supervised 0.805 by 0.02 to 0.10; easy and medium within 0.01. *Negative:* a gap above 0.2, or easy below 0.99.

### Stage B: the wrong-cell oracle, at training time only (V3)

Stated up front: B uses labels. Start model: the 4.1 plateau model (so B does not wait for A), and later the A3 model.

**B2 first, as a prediction not worth a run [C].** Using the oracle to remask and refill, then training the denoiser on the repaired grids (SUNDAE-like), turns nearly every failed train puzzle into a labelled one, since `fix_oracle` reaches 0.907 in 4 passes and 1.000 in 8 **[M, 4.5b]**. That is `sup-all` of 4.6d, which gave 0.610 at 4 steps against 0.602 for self-training, difference +0.007 [−0.006, +0.022] **[M]**. Expected gain over self-training at 4 steps: none. Do not spend a launch on it unless B1 needs repaired grids anyway.

**B1, the model's own error detector. One launch, about 1 h.**
- *Objective.* One logit per cell on the trunk; input is the model's own completed grid (all cells visible) from an N-step fill of hard train puzzles; binary cross-entropy against the oracle bit, on filled cells. Train the trunk too (4.6b: a linear readout of frozen features reaches recall 0.60 at false-alarm rate 0.22, AUROC 0.59 to 0.64 inside failed grids **[M]**), and keep the diffusion loss on at equal weight so the denoiser does not drift. This is ELECTRA with the model's sampler as generator, and PRISM with oracle targets in place of its self-correction loss.
- *Reuse.* `src/phase4/detector_gate.py` (`features`, `fix`, `degraded`), `src/phase4/eraser.py::sample_fix`, `src/phase3/conf_token.py` for the head.
- *Arms.* Head on a frozen trunk; head and trunk; head and trunk with batches of failed grids only (against the "failed grid vs solved grid" shortcut of 4.6b). Seeds 0 to 2.
- *Compute [C].* Grids for 33,334 puzzles at 4 steps: about 1 min. About 5,000 steps at 0.3 to 0.4 s: 30 min per run, about 1 h for three side by side.
- *Target, from 4.6b [M].* Recall at least 0.9 at false-alarm rate at most 0.3 **inside failed test grids**. At that point the degraded oracle gave +0.05 to +0.08 at 8 passes and +0.02 to +0.04 at 4.
- *Evaluation, no oracle at inference.* Fill-then-fix with the head, detector passes charged to the budget and not charged, against `plain` at the same number of passes. Also 4-step and 16-step plain solve rate of the trained trunk, to see drift.
- *Controls.* The diffusion loss alone for the same steps on the hard band, and on all bands (the control 4.6d did not run); a head trained on shuffled oracle bits.
- *Predictions [C].* Trained trunk: recall 0.75 to 0.9 inside failed grids, so the target is missed or just met. Fill-then-fix not charged beats `plain` at 8 passes only if recall reaches 0.9; charged, it does not beat `plain` at 8 passes. Low confidence throughout: a real detector's misses are correlated within a grid, which the degraded oracle did not model.
- *Negative result.* Recall inside failed grids stays below 0.7 with the trunk trained: detection is not learnable at this size from the final grid. That is a capacity statement worth having.

### Stage C: RL on the remasking policy, validity reward only

Only if B1 comes within reach of its target, or as the label-free variant below.
- *Setup.* Fill 32 puzzles with N − 2 steps; for each grid draw 8 remask sets from the head (independent Bernoulli per filled cell: the action likelihood is exact, no one-step proxy); one refill pass; reward = validity, times (1 − λ · share of cells remasked) so that "remask everything" is not the optimum (step-count reward of Jazbec et al., [2512.09106](https://arxiv.org/abs/2512.09106) **[P, survey]**).
- *Objective.* Imitation of successful sets and positive-only weights first: every push-down arm lost in 4.4 and 4.6c **[M]**. wd1 as a third arm, because here its likelihood is exact (direction 5 of the earlier memo).
- *Cold start [C].* A set must cover nearly all of about 16 wrong cells. If each is drawn with probability 0.9, a set succeeds 18% of the time; at 0.8, 2.8% (7 of 256 per step); at 0.6, 0.03% (dead). False alarms are cheap **[M, 4.6b]**, so start with a high base rate and let the size penalty bring it down.
- *Two starts.* From the B1 head (V3 at training, nothing at inference); from the rule-conflict flag with a high base rate (V1/V2 only, the label-free version).
- *Compute [C].* 4 fill passes for 32 grids, one refill pass for 256, one small backward pass: 1 to 1.5 s per step; 500 steps × 3 seeds in about 30 min.
- *Predictions [C].* From the B1 head: +0.02 to +0.05 over B1 alone at 8 passes. From the conflict flag: no gain over `plain`.
- *Negative result.* The policy drifts to remasking everything or nothing; solve rate equals `plain` at N − 2 or N steps.

## 5. Risks

- **Mode collapse onto easy puzzles.** Self-training only ever sees what it solves. Hindsight masks decouple the grid from the mask; the hindsight-off arm measures the effect. Watch the band mix of the buffer and digit frequencies per position (the 4.0 check) in harvested grids.
- **The detector learns "failed grid" instead of "wrong cell"** (4.6b: AUROC 0.75 over all cells, 0.59 to 0.64 inside failed grids **[M]**). Always report the inside-failed-grids number; train one arm on failed grids only.
- **Oracle leakage.** Stage B is label use. A2 must provably never read `solutions`. Symmetry augmentation, small-k puzzles and the rule mask each inject knowledge and must be named in every table that uses them.
- **Evaluation on the same 667 hard test puzzles** as every phase 4 experiment. Choose settings on val (334 hard puzzles, intervals about ±0.05); count how many settings were tried; a fresh test split from a new generator seed costs seconds and would be cleaner for the final A3 number.
- **Mask mismatch.** A1 never sees a state with fewer than 30 masked cells; hindsight masks are not unique-solution puzzles. Either could cap the solve rate below the supervised model.
- **MPS non-determinism** and three seeds: intervals over puzzles do not cover seed variance.

## 6. How long until it can do it alone

All **[C]**, anchored on **[M]** timings (0.41 s per SFT step; 5 h for 43,500 steps; harvest 2.5 min per 20,000 puzzles).

| milestone | estimate | range | main assumption |
|---|---|---|---|
| first valid rollouts on the easy band | 1 h (A1) | 0.5 h to never | givens teach enough; else +1 to 2 h for a fallback route |
| easy band above 0.95, alone | 1.5 to 2 h | 1 to 4 h | supervised needed 4 min; A1 sees sparser contexts |
| hard band above 0.5 at 16 steps | 3 to 4 h | 2 to 8 h | hindsight grids are as good as stored ones |
| level with the supervised model (hard 0.81 at 16 steps) | 6 to 11 h | 5 h to not reached | 1.2 to 2 times the supervised time: cold start, harvest overhead, fewer distinct grids early |
| beyond it at 4 steps through a learned detector (B, C) | not estimated | | depends on recall 0.9, which no experiment has reached |

So: about one to two nights of launches for Stage A. The estimate is a guess with a factor of 2 on either side, and the first row can fail outright.

## 7. Recommended first step

**Run A1 only: 60 minutes of masked training on the givens, from random weights, then the valid share per band at 4, 16 and 81 sampler steps.** About 1 h 10 min in one launch; one new loss option and one test.

Why this one: it is the cheapest experiment that decides whether the cold start exists without any tool; every later stage depends on it; its control is free (the supervised curve already on disk); and both outcomes are informative. Write the three predictions of A1 into `experiments/<id>/README.md` first.

## 8. Not verified

- **Not opened:** Bengio et al. 2009 (curriculum learning); CLR-DRNets (Bai, Chen, Gomes, CP 2021: PDF unreadable, title and claim from a search snippet only).
- **Search summaries or secondary sources only:** TRM's 87.4% and HRM's 55.0% on Sudoku-Extreme; Chang et al. (2312.11522); RWR (ICML 2007 listing, paper not read).
- **Attribution from memory:** predictor-corrector steps in Campbell et al. and corrector sampling in Discrete Flow Matching (not in their abstracts); the count of 6.67 × 10^21 solved grids (also in the phase 4 spec).
- **Abstract pages only** for every paper opened for this memo; no full text was read. "Sudoku among the tasks" for ConsFormer, DDReasoner and PRISM means no number was seen. DDReasoner's first stage is not called supervised in its abstract. Venues were seen on arXiv pages, not checked against proceedings.
- **Carried over** from the two earlier memos with their own caveats: everything marked "earlier memos" or "survey".
- **"Not found" claims** rest on four web searches, not a systematic review.
- Everything marked **[C]**, in particular: that training on givens produces valid rollouts, the a^k approximation, the timing table, and the remask-set arithmetic (it assumes independent draws and 16 wrong cells).

## References

Opened for this memo (arXiv abstract pages):
- AlphaZero: Silver et al., 2017. https://arxiv.org/abs/1712.01815
- Expert Iteration: Anthony, Tian, Barber, 2017. https://arxiv.org/abs/1705.08439
- Formal Mathematics Statement Curriculum Learning: Polu et al., 2022. https://arxiv.org/abs/2202.01344
- Absolute Zero: Zhao et al., 2025. https://arxiv.org/abs/2505.03335
- Self-Verified Distillation: Lee, Liang, 2026. https://arxiv.org/abs/2605.26132
- NeuroSAT: Selsam et al., 2018. https://arxiv.org/abs/1802.03685
- Self-Imitation Learning: Oh, Guo, Singh, Lee, 2018. https://arxiv.org/abs/1806.05635
- DAgger: Ross, Gordon, Bagnell, 2010. https://arxiv.org/abs/1011.0686
- Scheduled Sampling: Bengio, Vinyals, Jaitly, Shazeer, 2015. https://arxiv.org/abs/1506.03099
- Reverse Curriculum Generation: Florensa et al., 2017. https://arxiv.org/abs/1707.05300
- Hindsight Experience Replay: Andrychowicz et al., 2017. https://arxiv.org/abs/1707.01495
- GCSL: Ghosh et al., 2019. https://arxiv.org/abs/1912.06088
- Ambient Diffusion: Daras et al., 2023. https://arxiv.org/abs/2305.19256
- Recurrent Relational Networks: Palm, Paquet, Winther, 2017. https://arxiv.org/abs/1711.08028
- SATNet: Wang, Donti, Wilder, Kolter, 2019. https://arxiv.org/abs/1905.12149
- Recurrent Transformer for CSPs: Yang, Ishay, Lee, 2023. https://arxiv.org/abs/2307.04895
- Semantic loss: Xu, Zhang, Friedman, Liang, Van den Broeck, 2017. https://arxiv.org/abs/1711.11157
- ConsFormer: Xu, Li, Sanner, Khalil, 2025. https://arxiv.org/abs/2502.15794
- DDReasoner: Zhang et al., 2025. https://arxiv.org/abs/2508.16524
- Neuro-Symbolic Sudoku Solver: Hathidara, Pandey, 2023. https://arxiv.org/abs/2307.00653
- Beyond Autoregression: Ye et al., 2024. https://arxiv.org/abs/2410.14157
- Train for the Worst, Plan for the Best: Kim, Shah, Kontonis, Kakade, Chen, 2025. https://arxiv.org/abs/2502.06768
- IRED: Du, Mao, Tenenbaum, 2024. https://arxiv.org/abs/2406.11179
- HRM: Wang et al., 2025. https://arxiv.org/abs/2506.21734
- TRM: Jolicoeur-Martineau, 2025. https://arxiv.org/abs/2510.04871
- Self-Aware Markov Models: Kornhardt, Chemseddine, Wald, Steidl, 2026. https://arxiv.org/abs/2603.16661
- GFlowNets: Bengio, Jain, Korablyov, Precup, Bengio, 2021. https://arxiv.org/abs/2106.04399
- Campbell et al., 2022. https://arxiv.org/abs/2205.14987
- Discrete Flow Matching: Gat et al., 2024. https://arxiv.org/abs/2407.15595
- GIDD: von Rütte et al., 2025. https://arxiv.org/abs/2503.04482
- PRISM: Kim et al., 2025. https://arxiv.org/abs/2510.01384
- DDPD: Liu et al., 2024. https://arxiv.org/abs/2410.06264
- SCOPE / D3IM: Yu et al., 2026. https://arxiv.org/abs/2606.01026
- ELECTRA: Clark, Luong, Le, Manning, 2020. https://arxiv.org/abs/2003.10555
- Levenshtein Transformer: Gu, Wang, Zhao, 2019. https://arxiv.org/abs/1905.11006
- Edit Flows: Havasi, Karrer, Gat, Chen, 2025. https://arxiv.org/abs/2506.09018
- Mask-Predict: Ghazvininejad, Levy, Liu, Zettlemoyer, 2019. https://arxiv.org/abs/1904.09324
- SUNDAE: Savinov et al., 2021. https://arxiv.org/abs/2112.06749

Found through search, not opened as a paper:
- Chang, Flokas, Lipson, Spranger, NeurIPS 2020. https://arxiv.org/abs/2312.11522
- RWR: Peters, Schaal, ICML 2007. https://icml.cc/imls/conferences/2007/proceedings/abstracts/98.htm
- CLR-DRNets: Bai, Chen, Gomes, CP 2021. https://drops.dagstuhl.de/storage/00lipics/lipics-vol210-cp2021/LIPIcs.CP.2021.17/LIPIcs.CP.2021.17.pdf

Reused from `docs/RL_DLLM_SURVEY.md` and `docs/RL_DLLM_DIRECTIONS.md` (see their own verification notes): d1 2504.12216, wd1 2507.08838, SPG 2510.09541, GDSD 2605.29398, MDPO 2508.13148, DDPO 2305.13301, SEPO 2502.01384, DDPP 2410.08134, RTB 2405.20971, IGPO 2509.10396, CanvasAnneal 2609.13060, AWR 1910.00177, RAFT 2504.11343, Davis and Recht 2510.13651, self-distillation through time 2410.21035, Mohamud et al. 2606.00295, Jazbec et al. 2512.09106, ReMDM 2503.00307, P2 2502.03540, Informed Correctors 2407.21243, Token-Critic 2209.04439, RemeDi 2509.23653.
