# Looped transformers and masked diffusion: literature memo for phase 5

Written on 24 Nov 2026, before any phase 5 code. Six search rounds, each started from what the previous one found. The goal was to check, for each candidate experiment, what is already known, so that phase 5 tests open questions and not ones that already have an answer.

Most papers below are 2026 arXiv preprints. Several entries were read from abstract pages or partial extracts, not the full PDF; those are marked *(abstract/extract)*. As in `REFERENCES.md`: verify metadata before citing in any write-up.

## Round 1: what happened in November 2026 (context)

- **Thinking Machines, Inkling 2 Mini** (20 Nov 2026): open-weight MoE, 26.4B total / 3.4B active parameters; the middle layers run 3 times per token (5.4B effective active per token); runs on a 32 GB Mac. Sources: MarkTechPost, 21 Nov 2026; pasqualepillitteri.it, 23 Nov 2026.
- **Z.ai, GLM-5.4** (23 Nov 2026): about 750B parameters, built by *converting* the trained GLM-5 base: 28 of its 78 layers run twice per forward pass; claimed quality of 2–3T-parameter models; API now, open weights after safety work. Sources: letsdatascience.com, aiweekly.co, dev.to, SCMP (24 Nov 2026).
- **OpenAI GPT-6 Astra** (Sept 2026): reported to use "recurrent depth" in part of the model; its system card reports lower chain-of-thought monitorability and higher no-CoT capability; safety debate (Fortune 3 Sep 2026; Gizmodo; LessWrong posts). Later reporting on GPT-6.1 Sol ("two inference passes instead of three", via a Microsoft Foundry page).
- **mdlARC** (blog, Nov 2026): a ~100M looped *autoregressive* transformer on ARC-AGI-2 (8 blocks, blocks 3–6 looped 6 times); the author states the loop gain does not need answer refinement. https://mvakde.github.io/blog/looped-mdlarc/

Relevance: conversion of trained models into looped ones (GLM-5.4) and loop count as a deployment-time knob are now practice at frontier scale. Both are things a toy can probe.

## Round 2: looped / recursive masked diffusion, and recursive puzzle solvers

| work | what | relevance |
|---|---|---|
| **LoopMDM**: Lee, Hong, Kim, Lee, Park, Park, *Looped Diffusion Language Models*, arXiv:2605.26106 | Head / looped mid-block / tail; loop count S sampled uniformly in {1..S_max} per training step; NELBO on the final loop output; no input injection (noise injection hurt). Early-middle placement best. Sudoku: 1-layer MDM vs its looped version, d 384, 100k puzzles, **fixed left-to-right order**: baseline 10.9%, looped at S = 1 10.6%; an example goes 24 → 7 → 0 errors as S goes 1 → 3 (full-test numbers for S > 1 not reported in the extract). Adaptive loops: stop when relative hidden change < 0.1 (≈5 loops instead of 12 at equal quality; intermediate timesteps get more loops). Future work they list: learning the stop rule; combining with adaptive unmasking. No RL, no calibration. | Closest recipe for our model; their stop rule is a baseline for 5.5 |
| **R-MDM**: Carballo-Castro, Piskorz, Rauba, van der Schaar, Frossard, *Recursive Scaling in Masked Diffusion Models*, arXiv:2606.18022 (ICML 2026) | K-layer block applied L times per denoising step, shared LM head, logits after every loop, deep supervision over loops, learned loop-progress embedding. 9×9 Sudoku (1.8M boards): valid rate at T = 5 steps 65.2% → 93.6% with L = 5; at T = 40 95.8% → 98.0%. Iso-depth: 6⊗5 recursive (10.8M) 93.6% vs 30-layer non-recursive (53.1M) 77.8% at T = 5. Recursion partly substitutes for denoising steps. Adaptive decoder: stop looping when the drop in mean entropy between loops falls below λ; entropy predicts token errors (AUROC 0.86). No RL. *(extract)* | **Already shows the "tiny model + loops solves Sudoku" result and the loops-vs-steps trade-off.** 5.1 is a replication on our data, not a new result |
| **FP-MGM / CoFRe**: Miele, Qin, Carballo-Castro, Deschenaux, Frossard, *Fixed-Point Masked Generative Modeling*, arXiv:2605.31215 | Part of the denoiser replaced by a fixed-point solver over shared layers; depth adapts across sampling steps; latent reused across steps; pretrained MGMs convertible with short fine-tuning | Adaptive depth per step via convergence, no RL |
| **ALoDLM**: Fang et al., *Adaptively Looped Diffusion Language Models*, arXiv:2610.04198 | Token-adaptive recurrence inside each denoising step: low-entropy tokens commit, a learned ExitGate decides halting (max depth 4); exit depths are latent variables in a NELBO with a geometric prior favouring shallow exits; gate trained with a score-function estimator. 1.7B/8B on Qwen3 backbones, 11 benchmarks, no Sudoku. Latent state not carried across denoising steps. | **The learned-halting idea exists in MDMs**, trained on likelihood, not on a task reward |
| **CCDD**, arXiv:2510.03206 | Continuous + discrete co-evolving diffusion; on Sudoku / 3-SAT / Countdown a looped transformer with 2 loops beats MDM, CCDD beats both | Context |
| **Diffusion as a training curriculum for timestep-free iterative reasoning**, Drozdova et al., arXiv:2609.01449 | Persistent hidden state + shared update, no timestep; Sudoku-Extreme 99.9% at 10,000 iterations; attention restricted to row/column/box | Carrying state across steps; context *(abstract page)* |
| **LAW**, arXiv:2610.30068; **Loopholing Discrete Diffusion** | Pass a latent across sampling steps (self-conditioning); LAW 96.5% on hard Sudoku | Context |
| **Thinking with Looped Flows**, arXiv:2609.11801 | Recurrence trained with local denoising objectives | Context |
| **TRM**: Jolicoeur-Martineau, *Less is More: Recursive Reasoning with Tiny Networks*, arXiv:2510.04871 | 5M (MLP) / 7M models, Sudoku-Extreme 87.4% vs HRM 55.0% (27M); deep supervision, ACT halting | Known prior art for tiny recursive Sudoku solvers |
| **Speed is Confidence**, Dillon, arXiv:2601.19085 | On TRM: puzzles where a chain halts at step 0 are 99.2% correct, those needing ≥ 8 steps 78%; halt-first ensembling 97.2% | Halting speed as a correctness signal (relevant to 5.3) |
| **InfiLoop** (HF `pengxiang/InfiLoop-Sudoku`), 6.83M, Sudoku-Extreme 97.9% | Context |
| **Adaptive Order Policies for Masked Diffusion**, arXiv:2606.00295 | 6M 3-layer MDM on 9×9 Sudoku, learned unmasking order | Same scale as ours |

## Round 3: converting a trained model into a looped one

| work | finding |
|---|---|
| Bae et al., *Relaxed Recursive Transformers*, arXiv:2410.20672 (ICLR 2025) | Initialise a shared block from pretrained layers, "uptrain"; recursive Gemma 1B from Gemma 2B recovers most of the quality; depth-wise LoRA relaxes the tying |
| McLeish et al., *Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence*, arXiv:2511.07384 | Converting to recurrent beats plain post-training of the original at equal compute, on maths |
| Chen et al., *Training-Free Looped Transformers*, arXiv:2605.23872 | Naive reapplication of a mid-stack block usually **degrades** a frozen model; the loop application rule matters |
| Sukthanker, Cameron, *There Is No Free Loop*, arXiv:2610.42030 | Qwen3/3.5 conversions (LoRA, KL to original): only the 0.8B loop beats both an identically trained twin and the best unlooped use of the same inference FLOPs; at 4B/9B no significant gain; **GRPO after conversion gave no significant change**; a loop pays only under a fixed parameter budget |
| LoopUS, arXiv:2605.11011 | Recasts pretrained LLMs into looped latent refinement with halting; suggests applying it to diffusion LLMs as future work |
| GLM-5.4 (Round 1) | Conversion at 750B scale, method details not public |

Reading: conversion works with supervised uptraining or distillation; RL on a converted model has a null result at 4B/9B. Nothing found on converting a *masked diffusion* model by RL alone.

## Round 4: running more loops than in training (extrapolation, overthinking)

- Bansal et al. 2022, *End-to-end Algorithm Synthesis with Recurrent Networks* (arXiv:2202.05826): recall (input injection) and a progressive loss avoid overthinking.
- Fan et al., *Looped Transformers for Length Generalization*, arXiv:2409.15647: adaptive number of steps helps length generalisation.
- Geiping et al., *Huginn* / recurrent depth, arXiv:2502.05171: gains plateau beyond loops seen in training (as cited in arXiv:2604.15259).
- *Stability and Generalization in Looped Transformers*, arXiv:2604.15259; *Beyond the Training Horizon*, arXiv:2609.33144.
- Yang, Huang, Wang, Mirzasoleiman, *Learning to Extrapolate in Depth with Looped Transformers*, arXiv:2610.14456: extrapolation works inside a finite window; too many loops create false connections.
- Kuo, Chayti, Reizinger, Brendel, Jaggi, *Stabilizing Extrapolation in Looped Transformers via Learned Stochastic Stopping*, arXiv:2606.29983: runs with similar in-distribution accuracy differ sharply out of distribution; random loop counts in training reduce that variance; RL-Halting (REINFORCE on −loss, no compute cost) helps on 3 of 4 tasks.
- *Latent overthinking* (arXiv:2511.08577): extra iterations can turn right answers wrong.
- LoopMDM (Round 2): uniform S sampling lets the model run beyond S_max; very large S_max degrades.

Reading: well studied on algorithmic tasks with autoregressive or encoder models. For MDMs only the LoopMDM claim. Not found: whether RL fine-tuning moves the extrapolation window.

## Round 5: learned halting and RL on looped models

- ACT (Graves, arXiv:1603.08983); PonderNet (Banino et al., arXiv:2107.05407); Universal Transformer (Dehghani et al., arXiv:1807.03819).
- Ouro, Zhu et al., *Scaling Latent Reasoning via Looped Language Models*, arXiv:2510.25741: entropy-regularised learned exit; reported no gain from RL with verifiable rewards on its final checkpoint (as cited in arXiv:2610.00673).
- AdaPonderLM, arXiv:2603.01914: token-wise learned gates beat fixed pruning at iso-FLOPs.
- **RecurTrace**, Wang et al., arXiv:2609.03379: on a frozen looped Qwen3-1.7B, **ACT and PonderNet collapse to one loop at every penalty in {0.001, 0.01, 0.1}**, because few examples benefit from extra loops; an oracle label ("continue only if a deeper loop lowers the loss by more than δ") avoids collapse: 56.9% at 2.04 loops vs fixed T = 2 54.7%.
- RL-Halting (Round 4): REINFORCE on the stop decision, no compute term.
- ALoDLM (Round 2): score-function gate inside an ELBO.
- *Closing the Loop: Practical Training Recipes for Looped LMs*, arXiv:2610.00673: RLTT (GRPO across recurrent iterations) left MATH unchanged on a checkpoint that already solved about three quarters of its rollouts, and added 12 points on GSM8K from a checkpoint that solved about a third. RL helps when the rollout solve rate is low.
- LoopRPT, arXiv:2603.19714: reinforcement pre-training with dense step-wise rewards over loop transitions.
- Looped Actor, arXiv:2609.37432; FlexLoop, arXiv:2609.34488: adaptive loop depth inside RL policies (control tasks).

Reading: learned halting exists, mostly trained on likelihood or on oracle labels. With a compute penalty, outcome-style pressure tends to collapse to the minimum. **Not found: outcome-reward RL (solved / not solved, from a rule checker) that allocates loops per denoising step in a masked diffusion model, compared against the entropy-drop and hidden-change rules at matched compute.**

## Round 6: loop dynamics as an error signal

- R-MDM (Round 2): entropy predicts token errors, AUROC 0.86, used to decide what to commit.
- *Operational Proto-Introspection in Looped LMs*, arXiv:2607.18553: correctness readable from the trajectory of hidden states across loops (Ouro).
- *Monitoring latent reasoning in looped language models*, arXiv:2610.18127: per-iteration readout needs no logit-lens approximation in looped models.
- *Towards Looped Models Done Right, Part II*, arXiv:2610.06833: tokens converge at different depths.
- *Targeted Remasking*, arXiv:2605.26436: remask a committed token when its confidence drops between denoising iterations.
- PURL, Zheng et al., *When Should Diffusion Language Models Reconsider Their Commitments?*, arXiv:2611.11551: late, selective revision works better than frequent revision; on Sudoku PURL 71.6% vs MDLM 18.0% at K = 32 *(extract)*.

Reading: single-pass entropy is a known error signal. Not found: whether the trajectory across loops (flips, entropy drop, convergence depth) finds wrong *committed* cells better than single-pass confidence in an MDM, measured against the detector quality that phase 4 showed remasking needs (4.6b: about nine wrong cells in ten).

## Verdict per candidate experiment

| candidate | already known | still open | decision |
|---|---|---|---|
| A tiny model with loops solves what the same-size dense model cannot | R-MDM (Sudoku, iso-param and iso-depth), LoopMDM, TRM | nothing new; needed as our own baseline | **5.1, replication and gate** |
| Loops substitute for denoising steps | R-MDM | *why* in our setting: phase 4 found failures start with one or two early wrong cells (4.4b); do loops remove exactly those? | **5.2, cheap, eval only** |
| Loop dynamics as a wrong-cell detector | entropy AUROC 0.86 (R-MDM); trajectory readouts in AR looped LMs | cross-loop signals vs single-pass confidence for *committed* cells in an MDM, against the 4.6b bar | **5.3, cheap, eval only** |
| Extrapolation beyond trained loops | many papers on algorithmic tasks; LoopMDM claim for MDMs | whether RL fine-tuning moves the window | **5.4, small** |
| RL decides how many loops per denoising step | ALoDLM (likelihood-trained gate), RL-Halting (no compute cost), RecurTrace (penalties collapse) | outcome-reward RL with a compute cost in an MDM, vs entropy-drop / hidden-change rules at matched compute | **5.5, main experiment** |
| Convert a trained dense dLLM into a looped one with RL only | conversion with SFT/distillation works; GRPO after conversion null (No Free Loop); Ouro null | RL-only conversion of an MDM; our phase 4 history predicts a null | **5.6, last, cheap, likely negative** |

## News sources (Round 1)
- https://www.marktechpost.com/2026/11/21/thinking-machines-inkling-2-mini-looped-transformer/
- https://pasqualepillitteri.it/en/news/36281/what-is-a-looped-transformer-and-why-inkling-2
- https://letsdatascience.com/news/zai-launches-glm-54-its-first-loop-transformer-flagship-7b50e4c7
- https://aiweekly.co/alerts/zai-ships-glm-54-a-750b-loop-transformer-that-runs-28-of-78-layers-twice
- https://www.scmp.com/tech/big-tech/article/3375517/chinas-zai-aggressively-using-looped-transformer-method-improve-ai-models
- https://www.fortune.com/2026/09/03/reports-openais-astra-model-uses-a-new-more-efficient-ai-architecture-alarms-ai-safety-experts-who-worry-the-method-makes-models-harder-to-control/
- https://www.lesswrong.com/posts/PLisnSFir8y5AHkmP/how-concerned-should-we-be-about-openai-s-recurrent
