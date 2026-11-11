# Plan

## Goal
Learn how masked diffusion LMs work in practice (generation, unmasking, confidence, RL fine-tuning) through small experiments that run on a MacBook, and publish them as a clean repo plus a short technical write-up.

## Constraints
- Hardware: Apple Silicon MacBook, 24 GB unified memory. **No cloud GPUs.**
- Backend: PyTorch with MPS; CPU fallback where an op is unsupported.
- Small, controlled experiments (50–200 examples), fixed seeds, every number reproducible from a committed config.
- No novelty claims. Report negative results and limitations.

## Phase 1: exploring a real dLLM (inference only)
Spec: [PHASE1_LLADA_ON_MAC.md](PHASE1_LLADA_ON_MAC.md)

| Step | Output |
|---|---|
| 1.0 Load LLaDA-8B-Instruct on MPS, smoke test, measure memory and seconds per denoising step | `results/phase1/env_and_speed.md` |
| 1.1 Record unmasking order on a handful of prompts | heatmaps + GIFs in `results/phase1/unmasking/` |
| 1.2 Compare unmasking strategies × step budgets on Sudoku 4×4, Countdown, GSM8K subsets | `results/phase1/strategies.csv` + accuracy-vs-steps plot |
| 1.3 Does confidence predict errors? Token- and answer-level calibration | reliability diagrams + ECE table |
| 1.4 Write-up of phase 1 | `results/phase1/SUMMARY.md` |

## Phase 2: a toy dLLM and RL in miniature (training)
Spec: [PHASE2_TOY_SUDOKU_DLLM.md](PHASE2_TOY_SUDOKU_DLLM.md)

| Step | Output |
|---|---|
| 2.0 Sudoku 4×4 data generator, validity checker, splits | `src/phase2/data.py` + tests |
| 2.1 Tiny bidirectional transformer + masked diffusion training (SFT stage) | checkpoint (local only) + training curve |
| 2.2 Sampler with the same unmasking strategies as phase 1 | accuracy per strategy |
| 2.3 RL fine-tuning with a hand-written wd1 objective, plus ablation without the negative weight | reward curves, accuracy before/after |
| 2.4 Confidence before vs after RL | reliability diagrams |
| 2.5 Write-up of phase 2 | `results/phase2/SUMMARY.md` |

## Phase 3: playing with RL on the toy model (training)
Spec: [PHASE3_RL_PLAYGROUND.md](PHASE3_RL_PLAYGROUND.md). Added after phases 1 and 2 were finished, to follow up the open questions from phase 2.

| Step | Output |
|---|---|
| 3.0 Generic RL trainer with pluggable objective and reward; reproduce the phase 2 runs | `src/phase3/rl.py` + tests |
| 3.1 Take wd1 apart: horizon, sharp weights, negative dose, starting model, graded reward, reference term | curves + table per sub-step in `results/phase3/` |
| 3.2 wd1 next to REINFORCE, filtered imitation, d1-style GRPO and a supervised reference | comparison plot + table |
| 3.3 Reward for honest confidence (confidence token, Brier-shaped reward) | calibration before/after, vs temperature scaling |
| 3.4 Learn the unmasking order with RL on a frozen denoiser | headroom check, solve rate vs heuristics |
| 3.5 Write-up of phase 3 | `results/phase3/SUMMARY.md`, reports updated |

Three follow-ups were added during the phase, each with its own pre-registered prediction: 3.2b (KL penalty to the start), 3.1g (the better variants from other starting models), 3.3b (the confidence token on a frozen model). Three robustness checks were added after the summary, also with predictions: 3.6a (sequential likelihood in the loss), 3.6b (rarer reward, weaker checker), 3.6c (likelihood along the rollout's own path).

## Phase 4: a 9×9 Sudoku dLLM and the phase 3 objectives again (training)
Spec: [PHASE4_SUDOKU9_DLLM.md](PHASE4_SUDOKU9_DLLM.md). Added after phase 3 to see which of its conclusions survive a harder task.

| Step | Output |
|---|---|
| 4.0 9×9 generator, uniqueness check, difficulty bands, and a data report that gates training | `src/phase4/generate.py` + tests, `results/phase4/data_report.md` |
| 4.1 Size check, then masked diffusion training to a plateau | checkpoint (local only) + training curve |
| 4.2 Sampler strategies × step budgets, order headroom, calibration | tables + plots per difficulty band |
| 4.3 Choose the RL start by a rule fixed in the spec | share of valid rollouts at the start |
| 4.4 The seven phase 3 objectives at their phase 3 settings | comparison plot + table, predictions checked |
| 4.5 Settings sweeps: screen with one seed, confirm with three | per-arm table of settings tried |
| 4.7 Write-up | `results/phase4/SUMMARY.md`, reports updated |

## Final deliverable
- Clean repo with reproducible commands per experiment.
- `results/REPORT.md`: 2–4 pages, 4–6 figures, what I learned, what surprised me, limitations.
- `results/REPORT_SIMPLE.md`: the same content in plain language.

## Rough timeline (evenings and weekends)
- Week 1: 1.0–1.2 (first visuals early, they are the most shareable output)
- Week 2: 1.3–1.4, start 2.0–2.1
- Week 3: 2.2–2.4
- Week 4: 2.5 and the final report
- Weeks 5–6: phase 3 (3.0–3.2 first, then 3.3 and 3.4)

## Definition of done for any experiment
1. Config committed in `configs/` or `experiments/<id>/config.yaml`.
2. Exact command in `experiments/<id>/README.md`.
3. Result summary (CSV/plot) in `results/`.
4. Hardware, torch version, dtype, seed, wall-clock time recorded.
5. A two-line "what this shows / what it does not show" note.
