# Phase 5: looped masked diffusion on 9×9 Sudoku, and RL that decides how long to think

**Status: plan written on 24 Nov 2026, nothing has run.** Literature memo: [LOOPED_LITERATURE.md](LOOPED_LITERATURE.md) (six search rounds). Same rules as phases 1–4: local MacBook only, predictions committed before each run, failed runs kept, no novelty claims.

## Why this phase
In November 2026 looped transformers (one block of layers run several times with the same weights) went into production models: Inkling 2 Mini loops its middle layers 3 times, GLM-5.4 was made by converting a trained model so that 28 layers run twice. Masked diffusion models already have an outer loop (denoising steps); a looped denoiser adds an inner one (depth per step). Phase 4 left two facts that make this interesting on our toy:
- on 9×9 the sampler budget matters a lot: the supervised model solves 0.535 / 0.814 / 0.871 of the hard band at 4 / 16 / 81 steps (4.1);
- failures are one or two early wrong cells followed by cells that fit them (4.4b), and remasking cannot fix them unless a detector finds about nine wrong cells in ten (4.6b).

The literature memo shows that "a small looped MDM matches a much bigger dense one on Sudoku" and "loops partly replace denoising steps" are already known (R-MDM, arXiv:2606.18022; LoopMDM, arXiv:2605.26106). Phase 5 replicates them as its baseline and spends most of its budget on what the memo did not find:
1. **RL with a solved / not-solved reward deciding how many loops to run at each denoising step** (5.5, the main experiment);
2. whether the loop trajectory finds wrong committed cells better than single-pass confidence (5.3);
3. whether loops remove the early wrong cells that phase 4 identified as the failure mode (5.2).

Outcomes are open. A null on 5.5 (RL no better than a simple entropy rule) is a result and will be reported as one.

## Lessons from phase 4 built into the design
- **Learning-rate confound.** Every phase 4 model was trained at a constant learning rate, and every later step at a lower one; the "gain" of several RL objectives was the learning-rate drop (4.9e, 4.9f). Phase 5 trains every model from scratch with warm-up + cosine decay to a fixed step count. Every fine-tuning arm has a control arm with the same steps and the same learning-rate schedule.
- **Overused test set.** The hard test band was read many times in 4.4–4.9. Phase 5 generates a fresh test split once (`test5`, new generator seed, no solution shared with train/val/test) and reads it only for the final number of each experiment. Development and selection use val.
- **Seeds.** Screening with one seed is allowed; any comparison that ends up in the summary uses three seeds and a paired bootstrap over puzzles.

## Model: `LoopedTinyDLLM`
Same embeddings, head and masked diffusion loss as phase 4 (`src/phase2/model.py`), with the encoder split in three parts:

```
tokens → embed (+ pos, row, col, box) → prelude (P layers) → core (C layers) × S → coda (Q layers) → norm → head
```

- `S` = loops per denoising step. Parameters do not depend on `S`; compute per forward pass is proportional to `P + C·S + Q` layer passes (the unit used for every compute comparison below: **block passes per puzzle** = Σ over denoising steps of `P + C·S_t + Q`).
- Training: `S` sampled uniformly in {1, …, S_max} per batch (LoopMDM recipe, also the stochastic schedule that stabilises extrapolation in arXiv:2606.29983). Loss on the final output.
- Option `inject: true`: add the prelude output to the core input at every loop (recall, Bansal et al. 2022). Default off (LoopMDM found noise injection harmful; plain input injection is an ablation in 5.4).
- Per-loop readout for analysis only: `coda → norm → head` applied to the core state after every loop, without gradient. Gives per-loop logits, entropy, argmax flips and relative hidden change `ρ_s = ||h_s − h_{s−1}|| / ||h_s||`.
- Halting head for 5.5: linear layer on the mean-pooled core state over still-masked cells → `p(halt)` after each loop. One decision per puzzle per denoising step (per-cell halting, as in ALoDLM, is a stretch goal 5.5c).
- `S = 1` with `P + C + Q` layers is exactly a dense model, so the dense baselines use the same class.

Unit tests: `S = 1` equals the dense forward with the same weights; block-pass counter; per-loop readout does not change outputs; halting head never stops before loop 1 and always by `S_max`; sampler with a fixed `S` reproduces phase 4 numbers on a phase 4 checkpoint loaded as `P = 6, C = Q = 0`.

## Cost (from the 10 Nov 2026 benchmark, d 160, batch 256, MPS)
About 13 ms per layer per forward pass, about 50 ms per layer per training step. A model with `P + C·S + Q = 6` costs what the 1.86M dense model costs (81 ms forward, 292 ms per step). Evaluation of 2,001 puzzles at 81 sampler steps and 6 block passes per step takes about 1 min; at 24 block passes per step about 4 min. Every run is planned under 2 hours or split into resumable launches; anything longer is asked first (CLAUDE.md).

## 5.0 Infrastructure and fresh test split (no training)
- `src/phase5/model.py` (`LoopedTinyDLLM`), `src/phase5/sampler.py` (phase 2 sampler + loops, per-loop readout, halting, block-pass counter), tests as above.
- `test5`: 2,001 puzzles, same bands and sizes as phase 4 test, generator seed 5, gate checks of `data_report.md` rerun on it (uniqueness, no overlap with any phase 4 split, also after digit relabelling).
- Done when: tests pass, `test5` gate passes, a phase 4 checkpoint loaded into the new class reproduces its val numbers within 0.005.

## 5.1 Replication and gate: does looping help on our data?
**Arms** (d 160, 4 heads, all trained from scratch with the same schedule: 500 warm-up steps, cosine decay to 0 at 30,000 steps, batch 256, lr 3e-4, phase 4 data):

| arm | layout | parameters | block passes per forward |
|---|---|---|---|
| `D6` | dense, 6 layers | 1.86M | 6 |
| `D3` | dense, 3 layers | ≈ 0.95M | 3 |
| `L3` | P1 + C1 × S + Q1, S ~ U{1..6} | ≈ 0.95M | 2 + S |

`D3` is the iso-parameter control, `D6` the iso-depth control of `L3` at S = 4. Optional `5.1b` (only if 5.1 passes): the same three arms at d 96, for the "model too small to solve it without loops" question.

**Measure** on val, then once on `test5`: valid share per band at sampler steps {4, 16, 81} × S ∈ {1, …, 8}, confidence sampler, greedy; training FLOPs per arm.

**Predictions (to be copied into `experiments/5.1/README.md` and revised only before the run):**
1. `L3` at S = 4 beats `D3` on the hard band at 16 steps by ≥ 0.05, interval above zero.
2. `L3` at S = 4 is within 0.05 of `D6` at 16 steps (R-MDM reports looped ≥ dense at equal depth).
3. `L3` gains most at 4 sampler steps.
4. `L3` at S = 1 is far below `D3` (it has only 3 layer passes and was trained mostly at larger S).

**Gate.** If prediction 1 fails (looping does not beat its iso-parameter control at 16 steps), steps 5.2–5.5 are not run on `L3`: write up 5.1 and decide with the user whether to try the R-MDM layout (whole block looped, loop-step embedding, deep supervision) before going on.

Cost: three arms × about 2.5 h, run side by side (split into two resumable launches each; ask before starting).

## 5.2 Do loops remove the early wrong cells? (eval only, about 1 h)
From the 5.1 checkpoints:
- Grid S ∈ {1, 2, 4, 6, 8} × sampler steps {1, 2, 4, 8, 16, 32, 81}; plot valid share against block passes per puzzle. Is there a budget at which "few steps, many loops" beats "many steps, one loop" at equal block passes?
- With the 4.4b tooling: share of rollouts whose *first* wrong cell appears at denoising step 1, 2, …; number of wrong cells per failed rollout; per S.

**Predictions:** at ≤ 16 block passes per cell-commit round, more loops beat more steps; the share of rollouts with a wrong cell committed at step 1 falls with S (from the phase 4 level) and accounts for most of the gain at 4 sampler steps; at 81 steps the curves meet.

## 5.3 Is the loop trajectory a better wrong-cell detector? (eval only, about 1 h)
For every cell committed during sampling, record: final softmax confidence; entropy; entropy drop across loops (R-MDM's stop signal); number of argmax flips across loops; last loop at which the argmax changed; `ρ` at the last loop. Labels: is the committed digit wrong (from the stored solution, analysis only).

- AUROC per signal and for a logistic combination (fit on val, read on `test5`), per band and per denoising step.
- Recall of wrong cells at 10% false positives, against the 4.6b bar (about 0.9 needed for remasking to pay; a linear readout of the frozen phase 4 model reached about 0.6).
- Two uses: (a) the best signal as the unmasking score instead of confidence; (b) as a remask trigger (mask and re-predict cells above a threshold, as in 4.5b).

**Predictions:** cross-loop signals add 0.02–0.05 AUROC over single-pass confidence; no signal reaches the 4.6b bar, so remasking gains stay ≤ 0.02; using the signal as the unmasking score changes the valid share by less than 0.02. A recall above 0.8 would be reported as a positive surprise and followed up.

## 5.4 More loops than in training (eval + two small ablations)
- From `L3` (trained with S ≤ 6): S ∈ {6, 8, 12, 16, 24} at 16 and 81 sampler steps. Where does the curve peak, does it fall (overthinking)?
- Ablations, one seed, 15,000 steps each, same schedule shortened: `L3-fixed` (S = 4 always) and `L3-inject` (uniform S + input injection).
- After 5.5 and 5.6: the same sweep on their final models (does RL move the window?).

**Predictions:** uniform-S `L3` holds within 0.02 of its best value up to S = 12 and degrades by 24; `L3-fixed` peaks at S = 4 and drops beyond 6; injection widens the window.

## 5.5 RL decides how many loops to run (main experiment)
**Question.** With only a rule-checker reward, can a policy learn where in the denoising process extra loops pay, and does it beat simple stop rules at the same compute?

**Setup.**
- Start: `L3` from 5.1. Stage 5.5a: denoiser frozen, only the halting head trains (as 3.4 did for the unmasking order: isolates the allocation question from changes in the denoiser).
- Rollout: sample a hard train puzzle with the confidence sampler at 4 and at 16 steps; after each loop at each step the head samples halt / continue; `S_t ∈ {1, …, S_max = 8}`.
- Reward: `R = valid(grid) − λ · (block passes / block passes at S_max)`, λ ∈ {0, 0.1, 0.3}. Group of 8 rollouts per puzzle, group-mean baseline, REINFORCE on the halting log-probabilities, entropy bonus 0.001; 32 puzzles per step, lr 1e-3 for the head, 1,000 steps.
- Anti-collapse, decided now because RecurTrace saw penalties collapse to one loop: a run whose mean S stays at 1.0 ± 0.05 for its last 200 steps is reported as collapsed; one extra arm with a 2-loop floor.

**Baselines at matched compute** (each swept to trace a curve of valid share vs mean block passes on val):
- fixed S ∈ {1, …, 8};
- entropy-drop rule (R-MDM): stop when mean entropy over masked cells drops by less than λ_e between loops;
- hidden-change rule (LoopMDM): stop when `ρ < ε`;
- oracle-label head (RecurTrace-style, uses solutions, so it is a supervised reference, not label-free): continue when the next loop lowers the cross-entropy on masked cells by more than δ.

**Read-out.** Valid share on `test5` at the mean block passes of each RL arm, read off each baseline curve by interpolation; paired bootstrap; three seeds for the RL arms. Also: where the policy spends loops (by denoising step, by number of masked cells, by band).

**Success criterion, fixed now:** an RL arm beats the best rule at matched mean block passes by ≥ 0.02 on the hard band of `test5`, interval above zero, in at least 2 of 3 seeds.

**Predictions:**
1. λ = 0 learns to run S_max almost always (no reason to stop), matching fixed S = 8.
2. λ = 0.3 collapses to S = 1 (as RecurTrace found for ACT/PonderNet); the 2-loop floor arm does not collapse.
3. The best non-collapsed RL arm is within ±0.02 of the entropy-drop rule at matched compute (success criterion **not** met).
4. Whatever its score, the policy puts more loops in the first denoising steps than in the last (where phase 4 says errors start).

Stage 5.5b, only if 5.5a meets the success criterion or comes within 0.01 of it: unfreeze the denoiser and train it together with the head, using the best label-free objective of phase 4 (training on own checker-valid grids), with a control arm trained the same way at fixed S (same steps, same lr schedule). Stage 5.5c (stretch): per-cell halting.

Cost: an RL step at 4 sampler steps and S ≤ 8 is about 32 forward passes of 2–10 layer passes on a batch of 256: about 2–3 s; 1,000 steps ≈ 45 min per arm. Three λ × three seeds + floor arm ≈ 10 runs, three in parallel.

## 5.6 Converting a trained dense model into a looped one (last; likely negative)
- Start: `D6` from 5.1 (cosine-trained, so no learning-rate confound). Loop layers 2–3 (early-middle, as LoopMDM) twice: same parameters, 8 layer passes.
- Arms, 2,000 steps each, same lr schedule (warm-up, cosine from 3e-5), one seed to screen, three for the final pair:
  - `free`: no training (training-free loop);
  - `sft`: masked diffusion loss on stored solutions, with the loop;
  - `self`: training on its own checker-valid grids (no labels), with the loop;
  - `pos`: positive-only wd1 (β = 0), with the loop;
  - `sft-ctrl`, `self-ctrl`: the same training without the loop.
- Read: each looped arm against its own no-loop control, and the label-free arms against `sft`.

**Predictions:** `free` loses (arXiv:2605.23872 reports naive reapplication degrades); `sft` beats `sft-ctrl` by 0.00–0.04; `self` and `pos` are within ±0.02 of their controls, in line with phase 4 and with the null GRPO result after conversion in arXiv:2610.42030. A clear positive for `self` would be the most interesting outcome of the phase and would get three seeds and a fresh look before being written up.

## 5.7 Write-up
`results/phase5/SUMMARY.md` in the phase 4 format (short version, main table, corrected readings, limits), reports updated, notebook `notebooks/phase5_looped.ipynb` loading committed results.

## Order, decision points, budget

| step | depends on | machine time (estimate) | decision after it |
|---|---|---|---|
| 5.0 | – | < 1 h | – |
| 5.1 | 5.0 | ≈ 8 h of training, 3 arms in parallel | gate: go on with `L3` or switch layout |
| 5.2, 5.3 | 5.1 | ≈ 2 h eval | which signal to give the policy as extra input in 5.5 (optional) |
| 5.5a | 5.1 | ≈ 4 h | 5.5b only if near the success criterion |
| 5.4 | 5.1, then 5.5 | ≈ 2 h | – |
| 5.6 | 5.1 | ≈ 3 h | – |
| 5.7 | all | – | – |

## What would matter beyond the toy
Stated with the usual caveat: a 1M-parameter Sudoku model says nothing directly about frontier models; it can only tell whether an idea is worth testing at a larger scale.
- If 5.5 works: loop allocation can be learned from a verifiable reward alone, the same kind of signal used to post-train reasoning models, instead of from likelihood or labels. That is relevant to models that expose the loop count at deployment (GLM-5.4, Inkling 2 Mini).
- If 5.3 works: the trajectory across loops is a cheap error signal, useful both for decoding and for monitoring latent computation, which is the concern raised around GPT-6 Astra.
- If 5.6 is positive for `self`: conversion to a looped model would not need stored solutions. If it is null, it adds a small, controlled data point to the null GRPO result of arXiv:2610.42030.

## Decisions taken in this plan
- d 160, 4 heads and the phase 4 data, so that numbers are comparable with phase 4.
- The 3-layer looped layout `P1 + C1 × S + Q1` (not the whole block looped) as the first choice; R-MDM's layout is the fallback after the gate.
- One halting decision per puzzle per step first; per cell only as a stretch.
- Fresh `test5` split; cosine schedule everywhere; control arm with the same schedule for every fine-tuning arm.
