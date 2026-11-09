# Phase 3: playing with RL on the toy Sudoku dLLM

Goal: use the phase 2 toy (0.8M-parameter masked diffusion model on 4×4 Sudoku, RL runs of about one minute) as a playground to understand *why* the phase 2 RL results came out the way they did, to put wd1 next to simpler RL objectives, and to try two extensions: a reward for honest confidence and a learned unmasking order.

Everything stays at toy scale on MPS. RL on LLaDA-8B is out of scope: the bf16 weights alone take 16 of the 24 GB, which leaves no room for gradients.

No novelty claims. The variants below are exploratory experiments on a toy task. Before describing any of them as new, check the literature in `REFERENCES.md`.

## Where phase 2 left off

Phase 2 result: from a weak start (0.48 test solve rate), wd1 reached 0.74 and the positive-only ablation 0.93 after an early collapse to about 0.10. Three reasons were offered and none was tested: binary rewards, a mid-warm-up starting checkpoint, and no KL/reference term.

Re-reading the phase 2 logs (`results/phase2/rl/*.csv`, means over 3 seeds) adds three observations that shape this phase:

1. **The weights were almost flat for every τ tried.** The entropy of w⁺ was 1.75–1.90 (τ = 0.5), about 2.0 (τ = 1) and 2.06 (τ = 2), against a maximum of ln 8 = 2.08. With 0/1 rewards and τ = 1, a group with 2 valid completions out of 8 puts 52% of the w⁺ mass on the 6 invalid ones. So the τ sweep never left the near-uniform regime, and positive-only was mostly raising the likelihood of *all* its samples.
2. **wd1 lowered the estimated likelihood of valid completions too.** One-step log p̂ went from −7.4 to −10.9 for valid samples and from −9.7 to −14.7 for invalid ones (first 50 vs last 50 steps). The gap widened, but both fell. Positive-only moved both *up* (to about −5.2 and −7.5).
3. **wd1 was still improving when the runs stopped.** Val solve rate was 0.69, 0.70, 0.73 at steps 400, 450, 500, and by then 69% of the groups were skipped for having equal rewards.

These are observations from existing logs, not tested explanations. Phase 3 turns them into experiments.

## Ground rules for every experiment

- **Same data, model and evaluation as phase 2** unless a step says otherwise: splits by solution, `sft_weak.pt` (SFT step 120) as the start, test solve rate with the greedy `confidence` sampler at S = 8.
- **Write the prediction first.** Each `experiments/3.x/README.md` gets a "Prediction" line committed *before* the run, then the outcome next to it. A wrong prediction stays in.
- **Seeds:** 3 per setting for sweeps, 5 for the headline comparisons. Report mean ± std over seeds and the per-seed values. Select on val, report on test.
- **Equal budgets:** methods are compared at the same number of rollouts and gradient steps.
- **Phase 2 stays frozen.** New code goes in `src/phase3/` and imports the phase 2 model, sampler, data and likelihood helpers. Phase 2 results are not regenerated.
- **Cost:** a 500-step RL run takes about 1 minute. No single launch should exceed 2 hours; ask before starting one that would.

## 3.0 Generic RL trainer and reproduction

- `src/phase3/rl.py`: one training loop with pluggable `objective` (how rewards become a loss) and `reward` (how a grid is scored). Resumable, one CSV row per step, as in phase 2.
- Extra diagnostics logged every step:
  - share of the w⁺ mass that falls on invalid samples;
  - effective sample size of the weights;
  - one-step log p̂ of valid and invalid samples;
  - mean predictive entropy on masked cells (how sharp the model is);
  - val solve rate at S = 8 and at S = 1.
- **Done when:** wd1 (τ = 1) and positive-only, seed 0, 500 steps, land within the phase 2 seed spread (0.74 ± 0.01 and 0.93 ± 0.02 on test). MPS is not bit-deterministic, so an exact match is not expected.
- Unit tests: each objective's sign on a hand-made two-sample group, each reward function on known grids.

## 3.1 Track A: taking wd1 apart

| ID | Question | Runs |
|---|---|---|
| 3.1a | Was step 500 too early? | wd1 and positive-only, 2000 steps, 5 seeds |
| 3.1b | What happens when the weights are actually sharp? | τ ∈ {0.05, 0.1, 0.25, 1} × {wd1, positive-only}, 3 seeds |
| 3.1c | How much negative push is the right amount? | loss = Σ (β·w⁻ − w⁺)·log p̂, β ∈ {0, 0.25, 0.5, 1, 2}, 3 seeds; plus a floored variant |
| 3.1d | Does the starting model change the story? | 3 SFT snapshots + 1 converged-but-weak model × {wd1, positive-only}, 3 seeds |
| 3.1e | Does a graded reward help? | binary vs fraction of valid rows/columns/boxes × {wd1, positive-only}, 3 seeds |
| 3.1f | Does a reference term temper wd1? | wd1 with a penalty towards the frozen start, 3 strengths, 3 seeds |

Details:

- **3.1a Horizon.** Fix the standard horizon for the rest of the phase from this result (likely 1000 steps). Watch whether positive-only stays at the top or degrades, and whether wd1 saturates because almost all groups are skipped.
- **3.1b Sharp weights.** For m valid samples out of G, the share of w⁺ on invalid samples is (G − m) / ((G − m) + m·e^(1/τ)): 52% at τ = 1, 5% at τ = 0.25, about 0% at τ = 0.1 (for m = 2). If the early collapse of positive-only is caused by reinforcing its own mistakes, it should shrink as τ falls. If it does not, that reading is wrong.
- **3.1c Negative dose.** β = 0 is positive-only and β = 1 is wd1. Run at τ = 1 and at the best τ from 3.1b. The floored variant stops pushing a sample down once its log p̂ is below a fixed floor, motivated by observation 2. It is not from the paper.
- **3.1d Starting point.**
  - Snapshots with val solve rate near 0.2, 0.4 and 0.6, picked from the dense evaluations in `results/phase2/sft_training.csv`.
  - A converged-but-weak model: SFT run to convergence on only a few of the 200 train solutions (pilot the number so that val lands in 0.3–0.6), then RL on puzzles from all 200 using the validity checker only. This is the realistic case for RL: few labels, a cheap checker. It also removes the "one sharpening step away" objection.
- **3.1e Graded reward.** r = number of valid units / 12 (4 rows, 4 columns, 4 boxes). It needs no labels. Evaluation is still the binary solve rate.
- **3.1f Reference term.** First clone the wd1 repository into `~/dllm-cache/upstream/`, pin the commit in `configs/phase3.yaml`, and read how the reference policy enters the objective there. Implement the closest toy equivalent and document every difference. Do not guess the form from memory.

## 3.2 Track B: wd1 next to simpler objectives

All hand-written, all on the same rollouts, likelihood estimator and budget. 5 seeds each.

| Method | Loss | Why include it |
|---|---|---|
| REINFORCE with group baseline | −Σ A_i · log p̂(o_i) | wd1 with linear instead of softmax weights; also run with A divided by the group std |
| Filtered imitation | −Σ over valid samples of log p̂(o_i), and the same with the standard masked-diffusion loss | the τ → 0 limit of positive-only; "train on your own valid outputs" |
| d1-style GRPO | clipped ratio exp(log p̂_θ − log p̂_old) with the same masks, normalised advantage | the method wd1 argues against; adapt from the d1 repository (pinned in `configs/phase1.yaml`) with attribution |
| Supervised reference | masked-diffusion loss on the true solutions, same number of updates and learning rate | what the same budget buys *with* labels; a reference line, not a competitor |

Output: one learning-curve plot and one table (final test solve rate, worst val point during training, steps to reach 0.7). The question is whether wd1's behaviour is specific to wd1 or shared by any policy-gradient objective here.

## 3.3 Track C: rewarding honest confidence

Phase 2 found that wd1 makes the model over-confident (cells: 0.95 confidence against 0.88 accuracy). RLCR (see `REFERENCES.md`) trains a model to state a confidence q and rewards it with correctness minus a Brier penalty, r′ = r − (q − r)², which is maximised by an honest q.

Toy version: add a 17th position holding a confidence token with 4 values, read as q ∈ {0.125, 0.375, 0.625, 0.875}. The output head already has 4 classes. The new position needs its own embedding, so the weak checkpoint is loaded non-strictly and the new embedding starts untrained. The token is sampled during rollouts and its log-probability is part of log p̂.

| ID | When the confidence token is unmasked | What q means |
|---|---|---|
| 3.3a | last, after the grid | "is the grid I just wrote valid?" (self-verification) |
| 3.3b | first, from the puzzle alone | "will I solve this puzzle?" (difficulty prediction) |

- Train with the best objective from 3.1/3.2, reward r′, against the same objective with the plain reward r.
- Metrics: test solve rate (it should not drop), ECE and AUROC of q, and ECE and AUROC of the implicit commit confidence as in phase 2.4.
- Baseline that needs no RL: temperature scaling of the commit confidence, fitted on val (Guo et al.). If one fitted number does as well, say so.
- Things to watch:
  - 3.3a may be easy for the wrong reason: validity is a function of the grid, so q can become a checker rather than an uncertainty estimate.
  - In 3.3b the grid is generated *after* q is committed, so a low q could become self-fulfilling. Report solve rate split by q.
  - The token starts untrained and learns from reward alone, which may be slow at lr 1e-5. A flat result is a valid outcome.

## 3.4 Track D: learning the unmasking order with RL

Phase 2 showed that order matters for the weak model: one cell per step solves 0.56 with the `confidence` rule and 0.15 with `random`. Can a learned rule beat the heuristic? Reference: Jazbec et al. in `REFERENCES.md`; read their formulation before implementing and do not claim equivalence.

- **3.4a Headroom check first (seconds, no training).** With the weak denoiser frozen, run an oracle that always commits a masked cell whose greedy prediction matches the true solution, and fails if there is none. Its solve rate is achievable by *some* order, so it bounds the headroom from below. If it is close to 0.56, stop here and report that order is not the bottleneck.
- **3.4b Policy.** The denoiser stays frozen. A small head scores each masked cell and one cell is sampled per step from a softmax over those scores; the denoiser's greedy value is committed.
  - v1 inputs: per-cell max probability, margin and entropy, so the policy can express the three heuristics and mixtures of them.
  - v2 inputs: the denoiser's final hidden states.
- **Training.** Reward = validity, G = 8 attempts per puzzle, REINFORCE with the group baseline. Here the log-probability of an order is exact (a product of per-step softmaxes), so no likelihood proxy is needed. That contrast with 3.1–3.2 is worth a sentence in the write-up.
- **Evaluation.** Test solve rate against `random`, `confidence`, `margin`, `entropy` at one cell per step, 5 seeds. Then: how similar is the learned order to the confidence order (rank correlation), and does it transfer to an RL-tuned denoiser from 3.1?
- Optional: several cells per step with a budget S = 8.

## Follow-ups added while running the phase

Not in the plan above; each has a README with a prediction committed before its runs.

- **3.2b** Why GRPO beat wd1 and REINFORCE: remove its KL term, add the same term to REINFORCE and to wd1 (`rl.ref_kl`).
- **3.1g** Rerun the variants that beat wd1 from the phase 2 start (filtered imitation, self-training, floored wd1, wd1 + KL) from the two harder starts of 3.1d.
- **3.3b** The confidence token on a frozen denoiser, so its target is stationary; REINFORCE instead of wd1 for the token.

## 3.5 Phase 3 summary

`results/phase3/SUMMARY.md`: for each track the question, the prediction, what happened, and what it does and does not show. Then update `results/REPORT.md` and `results/REPORT_SIMPLE.md` with a short phase 3 section and at most three new figures. Keep every negative and flat result.

## Order and rough cost

| Step | Compute (toy, MPS) |
|---|---|
| 3.0 trainer + reproduction | ~10 min |
| 3.1a–f | ~4 h in total, in six separate launches |
| 3.2 | ~40 min |
| 3.3a–b | ~1 h |
| 3.4a–b | ~1 h |

3.0 comes first. 3.1a and 3.1b come next because they fix the horizon and τ used everywhere else. 3.4a is independent and cheap, so it can run early to decide whether track D is worth building.

## Code layout
```
src/phase3/
  rl.py            # generic loop: rollouts, reward, objective, diagnostics, resume
  objectives.py    # wd1(beta, floor), positive-only, reinforce, filtered imitation, grpo
  rewards.py       # binary validity, graded units, Brier-shaped
  conf_token.py    # 17th-position confidence token (track C)
  order_policy.py  # unmasking policy head, exact order log-prob, oracle (track D)
  analysis.py      # tables, curves, calibration
tests/
  test_objectives_sign.py
  test_rewards.py
  test_order_policy.py
configs/phase3.yaml
```
