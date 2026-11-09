# Phase 2: a tiny masked diffusion model on 4×4 Sudoku, then RL

Goal: build every piece by hand at toy scale (data, model, diffusion training, sampler, RL objective) so each concept from phase 1 and from the d1/wd1 papers is understood from the inside. Everything trains on MPS in minutes to a few hours.

## 2.0 Data
- A 4×4 Sudoku grid has 16 cells, digits 1–4, each row, column and 2×2 box contains 1–4 once. There are exactly **288** valid solved grids: enumerate them all with a backtracking solver.
- Split the 288 solutions by seed into train (200) / val (44) / test (44). Puzzles for a split are only built from that split's solutions.
- Puzzle = solution with k cells hidden, k ∈ {6, …, 12}. Keep only puzzles with a **unique** solution (check with the solver). Store `(puzzle, solution, k)`.
- Vocabulary: `PAD=0, MASK=1, digits 1–4 → tokens 2–5`. Input sequence = the 16 cells in row-major order (optionally with a 2D position embedding).
- `validity(grid, puzzle) -> bool`: full grid valid and consistent with the givens. Unit tests: all 288 solutions valid, random grids mostly invalid, givens respected.

## 2.1 Model and masked diffusion training (the "SFT" stage)
- Model: bidirectional transformer encoder (no causal mask), ~4 layers, d_model 128, 4 heads, learned position embeddings for 16 positions. About 1M parameters.
- Training objective (LLaDA/MDLM style):
  1. Take a solved grid x0 and its puzzle. Given cells are always visible.
  2. Sample t ~ U(0.05, 1). Mask each non-given cell independently with probability t.
  3. Loss = cross-entropy on masked cells only, weighted by 1/t, averaged over the batch.
- Optimiser AdamW, lr 3e-4, batch 256, cosine schedule. Log train/val loss and val solve rate (with the 2.2 sampler) every N steps.
- **Deliberately produce a weak model too:** save checkpoints along training and pick an "SFT" checkpoint whose val solve rate is roughly 30–60%, so RL has room to improve. Document how it was chosen. Optionally also keep the fully trained model as an upper reference.

## 2.2 Sampler
- Start: all non-given cells masked. Budget S steps; commit ⌈remaining/steps_left⌉ cells per step.
- Strategies: `random`, `confidence`, `margin`, `entropy`, same definitions as phase 1 (share the selection function if possible).
- Temperature T for sampling the committed values (T=0 greedy for eval, T=1 for RL rollouts).
- Report solve rate per strategy and per step budget S ∈ {1, 2, 4, 8, 16} on the test puzzles. S=1 means "everything in one shot", which shows why multi-step unmasking matters.

## 2.3 RL fine-tuning with a hand-written wd1 objective
Reward: r = 1 if `validity(grid, puzzle)`, else 0.

Per training step:
1. Sample a batch of B puzzles from train. For each, draw G = 8 completions with the current model (T = 1, confidence strategy, fixed S).
2. Group advantage: A_i = r_i − mean(r_group). Skip groups where all rewards are equal.
3. wd1 weights inside each group, with temperature τ (default 1):
   - w⁺_i = softmax(A / τ)_i
   - w⁻_i = softmax(−A / τ)_i
4. Log-likelihood estimate (d1/wd1 "one-step" mean-field estimator): build the input with all non-given cells masked (optionally also mask each given cell with prob p_mask = 0.15), run one forward pass, and sum log p(o_i,k) over the non-given cells k.
5. Loss = Σ_i (w⁻_i − w⁺_i) · log p̂(o_i). Minimising this raises the likelihood of good completions and lowers that of bad ones. Several gradient steps (μ = 4) per batch of rollouts, as in d1/wd1.
6. Log: mean reward, val solve rate (greedy), loss, weight entropy, and the fraction of groups skipped.

Runs:
| ID | Variant |
|---|---|
| 2.3a | wd1 as above |
| 2.3b | ablation: positive-only (w⁻ = 0), which the wd1 paper reports as collapsing |
| 2.3c | τ ∈ {0.5, 1, 2} sensitivity |

At least 3 seeds for 2.3a and 2.3b. Plot reward and val solve rate vs RL step (mean ± std across seeds).

Sanity checks before trusting results:
- The loss sign: on a hand-made group with one correct and one wrong completion, one gradient step must increase log p̂(correct) and decrease log p̂(wrong). Write this as a unit test.
- Compare the one-step estimator with a slower "true" likelihood (unmask cells one at a time in a fixed order and sum log-probs) on a few samples, and report their correlation.

## 2.4 Confidence before vs after RL
- On the test puzzles, for the SFT checkpoint and the best RL checkpoint: per-cell commit confidence vs cell correctness, and puzzle-level mean confidence vs solved.
- Reliability diagrams side by side, ECE, AUROC.
- Question: does RL make the model more confident overall? Is it better or worse at knowing when it is wrong?

## 2.5 Phase 2 summary
`results/phase2/SUMMARY.md`: model size, training time on the Mac, SFT vs RL solve rates with seeds, the wd1 vs positive-only ablation, calibration before/after, limitations (toy scale, 4×4 only, small data), and what this does or does not say about large dLLMs.

## Code layout
```
src/phase2/
  sudoku.py        # enumeration, solver, uniqueness check, validity
  data.py          # splits, puzzle generation, tensors
  model.py         # tiny bidirectional transformer
  train_diffusion.py
  sampler.py       # strategies + step budgets (shared selection logic with phase 1)
  rl_wd1.py        # rollouts, advantages, weights, one-step log-lik, loss
  evaluate.py      # solve rates, calibration metrics
tests/
  test_sudoku.py
  test_wd1_sign.py
  test_sampler.py
```
