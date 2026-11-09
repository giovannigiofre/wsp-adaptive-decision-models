# Diffusion language models on a MacBook, in plain words

This is the simplified companion to [REPORT.md](REPORT.md). It covers the same two phases with less jargon and more "why". Numbers are the same ones as in the full report; nothing new was run for this page. It is a learning exercise, not research, and makes no novelty claims.

Where a sentence starts with *Likely why*, it is an interpretation that was **not** tested.

## The idea in one minute

A normal language model writes one word at a time, left to right. A **masked diffusion language model (dLLM)** works differently:

```
start:   [MASK] [MASK] [MASK] [MASK] [MASK]
step 1:  [MASK]   is   [MASK] [MASK]   .
step 2:   The     is   [MASK]  42      .
step 3:   The     is   answer  42      .     (order is free, several words per step)
```

- **Training:** take a real text, hide a random share of its words, and ask the model to guess the hidden ones. The model can look both left and right.
- **Generating:** start from all masks. At each step the model guesses every hidden word, *commits* a few of them, and keeps the rest masked for later. In the samplers used here a committed word is never revised.
- **Unmasking strategy:** the rule for choosing which words to commit. We compared four: `random`, `confidence` (commit what the model is most sure about), `margin` (biggest gap between first and second choice) and `entropy` (least spread-out guess).

Two questions run through the whole project: **does the order of unmasking matter?** and **can we trust the model's confidence?**

## Phase 1: a real dLLM (LLaDA, 8 billion parameters), inference only

We loaded LLaDA-8B-Instruct on the laptop and tested it on three tasks: 4×4 Sudoku, Countdown (reach a target number with three numbers) and GSM8K (school maths word problems).

| Question | What we saw | Why |
|---|---|---|
| Does it run on a laptop? | Yes: 16 GB of memory, about 0.9 s per step. A short answer takes about a minute. | An 8B model is at the limit of a 24 GB machine. |
| In what order does it write? | With the `confidence` rule it writes *almost* left to right (correlation between position and commit time: 0.64). | *Likely why:* words next to already-known text are the easiest to guess, so they are the most confident and go first. |
| Does the rule matter? | On Countdown, `confidence` beat `random` by 28 points [11, 44], averaged over step budgets. The three "smart" rules could not be told apart. | *Likely why:* `random` commits unsure words early, and they cannot be fixed later. |
| Do more steps help? | Halving the steps halved the time with no loss we could detect. | With samples this small, "no detectable loss" is not the same as "no loss". |
| Can we trust its confidence? | No. On Countdown it was 87% confident and 42% correct. On Sudoku, the cells it was most confident about were mostly the *wrong* ones. | The model copied the puzzle's `0` placeholders with high confidence. Confidence measures how easy a word is to produce, not whether it is correct. |

![unmasking order](phase1/unmasking/gsm8k-416_heatmap.png)

*Commit order for one GSM8K answer. With the confidence rule (default in LLaDA) each block fills in mostly left to right.*

![countdown strategies](phase1/strategies_countdown.png)

## Phase 2: a tiny dLLM built from scratch, then reinforcement learning

To understand the pieces from the inside, we built a toy version: a 0.8-million-parameter model that fills in 4×4 Sudoku grids. Training it takes 13 minutes on the laptop.

**Step 1: ordinary training.** Same recipe as above: hide cells, guess them. The fully trained model solves every test puzzle. To leave room for improvement we deliberately took an early, half-trained snapshot that solves **48%**.

**Step 2: reinforcement learning (RL).** RL needs no answer key, only a checker that says "valid" or "not valid":

1. For each puzzle, let the model produce 8 attempts.
2. Score each attempt: 1 if the grid is valid, 0 if not.
3. Compare each attempt with the average of its group of 8.
4. Nudge the model: make better-than-average attempts **more** likely and worse-than-average attempts **less** likely.

This is the **wd1** objective, written by hand here. We also ran a variant, **positive-only**, that keeps the "more likely" part and drops the "less likely" part. The wd1 paper reports that this variant collapses.

| Model | Test puzzles solved |
|---|---|
| half-trained start | 48% |
| start + wd1 | 74% |
| start + positive-only | 93% |
| fully trained (reference) | 100% |

![wd1 vs positive-only](phase2/rl_wd1_vs_positive_only.png)

What happened, in order:

- **Positive-only collapsed early**, in all three runs: from 42% down to about 10% on validation puzzles within 150 steps. This matches the paper's warning.
- **Then it recovered and won**, passing wd1 around step 400. This does *not* match the paper.
- **wd1 never dipped**, but it climbed slowly and was still at 74% when the runs stopped at step 500.

*Likely why the collapse happens* (a reading of the code and logs, not a tested cause): with 0/1 scores, the weights that decide "how much to reinforce each attempt" are almost flat. In a group with 2 valid attempts out of 8, the 6 invalid ones still receive about half of the total weight. So positive-only also reinforces the model's own mistakes, and the logs show it becoming more confident in wrong grids. wd1's "less likely" part cancels this.

*Why positive-only wins in the end:* not tested. Candidate reasons are the very easy task, the half-trained starting point, the short runs, and the missing "stay close to the starting model" term.

Two smaller findings:

- **Order matters for a weak model, not for a strong one.** The half-trained model solves 48% with the `confidence` rule and 14% with `random`. The fully trained model solves 99.8% in a single step either way.
- **RL changed confidence in two ways.** After wd1 the model was better at *ranking* right above wrong, but it became over-confident overall (95% confident, 88% correct on cells).

![calibration before and after RL](phase2/calibration_before_after.png)

## What did not go well

- **Phase 1 samples are tiny.** 12 Sudoku, 12 Countdown and 5 GSM8K examples instead of the planned 100/100/50, because each answer takes about a minute. The error bars are ±20 to ±40 points. These are pilot numbers and cannot be compared with published ones.
- **Sudoku with LLaDA told us nothing about strategies.** 0 puzzles solved in 144 attempts.
- **GSM8K answers were cut** to 128 tokens instead of 256.
- **The first model load failed** (the laptop started swapping). Loading straight onto the GPU fixed it.
- **The first toy model was stuck at 14%.** It only knew each cell's index. Telling it each cell's row, column and box fixed it.
- **The toy task is too easy.** The interesting regime exists only because we started from a half-trained snapshot, which is an artificial and fragile setup.
- **The RL is simplified.** 0/1 rewards, no "stay close" term, 500 steps, 3 runs per setting, one set of hyper-parameters.
- **Measurements are noisy.** The laptop GPU is not perfectly repeatable, and both phases ran at the same time, so timings are pessimistic.

## Open questions for a next phase

1. Would wd1 catch up with longer runs? It was still improving at step 500.
2. Does positive-only still collapse when the weights are made sharper, so that invalid attempts get almost no weight?
3. How much of wd1's "less likely" push is the right amount: all, half, none?
4. Do the conclusions change with a different starting model or a graded reward instead of 0/1?
5. Can RL be rewarded for honest confidence, so it does not become over-confident?

## Mini glossary

- **dLLM:** a language model that fills in masked text over several steps instead of writing left to right.
- **Commit / unmask:** fix the value of a masked position.
- **Step budget:** how many fill-in steps an answer gets. More steps cost more time.
- **Confidence:** the probability the model gives to the word it picked.
- **Calibration:** whether confidence matches reality. "80% confident" should mean "right 8 times out of 10".
- **ECE:** average gap between confidence and actual accuracy. 0 is perfect.
- **AUROC:** whether confidence ranks right answers above wrong ones. 0.5 is a coin flip; below 0.5 is misleading.
- **RL (reinforcement learning):** learning from a score on the model's own attempts, with no answer key.
- **wd1:** an RL objective for dLLMs that raises the likelihood of better-than-average attempts and lowers that of worse-than-average ones.
- **95% interval [a, b]:** the range the true value plausibly lies in, given how few examples we have.
