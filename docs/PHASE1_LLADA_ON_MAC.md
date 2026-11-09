# Phase 1: exploring LLaDA-8B-Instruct on a MacBook (inference only)

## Background (read first)
- LLaDA: a masked diffusion LM. The answer region starts as `[MASK]` tokens; at each denoising step the model predicts all masked positions at once and *commits* some of them. Generation is semi-autoregressive: the answer is split into blocks (e.g. 32 tokens) processed left to right; inside a block, the order is chosen by a *remasking/unmasking strategy*.
- The official sampler (`generate.py` in the LLaDA repo) exposes: `steps`, `gen_length`, `block_length`, `temperature`, `cfg_scale`, `remasking` (`low_confidence` or `random`).
- d1 / wd1 evaluate LLaDA on GSM8K, MATH500, Countdown and 4×4 Sudoku. We reuse their task formats and scoring so our numbers are comparable in spirit (not identical: different hardware, subsets, settings).

## 1.0 Setup, loading and speed

### Sources to pin (record exact commit/revision in `configs/phase1.yaml`)
- Model: `GSAI-ML/LLaDA-8B-Instruct` on Hugging Face (needs `trust_remote_code=True`).
- Sampler reference: `github.com/ML-GSAI/LLaDA` (`generate.py`). Copy it into `src/phase1/llada_generate.py` with attribution and the original license header, then extend it (do not rewrite from scratch).
- Task data + scoring: `github.com/dllm-reasoning/d1` (eval directory: Sudoku and Countdown files, answer parsers) and `github.com/xiaohangt/wd1`. GSM8K: `openai/gsm8k` test split.
- Check the LLaDA README for the required `transformers` version and pin it in `requirements.txt`.

### Memory plan (24 GB unified memory)
- bf16 weights ≈ 16 GB. Load with `torch_dtype=torch.bfloat16`, `low_cpu_mem_usage=True`, then `.to("mps")`. Batch size 1. `torch.inference_mode()`.
- Keep `gen_length` small: 64–128 for Sudoku/Countdown, 256 for GSM8K.
- If loading fails or the system swaps heavily:
  1. Close other apps and retry.
  2. Check whether macOS limits GPU-wired memory; document the limit and **ask the user** before changing any system setting (e.g. `sysctl iogpu.wired_limit_mb`).
  3. Fallback A: run on CPU in bf16 (slow but works) for the small visualisation steps.
  4. Fallback B: use a small masked diffusion LM (e.g. an MDLM ~170M checkpoint) for 1.1 only, and say so clearly in the results.
- Never download more than ~20 GB without asking the user.

### Smoke test
- One GSM8K prompt, `gen_length=128, steps=64, block_length=32`. Print the answer, peak memory (`torch.mps.driver_allocated_memory()` and `psutil`), seconds per denoising step, total seconds.
- Write `results/phase1/env_and_speed.md`. Use the measured speed to size every later experiment (target: each experiment run < 2 hours; reduce the number of examples if needed).

## 1.1 Visualising the unmasking order
- Instrument the sampler to record, for each answer position: the step at which it was committed, and its confidence (max softmax prob) at commit time.
- Prompts: 3 Sudoku, 3 Countdown, 3 GSM8K.
- Outputs per prompt:
  - Heatmap: x = answer position, colour = commit step (and a second panel with commit confidence).
  - GIF: the decoded text frame by frame, `[MASK]` shown as `░`.
- Compare `remasking=low_confidence` vs `random` on the same prompt.
- Save to `results/phase1/unmasking/`.

## 1.2 Unmasking strategies × step budgets
Implement these commit rules inside a block (each picks the k positions to commit at a step):
| Name | Score per masked position |
|---|---|
| `random` | uniform random |
| `confidence` | max softmax prob (LLaDA's `low_confidence`) |
| `margin` | top-1 prob − top-2 prob |
| `entropy` | −entropy of the predictive distribution (commit the lowest-entropy first) |

Grid: strategies × `steps ∈ {gen_length/4, gen_length/2, gen_length}` × tasks.
- Data: 100 Sudoku, 100 Countdown, 50 GSM8K (fixed seed, list of IDs saved in `experiments/1.2/ids.json`). Reduce if 1.0 shows it is too slow, and record the reduction.
- Greedy (`temperature=0`), `block_length=32`, zero-shot prompts in the d1 format.
- Metrics: accuracy (d1 scorers), mean wall-clock per example, number of forward passes.
- Output: `results/phase1/strategies.csv` and one plot per task: accuracy vs steps, one line per strategy, with 95% bootstrap intervals.

## 1.3 Does confidence predict errors?
Using the 1.2 runs with the `confidence` strategy and the full step budget:
- Answer-level: confidence = mean (and min) commit confidence over the answer tokens; label = correct/incorrect. Reliability diagram (10 bins), ECE, AUROC for "confidence predicts correctness".
- Token-level (Sudoku only, where every cell is checkable): per-cell commit confidence vs whether that cell is correct.
- Also check: are tokens committed *early* more or less likely to be right than tokens committed late?
- Output: `results/phase1/calibration/` (plots + `metrics.csv`).

## 1.4 Phase 1 summary
`results/phase1/SUMMARY.md`: setup, speed, the 3–4 best figures, numbers with intervals, limitations (subset sizes, bf16 on MPS, no exact match with paper settings), next questions for phase 2.

## Code layout
```
src/phase1/
  load.py            # model/tokenizer loading, device + dtype handling, memory report
  llada_generate.py  # copied LLaDA sampler + instrumentation + strategies
  tasks.py           # prompt builders and scorers (adapted from d1, attributed)
  run_grid.py        # 1.2 grid runner, resumable, writes CSV row per example
  calibration.py     # ECE, reliability diagrams, AUROC
  viz.py             # heatmaps and GIFs
tests/
  test_scorers.py    # scorer sanity checks on known answers
  test_strategies.py # strategy selection on a fake logits tensor
```
Runs must be **resumable** (append per-example rows; skip IDs already done), because long runs on a laptop get interrupted.
