# Instructions for Claude Code

You are implementing the experiments in this repo. Read `README.md`, `docs/PLAN.md`, then the phase spec you are working on (`docs/PHASE1_LLADA_ON_MAC.md`, `docs/PHASE2_TOY_SUDOKU_DLLM.md` or `docs/PHASE3_RL_PLAYGROUND.md`) before writing code.

## Hard constraints
- Runs **only on the local MacBook** (Apple Silicon, 24 GB). Use `torch` with `mps`, falling back to `cpu` per-op or per-run when MPS fails. Never suggest or set up cloud GPUs.
- Ask the user before: downloading more than ~20 GB, changing any macOS system setting, installing system-level packages, or starting a run expected to take more than ~2 hours.
- Do not commit model weights, checkpoints, datasets larger than ~5 MB, or large logs. Keep them under `~/dllm-cache/` or in `.gitignore`d folders.

## How to work
- Follow the step order in `docs/PLAN.md`. Finish a step's definition of done before moving on.
- Start every phase with the cheapest possible end-to-end run (1 example, few steps), then scale.
- Reuse upstream code where the spec says so (LLaDA sampler, d1 task formats/scorers). Copy with attribution and license headers; keep changes minimal and marked.
- Pin every external revision (HF model revision hash, git commits) in `configs/`.
- Fix seeds (`random`, `numpy`, `torch`). Note that MPS is not fully deterministic; record it.
- Make long runs resumable and write results incrementally (one CSV row per example or per eval).
- Write small unit tests for scorers, sampling strategies, the Sudoku checker, and the wd1 loss sign. Run `pytest` before committing.
- Every experiment gets `experiments/<id>/README.md` with: goal, exact command, config path, hardware, torch version, dtype, seed, wall-clock time, result location, and a two-line "what this shows / does not show".
- Plots: matplotlib, readable labels, saved as PNG in `results/`, small file size.

## Memory (24 GB Mac, several sessions in parallel)
- Get the device only through `phase2.utils.get_device()`: on MPS it caps the allocator (`MPS_MEMORY_FRACTION`, default 0.25). Without the cap, batch sizes that change per step made one RL run hold 5.5 GB for a 1.86M-parameter model. In training loops whose batch size varies, also call `torch.mps.empty_cache()` once per step.
- Parallel runs are fine. A few minutes after launching, check `footprint <pid> | head -8`: if `IOAccelerator` is far above weights + optimizer + batch, fix that before launching more.

## Honesty rules
- No novelty claims. When a result contradicts a paper, report it and list likely reasons (subset size, settings, hardware), do not tune until it matches.
- Report intervals (bootstrap or across seeds) when comparing methods.
- Keep failed runs and negative results in the summaries.

## Commits
Small, descriptive commits per step (e.g. `phase1: instrument sampler to record commit step and confidence`). Do not push without the user's OK.
