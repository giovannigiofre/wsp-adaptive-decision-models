# Playing with diffusion language models on a MacBook

A hands-on experiment notebook on **masked diffusion language models (dLLMs)**, run entirely on a laptop (Apple Silicon, 24 GB unified memory, no external GPUs).

Autoregressive LLMs write one token at a time, left to right. Masked dLLMs start from a fully masked answer and *unmask* it over several steps, many tokens in parallel. That changes a lot of things we take for granted: generation order, how much compute an answer costs, what "confidence" means, and how reinforcement learning (RL) can be applied. This repo explores those questions with small, honest, reproducible experiments.

This is **not** a research paper and makes no novelty claims. The goal is to learn by building, measure carefully, and report what works and what does not.

## Two phases

| Phase | What | Where it runs | Doc |
|---|---|---|---|
| 1 | Explore a real dLLM (LLaDA-8B-Instruct): visualise unmasking order, compare unmasking strategies and step budgets, check whether token confidence predicts errors | Inference only, MPS | [docs/PHASE1_LLADA_ON_MAC.md](docs/PHASE1_LLADA_ON_MAC.md) |
| 2 | Train a tiny masked diffusion model from scratch on 4×4 Sudoku, then fine-tune it with RL using a hand-written implementation of the wd1 weighted policy objective; compare accuracy and confidence before/after RL | Training, MPS | [docs/PHASE2_TOY_SUDOKU_DLLM.md](docs/PHASE2_TOY_SUDOKU_DLLM.md) |

Overall plan, timeline and deliverables: [docs/PLAN.md](docs/PLAN.md). Reading list: [docs/REFERENCES.md](docs/REFERENCES.md).

## Status
Phase 1 and phase 2 have both run end to end (phase 1 on pilot-sized samples). Start with [results/REPORT_SIMPLE.md](results/REPORT_SIMPLE.md) for a plain-language tour or [results/REPORT.md](results/REPORT.md) for the technical write-up, then the executed notebooks `notebooks/phase1_llada.ipynb` and `notebooks/phase2_sudoku_wd1.ipynb`.

## Structure
- `docs/` plan, phase specs, references. `docs/archive/wsp/` keeps the earlier WSP adaptive-decision-model notes.
- `configs/` one YAML per phase.
- `src/phase1/`, `src/phase2/` implementation.
- `experiments/` one folder per experiment ID with the exact command and config used.
- `notebooks/` executed walkthroughs per phase (load committed results, no 8B model reload).
- `results/` compact summaries (CSV, small PNG/GIF). No weights, no large logs.
- `scripts/` environment utilities.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/inspect_environment.py
```

## Hardware note
Everything must run on a MacBook (Apple Silicon, 24 GB). Phase 1 uses an 8B model in bf16 (~16 GB of weights), so memory is tight: see the memory section in the Phase 1 doc. Every result records hardware, torch version, dtype, seed, and wall-clock time.

## License
Original code and docs are MIT-licensed. Third-party models, datasets and code keep their own licenses (LLaDA, d1, wd1, GSM8K).
