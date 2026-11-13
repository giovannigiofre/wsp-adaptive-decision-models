# Playing with diffusion language models on a MacBook

A hands-on experiment notebook on **masked diffusion language models (dLLMs)**, run entirely on a laptop (Apple Silicon, 24 GB unified memory, no external GPUs).

Autoregressive LLMs write one token at a time, left to right. Masked dLLMs start from a fully masked answer and *unmask* it over several steps, many tokens in parallel. That changes a lot of things we take for granted: generation order, how much compute an answer costs, what "confidence" means, and how reinforcement learning (RL) can be applied. This repo explores those questions with small, honest, reproducible experiments.

This is **not** a research paper and makes no novelty claims. The goal is to learn by building, measure carefully, and report what works and what does not.

## Phases

| Phase | What | Where it runs | Doc |
|---|---|---|---|
| 1 | Explore a real dLLM (LLaDA-8B-Instruct): visualise unmasking order, compare unmasking strategies and step budgets, check whether token confidence predicts errors | Inference only, MPS | [docs/PHASE1_LLADA_ON_MAC.md](docs/PHASE1_LLADA_ON_MAC.md) |
| 2 | Train a tiny masked diffusion model from scratch on 4×4 Sudoku, then fine-tune it with RL using a hand-written implementation of the wd1 weighted policy objective; compare accuracy and confidence before/after RL | Training, MPS | [docs/PHASE2_TOY_SUDOKU_DLLM.md](docs/PHASE2_TOY_SUDOKU_DLLM.md) |
| 3 | Use the toy model as an RL playground: take wd1 apart, compare it with simpler objectives, reward honest confidence, learn the unmasking order | Training, MPS | [docs/PHASE3_RL_PLAYGROUND.md](docs/PHASE3_RL_PLAYGROUND.md) |
| 4 | Scale the toy to 9×9 Sudoku (1.86M parameters, 100k generated puzzles): redo the RL comparison where sampler steps matter, look at where rollouts fail, and train with no stored solutions (printed digits only, then a rule checker) | Training, MPS | [docs/PHASE4_SUDOKU9_DLLM.md](docs/PHASE4_SUDOKU9_DLLM.md) |

Overall plan, timeline and deliverables: [docs/PLAN.md](docs/PLAN.md). Reading list: [docs/REFERENCES.md](docs/REFERENCES.md).

## Status
**Finished (November 2026).** All four phases have run end to end (phase 1 on pilot-sized samples) and no further experiments are planned.

Where to start:
- [results/REPORT_SIMPLE.md](results/REPORT_SIMPLE.md): a plain-language tour of all four phases.
- [results/REPORT.md](results/REPORT.md): the technical write-up.
- Per-phase detail: `results/phase1/SUMMARY.md` to [results/phase4/SUMMARY.md](results/phase4/SUMMARY.md), and one README per experiment in `experiments/<id>/` with the exact command, the predictions written before the run and how they turned out.
- Executed notebooks: `notebooks/phase1_llada.ipynb`, `notebooks/phase2_sudoku_wd1.ipynb`, `notebooks/phase3_rl_playground.ipynb`, `notebooks/phase4_sudoku9.ipynb`.

## What came out, in five lines
1. LLaDA's default decoding is close to left-to-right, fewer denoising steps cost little at pilot size, and its token confidence can be misleading (phase 1).
2. On the 4×4 toy, a positive-only ablation beat the hand-written wd1 objective; the reason was mechanical (flat weights, an unbounded push-down term, one lucky starting snapshot) and plain training on the model's own valid outputs beat every RL objective (phases 2 and 3).
3. On 9×9 the order and number of unmasking steps matter a lot, and failures are one or two early wrong cells followed by cells that fit them (phase 4).
4. On 9×9 no RL objective beat a plain learning-rate drop: the gain first read as "self-training helps" disappeared once the starting model had been trained with a low learning rate at the end, and from that model plain wd1 loses 0.08 at 4 sampler steps where its positive-only version loses 0.01 (phase 4).
5. A model trained only on the printed digits of the puzzles, with no solutions and no checker, gets within 0.14 / 0.07 / 0.00 of the supervised one on hard puzzles at 4 / 16 / 81 sampler steps; a loop that harvests its own checker-valid grids starts faster and ends no higher, apart from 0.03 at 4 steps from a last short pass on its own valid grids (phase 4).

These are results on one laptop, small models and few seeds. The limits are listed in each summary.

## Structure
- `docs/` plan, phase specs, references, and four literature memos written during phase 4 (`RL_DLLM_SURVEY.md`, `RL_DLLM_DIRECTIONS.md`, `SELF_LEARNING_FROM_CHECKER.md`, `SAMPLE_EFFICIENT_SELF_LEARNING.md`). `docs/archive/wsp/` keeps the earlier WSP adaptive-decision-model notes.
- `configs/` one YAML per phase.
- `src/phase1/` to `src/phase4/` implementation; `tests/` unit tests (scorers, samplers, Sudoku checkers, wd1 loss sign).
- `experiments/` one folder per experiment ID with the exact command and config used.
- `notebooks/` executed walkthroughs per phase (load committed results, no 8B model reload).
- `results/` compact summaries (CSV, small PNG/GIF). No weights, no large logs.
- `scripts/` environment utilities and one launcher per phase (`run_phase1.sh` to `run_phase4.sh`).

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
