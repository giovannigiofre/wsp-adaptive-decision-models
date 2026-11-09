# 1.1 Visualise unmasking order

- **Goal:** see in which order LLaDA commits answer tokens under its default `low_confidence` remasking compared with `random`, as GIFs (decoded text after each step) and heatmaps (commit step and commit confidence per position).
- **Command:** `HF_HOME=~/dllm-cache/hf HF_HUB_OFFLINE=1 ~/dllm-cache/venv/bin/python -m src.phase1.unmasking --n_per_task 3 --gen_length 128 --steps 64` (stage 1 of `scripts/run_phase1.sh`)
- **Config:** `configs/phase1.yaml`. 3 prompts per task (Sudoku 4x4, Countdown-3, GSM8K test), subsampled with seed 43. gen_length 128, 64 steps (2 tokens per step), block_length 32, temperature 0, no CFG.
- **Hardware:** Apple M5, 24 GB, MPS. Shared with the Docker VM and the phase-2 training job (see `results/phase1/env_and_speed.md`).
- **Software / dtype:** torch 2.15.0, transformers 4.38.2. bf16 weights; scores in fp32.
- **Seed:** 42. The `random` strategy uses a CPU generator seeded with 42 for every prompt, so all 9 random runs share the same random order (within-block Spearman rho = -0.18 each time). The 1.2 grid seeds each example separately.
- **Wall-clock:** 25 min for 18 generations (54-101 s each).
- **Results:** `results/phase1/unmasking/`: `<id>_{low_confidence,random}.gif`, `<id>_{...}.json` (commit_step, commit_conf, tokens, text, score), `<id>_heatmap.png`, `runs.csv`.

## Outcome
- `low_confidence` fills each 32-token block roughly left to right. The mean within-block Spearman rho between position and commit step is 0.64 (per prompt 0.37-0.86). The two closing end-of-text tokens at the far end of the last block are usually committed in that block's first 1-4 steps (8 of 9 prompts) with confidence >= 0.94. The answers otherwise fill all 128 tokens, so there is no long padding tail at this length.
- `random` scatters commits inside the current block and only respects block order. Its text is often broken until late steps.
- Scores on these 9 prompts (single runs, not a benchmark):
  - low_confidence: GSM8K 3/3, Countdown 1/3, Sudoku cell accuracy 0.00-0.25.
  - random: GSM8K 1/3, Countdown 0/3, Sudoku cell accuracy 0.00-0.38.

**Shows:** with confidence-based remasking, LLaDA's decoding inside a block is mostly but not strictly left to right (an "almost autoregressive" order), whereas random order produces visibly incoherent intermediate text.
**Does not show:** any accuracy difference with statistical support (n = 3 per task). That is what 1.2 is for. It also does not cover gen lengths other than 128.
