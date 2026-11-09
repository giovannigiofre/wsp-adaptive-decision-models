# 2.1 Tiny dLLM + masked-diffusion training (the "SFT" stage)

**Goal:** train a ~0.8M-parameter bidirectional transformer on 4×4 Sudoku with the LLaDA/MDLM masked-diffusion loss, and pick a deliberately weak checkpoint (val solve rate 30–60%) to start RL from.

**Command**
```
PYTHONPATH=src python -m phase2.train_diffusion --config configs/phase2.yaml
PYTHONPATH=src python -m phase2.plots sft
```

- Config: `configs/phase2.yaml` (`model:` and `diffusion_training:`). 4 layers, d_model 128, 4 heads, FF 512, dropout 0.1, pre-LN; learned 16-position embedding **plus learned row / column / 2×2-box embeddings** (`pos_embedding: structured`). 798,212 parameters. AdamW lr 3e-4, wd 0.01, 200 warm-up steps then cosine to 0, batch 256, 20,000 steps, grad-clip 1.0, t ~ U(0.05, 1), loss = Σ_masked CE / t / 16, batch mean.
- Hardware: Apple M5, 24 GB, macOS 27.0.1. Python 3.12.13, torch 2.15.0, device **MPS** (measured: a training step at batch 256 takes 20 ms on MPS vs 71 ms on CPU), fp32. Seed 0 (MPS is not bit-deterministic).
- Wall-clock: 778 s for 20k steps incl. ~120 evaluations. It shared the GPU with a concurrent LLaDA-8B job (phase 1), so the timing is pessimistic. The first attempt below (same model size, fewer evaluations) took 518 s.
- Results: `results/phase2/sft_training.csv` (every 10 steps up to 1,000, then every 500), `results/phase2/sft_selection.json`, `results/phase2/sft_training_curve.png`. Checkpoints (local only): `~/dllm-cache/phase2/sft/{sft_weak,sft_full,step_*}.pt`.

**Choosing the weak checkpoint.** Every evaluation saves a checkpoint and logs the val solve rate (305 val puzzles from the 44 held-out solution grids; greedy, confidence strategy, S = 8). After training, `sft_weak.pt` = the checkpoint whose val solve rate is in [0.30, 0.60] and closest to 0.45. Selection uses val only, never test. Result: **step 120, val 0.42** (test 0.48 at S = 8). The fully trained model (`sft_full.pt`, step 20,000) reaches val 1.00 and is kept as the upper reference.

The window is narrow: val solve rate goes 0.11 → 0.33 → 0.42 → 0.57 → 0.70 at steps 100–140 (still inside warm-up) and is ≥ 0.98 by step 190. So the "weak" model is a snapshot taken mid-transition. It is weak because it is under-trained, not because it is small.

**Failed attempt first (kept).** With only the learned 16-position embedding (the spec's default), training plateaued: val solve 0.14 after 20k steps (one-shot cell accuracy 0.65; 32% of k = 6 puzzles solved and 0% at k = 12). A pilot at lr 1e-3 was no better (val 0.10 at 3.3k steps, then stopped). Adding row/column/box embeddings (the spec's "optionally a 2D position embedding") solved val by step 200. Logs: `results/phase2/sft_attempts/`. This is a one-seed observation. Longer training or other hyper-parameters might have let the plain model escape the plateau; I did not search further.

**Shows:** masked-diffusion training on 4×4 Sudoku generalises to unseen solution grids once the model is told which row, column and box a cell belongs to. It also gives a reproducible way to get a 30–60% starting point.
**Does not show:** that the plain-position model cannot learn the task. It also does not show that the weak checkpoint resembles a weak *large* model: it is a mid-transition snapshot.
