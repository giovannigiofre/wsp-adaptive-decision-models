# 1.0 Setup, loading and speed

- **Goal:** load LLaDA-8B-Instruct in bf16 on MPS, run one GSM8K smoke test, measure memory and seconds per denoising step, and size the later experiments.
- **Command:** `HF_HOME=~/dllm-cache/hf HF_HUB_OFFLINE=1 ~/dllm-cache/venv/bin/python -m src.phase1.smoke`
- **Config:** `configs/phase1.yaml` (model revision `08b83a6f…`, LLaDA commit `9182493…`, d1 commit `faf85d6…`)
- **Hardware:** Apple M5, 24 GB unified memory (25.8 GB reported), macOS 27.0.1. Docker VM (~8 GB) and VS Code running; phase-2 job intermittently on the same GPU.
- **Software:** Python 3.12.13, torch 2.15.0 (MPS), transformers 4.38.2 (version required by the LLaDA README).
- **dtype:** bfloat16 weights; softmax/strategy scores computed in float32.
- **Seed:** 42 (`random`, `numpy`, `torch`); MPS is not fully deterministic.
- **Wall-clock:** model download ~75 min (16 GB at ~4 MB/s; the first attempt via the Xet backend stalled and was restarted with `HF_HUB_DISABLE_XET=1`); smoke script ~3 min (9 s load, 56 s smoke generation, ~2 min timing table).
- **Results:** `results/phase1/env_and_speed.md`, `results/phase1/speed.json`.

## Outcome
- Loads fine with `device_map={"": "mps"}`: 16.0 GB allocated on MPS, recommended max working set 19.1 GB; no system setting changed.
- Smoke test (GSM8K test[0], gen 128, steps 64, block 32): correct answer (18), 56 s total, **0.88 s / denoising step**.
- Steady-state cost on this shared, memory-tight machine is ~0.9-1.4 s per forward pass at 220-350 tokens, plus a one-off MPS warm-up per new sequence length.
- A first attempt thrashed swap because a killed process still held 16 GB; recorded in `env_and_speed.md`.

**Shows:** an 8B masked diffusion LM runs on a 24 GB MacBook in bf16 at about 1 s per denoising step, so ~1-2 minutes per 64-128 token answer.
**Does not show:** best achievable speed (other apps were using memory/GPU; no KV-cache or batching tricks), or numerical equivalence with CUDA runs.
