# 1.0 Environment and speed (LLaDA-8B-Instruct on MPS)

- Model: `GSAI-ML/LLaDA-8B-Instruct` @ `08b83a6feb34df1a6011b80c3c00c7563e963b07`, dtype bfloat16, device `mps`
- Machine: Apple M5, 25.8 GB unified memory; macOS-27.0.1-arm64-arm-64bit
- python 3.12.13, torch 2.15.0, transformers 4.38.2
- `iogpu.wired_limit_mb` = 0 (macOS default, not changed). MPS recommended max working set: 19.1 GB
- Concurrent load: a Docker Desktop VM (~8.2 GB resident, user's long-running containers, left untouched), VS Code, and intermittently another job of this repo (phase-2 toy model training, ~1M params). With the 16 GB model resident, the system sits at ~22-24 GB used of 25.8 GB and ~15 GB of swap is in use, so timings are those of a memory-tight, shared machine.
- Failed first attempt (recorded for honesty): the first smoke run loaded with `.to("mps")` after a CPU load; a leftover killed process still held 16 GB and the next run thrashed swap (first forward pass 142 s). After that process exited and loading switched to `device_map={"": "mps"}` (no CPU copy), loading takes ~9 s and memory is stable.
- Load time: 8.9 s (weights already in local HF cache)

## Memory (GB)

| when | process RSS | MPS driver allocated | MPS current allocated | system used | swap used |
|---|---|---|---|---|---|
| before load | 0.2 | 0.0 | 0.0 | 7.4 | 15.5 |
| after load | 0.2 | 17.1 | 16.0 | 22.4 | 19.2 |
| after smoke gen | 0.2 | 17.1 | 16.0 | 24.4 | 15.2 |
| after timing | 0.0 | 17.1 | 16.0 | 24.6 | 15.1 |

## Smoke test

GSM8K test[0], d1 zero-shot prompt, `gen_length=128, steps=64, block_length=32`, greedy, `confidence` (= LLaDA `low_confidence`).
- prompt tokens: 148; forward passes: 64
- total: 56.1 s; **0.88 s per denoising step**
- reference answer 18.0; extracted 18.0; correct: True

```
Janet's ducks lay 16 eggs per day. She eats 3 eggs for breakfast and bakes 4 eggs for her friends, which means she uses a total of \(3 + 4 = 7\) eggs per day. Therefore, she has \(16 - 7 = 9\) eggs left to sell. She sells each egg for $2, so she makes \(9 \times 2 = 18\) dollars per day.
</reasoning>
<answer>
\boxed{18}
</answer>
```

## Seconds per denoising step vs sequence length (8 steps each, batch 1)

| task prompt | prompt tokens | gen_length | total seq | s/step |
|---|---|---|---|---|
| sudoku | 220 | 64 | 284 | 1.06 |
| sudoku | 220 | 128 | 348 | 1.18 |
| countdown | 156 | 64 | 220 | 0.77 |
| countdown | 156 | 128 | 284 | 1.30 |
| gsm8k | 148 | 128 | 276 | 1.33 |
| gsm8k | 148 | 256 | 404 | 1.74 |

Notes:
- The timing rows include the MPS warm-up for each new sequence length (the first 1-3 forward passes at a new shape are several times slower), so they overstate the steady-state cost. A separate measurement on the Sudoku prompt (284 tokens) gave 1.1-1.4 s per forward pass warm; a short Countdown grid smoke run gave ~0.9 s/step. A forward pass at 256 random tokens measured 0.38 s once while the machine was otherwise idle, so contention/memory pressure roughly doubles to triples the cost here.
- MPS is not fully deterministic; numbers are single measurements, not averages over runs.

## Sizing decision for 1.2
Using ~1.3 s/step (Sudoku/Countdown, gen_length 64) and ~1.4 s/step (GSM8K, gen_length 128): one example over the full grid (4 strategies x steps {gen/4, gen/2, gen}) costs 7 x gen_length forward passes = 448 passes (~10 min) for Sudoku/Countdown and 896 passes (~21 min) for GSM8K at gen_length 128. GSM8K at gen_length 256 would be ~50 min per example. To keep total Phase 1 compute to ~6 h we run 12 Sudoku, 12 Countdown and 5 GSM8K examples, GSM8K at gen_length 128 (see experiments/1.2/README.md).
