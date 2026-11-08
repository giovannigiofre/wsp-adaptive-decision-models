"""Step 1.0: load LLaDA on MPS, one GSM8K smoke test, memory + seconds per denoising step.

Usage: python -m src.phase1.smoke [--out results/phase1/env_and_speed.md] [--device mps]
Also times a few steps at the sequence lengths used later (Sudoku/Countdown/GSM8K prompts)
so the 1.2 grid can be sized.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from . import tasks
from .llada_generate import generate_instrumented
from .load import REPO, env_info, load_config, load_model, memory_report, seed_everything


def sync(dev):
    if dev.type == "mps":
        torch.mps.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results/phase1/env_and_speed.md"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--timing_steps", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config()
    seed_everything(cfg["seed"])
    mem0 = memory_report()
    model, tok, dev, load_s = load_model(cfg, args.device)
    mem_load = memory_report()
    print(f"loaded on {dev} in {load_s:.1f}s; mem {mem_load}", flush=True)

    # --- smoke test: one GSM8K prompt (d1 format), gen 128, steps 64, block 32
    item = tasks.load_task("gsm8k")[0]
    prompt = tasks.build_prompt(tok, "gsm8k", item["question"])
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev)
    sync(dev)
    t0 = time.time()
    with torch.inference_mode():
        out = generate_instrumented(model, ids, steps=64, gen_length=128, block_length=32,
                                    remasking="confidence")
    sync(dev)
    total = time.time() - t0
    mem_gen = memory_report()
    text = tok.decode(out["x"][0, ids.shape[1]:], skip_special_tokens=True)
    gen_raw = tok.decode(out["x"][0, ids.shape[1]:], skip_special_tokens=False)
    sc = tasks.score("gsm8k", gen_raw, item)
    print(text, flush=True)
    print(f"total {total:.1f}s, {total / out['nfe']:.3f}s/step, score {sc}", flush=True)

    # --- per-step timing at the prompt lengths used in 1.2
    timing = []
    for task, gl in [("sudoku", 64), ("sudoku", 128), ("countdown", 64), ("countdown", 128),
                     ("gsm8k", 128), ("gsm8k", 256)]:
        it = tasks.load_task(task)[0]
        pid = tok(tasks.build_prompt(tok, task, it["question"]), add_special_tokens=False,
                  return_tensors="pt")["input_ids"].to(dev)
        n = args.timing_steps
        sync(dev)
        t0 = time.time()
        with torch.inference_mode():
            generate_instrumented(model, pid, steps=n, gen_length=gl, block_length=gl, remasking="confidence")
        sync(dev)
        dt = (time.time() - t0) / n
        timing.append({"task": task, "prompt_len": pid.shape[1], "gen_length": gl,
                       "seq_len": pid.shape[1] + gl, "s_per_step": dt})
        print(timing[-1], flush=True)
    mem_end = memory_report()

    env = env_info()
    lines = [
        "# 1.0 Environment and speed (LLaDA-8B-Instruct on MPS)", "",
        f"- Model: `{cfg['model']['id']}` @ `{cfg['model']['revision']}`, dtype {cfg['model']['dtype']}, device `{dev}`",
        f"- Machine: Apple M5, {env['ram_gb']:.1f} GB unified memory; {env['platform']}",
        f"- python {env['python']}, torch {env['torch']}, transformers {env['transformers']}",
        f"- `iogpu.wired_limit_mb` = 0 (macOS default, not changed). MPS recommended max working set: "
        f"{mem_load.get('mps_recommended_max_gb', float('nan')):.1f} GB",
        "- Concurrent load: another job (phase-2 toy model training, ~1M params) was running on the same Mac "
        "during these measurements, so timings include some GPU contention.",
        f"- Load time: {load_s:.1f} s (weights already in local HF cache)", "",
        "## Memory (GB)", "", "| when | process RSS | MPS driver allocated | MPS current allocated | system used | swap used |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in [("before load", mem0), ("after load", mem_load), ("after smoke gen", mem_gen), ("after timing", mem_end)]:
        lines.append(f"| {name} | {m['rss_gb']:.1f} | {m.get('mps_driver_allocated_gb', 0):.1f} | "
                     f"{m.get('mps_current_allocated_gb', 0):.1f} | {m['sys_used_gb']:.1f} | {m['swap_used_gb']:.1f} |")
    lines += ["", "## Smoke test", "",
              f"GSM8K test[0], d1 zero-shot prompt, `gen_length=128, steps=64, block_length=32`, greedy, `confidence` (= LLaDA `low_confidence`).",
              f"- prompt tokens: {ids.shape[1]}; forward passes: {out['nfe']}",
              f"- total: {total:.1f} s; **{total / out['nfe']:.2f} s per denoising step**",
              f"- reference answer {item['answer']}; extracted {sc['extracted']}; correct: {sc['correct']}", "",
              "```", text.strip()[:1500], "```", "",
              f"## Seconds per denoising step vs sequence length ({args.timing_steps} steps each, batch 1)", "",
              "| task prompt | prompt tokens | gen_length | total seq | s/step |", "|---|---|---|---|---|"]
    for t in timing:
        lines.append(f"| {t['task']} | {t['prompt_len']} | {t['gen_length']} | {t['seq_len']} | {t['s_per_step']:.2f} |")
    lines += ["", "Notes: MPS is not fully deterministic; numbers are single measurements, not averages over runs."]
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(REPO / "results/phase1/speed.json", "w") as f:
        json.dump({"smoke_total_s": total, "smoke_s_per_step": total / out["nfe"], "timing": timing,
                   "load_s": load_s}, f, indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
