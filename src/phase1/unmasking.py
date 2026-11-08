"""Step 1.1: record the unmasking order on 3 Sudoku, 3 Countdown, 3 GSM8K prompts,
low_confidence (= `confidence`) vs `random`, same prompt. Writes heatmaps + GIFs + a CSV.

Usage: python -m src.phase1.unmasking [--n_per_task 3] [--gen_length 128] [--steps 64]
"""
from __future__ import annotations

import argparse
import csv
import json
import time

import numpy as np
import torch

from . import tasks
from .llada_generate import generate_instrumented
from .load import REPO, load_config, load_model, seed_everything
from .viz import commit_heatmap, decode_frames, text_gif

MASK_ID = 126336


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_per_task", type=int, default=3)
    ap.add_argument("--gen_length", type=int, default=128)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--tasks", default="sudoku,countdown,gsm8k")
    args = ap.parse_args()
    cfg = load_config()
    seed = cfg["seed"]
    seed_everything(seed)
    out_dir = REPO / "results/phase1/unmasking"
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tok, dev, _ = load_model(cfg)
    bl = cfg["sampler"]["block_length"]
    rows = []
    t_all = time.time()
    for task in args.tasks.split(","):
        # seed+1 so these prompts are (very likely) not the same ones as the first 1.2 examples
        items = tasks.subsample(tasks.load_task(task), args.n_per_task, seed + 1)
        for it in items:
            pid = tok(tasks.build_prompt(tok, task, it["question"]), add_special_tokens=False,
                      return_tensors="pt")["input_ids"].to(dev)
            steps_d, conf_d = {}, {}
            for strat in ["confidence", "random"]:
                t0 = time.time()
                with torch.inference_mode():
                    out = generate_instrumented(model, pid, steps=args.steps, gen_length=args.gen_length,
                                                block_length=bl, remasking=strat, record_frames=True, seed=seed)
                wall = time.time() - t0
                ans = out["x"][0, pid.shape[1]:].cpu().numpy()
                text = "".join(tok.decode([int(t)], skip_special_tokens=False) for t in ans)
                sc = tasks.score(task, text, it)
                name = "low_confidence" if strat == "confidence" else "random"
                steps_d[name] = out["commit_step"][0].numpy()
                conf_d[name] = out["commit_conf"][0].numpy()
                frames = decode_frames(tok, out["frames"], MASK_ID)
                text_gif(frames, out_dir / f"{it['id']}_{name}.gif", header=f"{it['id']} {name}")
                rows.append({"task": task, "id": it["id"], "remasking": name, "steps": args.steps,
                             "gen_length": args.gen_length, "score": sc["score"], "correct": int(sc["correct"]),
                             "wall_s": round(wall, 1), "mean_conf": float(np.mean(conf_d[name])),
                             "generation": text})
                with open(out_dir / f"{it['id']}_{name}.json", "w") as f:
                    json.dump({"commit_step": steps_d[name].tolist(),
                               "commit_conf": [round(float(c), 4) for c in conf_d[name]],
                               "tokens": ans.tolist(), "text": text, "score": sc["score"]}, f)
                print(f"{it['id']} {name} score={sc['score']:.2f} wall={wall:.0f}s", flush=True)
            commit_heatmap(steps_d, conf_d, f"{it['id']}  gen_length={args.gen_length} steps={args.steps} "
                           f"block={bl}", out_dir / f"{it['id']}_heatmap.png", block_length=bl)
    with open(out_dir / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"done in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
