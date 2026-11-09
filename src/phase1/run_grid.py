"""Step 1.2 grid runner: strategies x step budgets x task subset. Resumable.

One CSV row per (example, strategy, steps) appended to results/phase1/raw/<task>.csv, plus one
JSONL record with per-token commit step/confidence to results/phase1/tokens/<task>.jsonl
(used by 1.3). Examples are the outer loop so a partial run stays balanced across configs.

Usage:
  python -m src.phase1.run_grid --task sudoku --n 25 --gen_length 128
  python -m src.phase1.run_grid --task gsm8k --n 12 --gen_length 256 --strategies confidence,random
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
import torch

from . import tasks
from .llada_generate import STRATEGIES, generate_instrumented
from .load import REPO, load_config, load_model, seed_everything

EOS_IDS = (126081, 126348)  # <|endoftext|>, <|eot_id|>
FIELDS = ["task", "id", "strategy", "steps", "gen_length", "block_length", "score", "correct", "extracted",
          "nfe", "wall_s", "n_content_tokens", "mean_conf", "min_conf", "ans_mean_conf", "ans_min_conf",
          "generation"]


def token_offsets(tok, ids):
    pieces = [tok.decode([int(t)], skip_special_tokens=False) for t in ids]
    starts = np.cumsum([0] + [len(p) for p in pieces[:-1]])
    return "".join(pieces), starts, pieces


def char_to_token(starts, c):
    return int(np.searchsorted(starts, c, side="right") - 1)


def answer_char_span(task, text, sc):
    if task == "sudoku":
        return sc.get("span")
    m = None
    for m in __import__("re").finditer(r"\\boxed\{", text):
        pass
    if m is not None:
        end = text.find("}", m.end())
        return (m.start(), end + 1 if end >= 0 else len(text))
    i = text.find("<answer>")
    return (i, len(text)) if i >= 0 else None


def analyse(task, tok, item, ans_ids, commit_step, commit_conf):
    text, starts, _ = token_offsets(tok, ans_ids)
    sc = tasks.score(task, text, item)
    eos = [i for i, t in enumerate(ans_ids) if int(t) in EOS_IDS]
    n_content = eos[0] if eos else len(ans_ids)
    cc = commit_conf[:max(n_content, 1)]
    row = {"score": sc["score"], "correct": int(sc["correct"]), "extracted": str(sc["extracted"])[:80],
           "n_content_tokens": n_content, "mean_conf": float(np.mean(cc)), "min_conf": float(np.min(cc))}
    span = answer_char_span(task, text, sc)
    tok_span = None
    if span is not None and span[1] > span[0]:
        a, b = char_to_token(starts, span[0]), char_to_token(starts, span[1] - 1)
        tok_span = (a, b + 1)
        row["ans_mean_conf"] = float(np.mean(commit_conf[a:b + 1]))
        row["ans_min_conf"] = float(np.min(commit_conf[a:b + 1]))
    else:
        row["ans_mean_conf"] = row["ans_min_conf"] = float("nan")
    cells = None
    if task == "sudoku" and sc.get("span"):
        # map each empty cell of the extracted 16-digit string back to the token that produced it
        s0 = sc["span"][0]
        raw = text[s0:sc["span"][1]]
        digit_pos = [s0 + i for i, ch in enumerate(raw) if not ch.isspace()][:16]
        cells = []
        for k, cell in enumerate(sc["empty_idx"]):
            if cell < len(digit_pos):
                ti = char_to_token(starts, digit_pos[cell])
                cells.append([cell, ti, float(commit_conf[ti]), int(commit_step[ti]), int(sc["cell_correct"][k])])
    return row, text, tok_span, cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["sudoku", "countdown", "gsm8k"])
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--gen_length", type=int, default=None)
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--steps_fractions", default=None)
    ap.add_argument("--tag", default="", help="suffix for output files (e.g. smoke)")
    ap.add_argument("--max_minutes", type=float, default=1e9, help="stop cleanly after this wall-clock")
    args = ap.parse_args()

    cfg = load_config()
    seed = cfg["seed"]
    seed_everything(seed)
    gl = args.gen_length or cfg["tasks"][args.task]["gen_length"]
    bl = cfg["sampler"]["block_length"]
    fracs = [float(f) for f in args.steps_fractions.split(",")] if args.steps_fractions else cfg["grid"]["steps_fraction"]
    nblk = gl // bl
    steps_list = sorted({max(nblk, int(round(f * gl / nblk)) * nblk) for f in fracs})
    strategies = args.strategies.split(",")

    items = tasks.subsample(tasks.load_task(args.task), args.n, seed)
    os.makedirs(REPO / "experiments/1.2", exist_ok=True)
    ids_path = REPO / "experiments/1.2/ids.json"
    ids = json.load(open(ids_path)) if ids_path.exists() else {}
    ids[args.task + args.tag] = [it["id"] for it in items]
    json.dump(ids, open(ids_path, "w"), indent=1)

    raw_dir, tok_dir = REPO / "results/phase1/raw", REPO / "results/phase1/tokens"
    raw_dir.mkdir(parents=True, exist_ok=True)
    tok_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / f"{args.task}{args.tag}.csv"
    jsonl_path = tok_dir / f"{args.task}{args.tag}.jsonl"
    done = set()
    if csv_path.exists():
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                done.add((r["id"], r["strategy"], int(r["steps"]), int(r["gen_length"])))
    todo = [(it, s, st) for it in items for s in strategies for st in steps_list
            if (it["id"], s, st, gl) not in done]
    print(f"{args.task}: {len(items)} examples x {strategies} x steps {steps_list}, gen {gl}; "
          f"{len(done)} done, {len(todo)} to do", flush=True)
    if not todo:
        return

    model, tok, dev, load_s = load_model(cfg)
    print(f"model loaded in {load_s:.0f}s on {dev}", flush=True)
    new_file = not csv_path.exists()
    fcsv = open(csv_path, "a", newline="")
    w = csv.DictWriter(fcsv, fieldnames=FIELDS)
    if new_file:
        w.writeheader()
    fj = open(jsonl_path, "a")
    t_start = time.time()
    prompt_cache = {}
    for k, (it, strat, steps) in enumerate(todo):
        if (time.time() - t_start) / 60 > args.max_minutes:
            print("max_minutes reached, stopping", flush=True)
            break
        if it["id"] not in prompt_cache:
            p = tasks.build_prompt(tok, args.task, it["question"])
            prompt_cache[it["id"]] = tok(p, add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev)
        ids_t = prompt_cache[it["id"]]
        ex_seed = seed + int(it["id"].split("-")[-1])
        torch.manual_seed(ex_seed)
        t0 = time.time()
        with torch.inference_mode():
            out = generate_instrumented(model, ids_t, steps=steps, gen_length=gl, block_length=bl,
                                        temperature=cfg["sampler"]["temperature"], cfg_scale=cfg["sampler"]["cfg_scale"],
                                        remasking=strat, seed=ex_seed)
        if dev.type == "mps":
            torch.mps.synchronize()
        wall = time.time() - t0
        ans_ids = out["x"][0, ids_t.shape[1]:].cpu().numpy()
        cs, cc = out["commit_step"][0].numpy(), out["commit_conf"][0].numpy()
        row, text, tok_span, cells = analyse(args.task, tok, it, ans_ids, cs, cc)
        row.update({"task": args.task, "id": it["id"], "strategy": strat, "steps": steps, "gen_length": gl,
                    "block_length": bl, "nfe": out["nfe"], "wall_s": round(wall, 2), "generation": text})
        w.writerow(row)
        fcsv.flush()
        fj.write(json.dumps({"task": args.task, "id": it["id"], "strategy": strat, "steps": steps, "gen_length": gl,
                             "correct": row["correct"], "score": row["score"], "n_content": row["n_content_tokens"],
                             "tokens": ans_ids.tolist(), "commit_step": cs.tolist(),
                             "commit_conf": [round(float(c), 5) for c in cc], "ans_tok_span": tok_span,
                             "cells": cells}) + "\n")
        fj.flush()
        el = time.time() - t_start
        print(f"[{k + 1}/{len(todo)}] {it['id']} {strat} steps={steps} score={row['score']:.2f} "
              f"wall={wall:.0f}s elapsed={el / 60:.1f}m eta={el / (k + 1) * (len(todo) - k - 1) / 60:.0f}m", flush=True)
    fcsv.close()
    fj.close()


if __name__ == "__main__":
    main()
