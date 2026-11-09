"""SFT on the puzzles of only N of the 200 train solutions, run to convergence (phase 3.1d).

Phase 2's weak start is an early snapshot of a model that would reach 100% a few steps later. This
gives a different kind of weak model: one that has converged on little labelled data. RL can then use
the validity checker on puzzles from all 200 train solutions, with no further labels.

Same model, loss, optimiser and schedule as phase2.train_diffusion; only the training set and the
number of steps change.

PYTHONPATH=src python -m phase3.sft_subset --n-solutions 10
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch

from phase2.data import build_datasets
from phase2.evaluate import solve_rate
from phase2.model import build_model
from phase2.train_diffusion import diffusion_loss, eval_loss
from phase2.utils import env_record, get_device, load_cfg, seed_all

from .rl import CACHE, git_commit, sha256

OUT = Path("results/phase3/3.1d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase3.yaml")
    ap.add_argument("--n-solutions", type=int, required=True)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    base = load_cfg(load_cfg(a.config)["base_config"])
    tc = base["diffusion_training"]
    dev = get_device(a.device or base["runtime"]["device"])
    name = f"sft_subset_n{a.n_solutions}_seed{a.seed}"
    ck = CACHE / "3.1d" / "sft"
    OUT.mkdir(parents=True, exist_ok=True)
    ck.mkdir(parents=True, exist_ok=True)
    ds = {k: v.to(dev) for k, v in build_datasets(base, base["data"].get("split_seed", 0)).items()}
    train, val = ds["train"], ds["val"]
    seed_all(a.seed)
    uniq = torch.unique(train.solutions, dim=0)
    chosen = uniq[torch.randperm(len(uniq))[:a.n_solutions].to(dev)]
    inside = (train.solutions[:, None, :] == chosen[None]).all(-1).any(1)
    sub, rest = train.subset(inside), train.subset(~inside)
    probe_in = sub.subset(torch.randperm(len(sub))[:len(val)].to(dev))
    probe_out = rest.subset(torch.randperm(len(rest))[:len(val)].to(dev))
    model = build_model(base).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc.get("weight_decay", 0.01))
    warm = tc.get("warmup_steps", 200)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    fields = ["step", "train_loss", "val_loss", "val_solve", "seen_solutions_solve", "unseen_train_solve",
              "elapsed_s"]
    rows, t0, run_loss, n_loss = [], time.time(), 0.0, 0
    print(f"{name}: {len(sub)} puzzles from {a.n_solutions} solutions", flush=True)
    model.train()
    for step in range(1, a.steps + 1):
        idx = torch.randint(0, len(sub), (tc["batch_size"],), device=dev)
        loss = diffusion_loss(model, sub.subset(idx), tc["t_min"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        run_loss += loss.item(); n_loss += 1
        if step % 250 == 0 or step == a.steps:
            rows.append({"step": step, "train_loss": run_loss / n_loss,
                         "val_loss": eval_loss(model, val, tc["t_min"]),
                         "val_solve": solve_rate(model, val, 8, "confidence"),
                         "seen_solutions_solve": solve_rate(model, probe_in, 8, "confidence"),
                         "unseen_train_solve": solve_rate(model, probe_out, 8, "confidence"),
                         "elapsed_s": time.time() - t0})
            run_loss, n_loss = 0.0, 0
            print(" ".join(f"{k}={v:.4g}" for k, v in rows[-1].items()), flush=True)
    with open(OUT / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows([{k: round(v, 4) for k, v in r.items()} for r in rows])
    path = ck / f"{name}.pt"
    torch.save(model.state_dict(), path)
    model.eval()
    late = [r["val_solve"] for r in rows if r["step"] > a.steps - 1000]
    summ = {"name": name, "n_solutions": a.n_solutions, "n_puzzles": len(sub), "steps": a.steps, "seed": a.seed,
            "final_val_solve": rows[-1]["val_solve"], "final_test_solve": solve_rate(model, ds["test"], 8, "confidence"),
            "val_solve_range_last_1000_steps": [min(late), max(late)],
            "final_seen_solutions_solve": rows[-1]["seen_solutions_solve"],
            "final_unseen_train_solve": rows[-1]["unseen_train_solve"], "checkpoint": str(path),
            "sha256": sha256(path), "code_commit": git_commit(), "wall_clock_s": time.time() - t0,
            "env": env_record(dev)}
    json.dump(summ, open(OUT / f"{name}.json", "w"), indent=2)
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
