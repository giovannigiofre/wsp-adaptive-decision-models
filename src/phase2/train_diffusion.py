"""Masked-diffusion ("SFT") training of the tiny dLLM on 4x4 Sudoku, LLaDA/MDLM style.

Per example: givens always visible; t ~ U(t_min, 1); each non-given cell masked with prob t;
loss = sum over masked cells of CE / t, divided by the 16 cells, averaged over the batch.

PYTHONPATH=src python -m phase2.train_diffusion --config configs/phase2.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import PuzzleSet, build_datasets, digits_to_tokens
from .evaluate import solve_rate
from .model import build_model, n_params
from .utils import CACHE, apply_overrides, env_record, get_device, load_cfg, seed_all


def diffusion_loss(model, ps: PuzzleSet, t_min: float, gen: torch.Generator | None = None) -> torch.Tensor:
    x0, given = ps.solutions, ps.puzzles != 0
    B = x0.shape[0]
    dev = x0.device
    t = t_min + (1 - t_min) * torch.rand(B, 1, device=dev, generator=gen)
    mask = (~given) & (torch.rand(B, 16, device=dev, generator=gen) < t)
    logits = model(digits_to_tokens(torch.where(mask, torch.zeros_like(x0), x0)))
    ce = F.cross_entropy(logits.reshape(-1, 4), (x0 - 1).reshape(-1), reduction="none").view(B, 16)
    return ((ce * mask).sum(1) / t.squeeze(1) / 16).mean()


@torch.no_grad()
def eval_loss(model, ps: PuzzleSet, t_min: float, reps: int = 4) -> float:
    model.eval()
    gen = torch.Generator(device=ps.puzzles.device).manual_seed(123)
    v = sum(diffusion_loss(model, ps, t_min, gen).item() for _ in range(reps)) / reps
    model.train()
    return v


def eval_steps(max_steps: int, dense_until: int, dense_every: int, every: int) -> set[int]:
    s = set(range(dense_every, dense_until + 1, dense_every)) | set(range(every, max_steps + 1, every))
    return s | {max_steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase2.yaml")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/phase2")
    ap.add_argument("--ckpt-dir", default=str(CACHE / "sft"))
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key.sub=value")
    a = ap.parse_args()
    cfg = apply_overrides(load_cfg(a.config), a.set)
    tc = cfg["diffusion_training"]
    max_steps = a.max_steps or tc["max_steps"]
    dev = get_device(a.device or cfg["runtime"]["device"])
    seed = tc.get("seed", 0)
    seed_all(seed)
    ds = {k: v.to(dev) for k, v in build_datasets(cfg, cfg["data"].get("split_seed", 0)).items()}
    train, val = ds["train"], ds["val"]
    train_probe = train.subset(torch.randperm(len(train), device=dev)[:len(val)])
    model = build_model(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc.get("weight_decay", 0.01))
    warm = tc.get("warmup_steps", 200)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, max_steps) / max_steps)))
    ev = eval_steps(max_steps, tc.get("dense_eval_until", 3000), tc.get("dense_eval_every", 100), tc["eval_every"])
    ev_strategy, ev_steps = tc.get("eval_strategy", "confidence"), tc.get("eval_sampler_steps", 8)
    out, ck = Path(a.out), Path(a.ckpt_dir)
    out.mkdir(parents=True, exist_ok=True)
    ck.mkdir(parents=True, exist_ok=True)
    log_path = out / "sft_training.csv"
    fields = ["step", "train_loss", "val_loss", "val_solve", "train_solve", "lr", "elapsed_s"]
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(fields)
    print(f"params={n_params(model)} device={dev} train={len(train)} val={len(val)}", flush=True)
    t0, run_loss, n_loss = time.time(), 0.0, 0
    model.train()
    for step in range(1, max_steps + 1):
        idx = torch.randint(0, len(train), (tc["batch_size"],), device=dev)
        loss = diffusion_loss(model, train.subset(idx), tc["t_min"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        run_loss += loss.item(); n_loss += 1
        if step in ev:
            row = {"step": step, "train_loss": run_loss / n_loss, "val_loss": eval_loss(model, val, tc["t_min"]),
                   "val_solve": solve_rate(model, val, ev_steps, ev_strategy),
                   "train_solve": solve_rate(model, train_probe, ev_steps, ev_strategy),
                   "lr": sched.get_last_lr()[0], "elapsed_s": time.time() - t0}
            with open(log_path, "a", newline="") as f:
                csv.DictWriter(f, fields).writerow(row)
            torch.save(model.state_dict(), ck / f"step_{step:06d}.pt")
            run_loss, n_loss = 0.0, 0
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
    wall = time.time() - t0
    shutil.copy(ck / f"step_{max_steps:06d}.pt", ck / "sft_full.pt")
    # Pick the weak "SFT" checkpoint on *val* solve rate (test is never used for selection).
    import pandas as pd
    df = pd.read_csv(log_path)
    lo, hi = tc["target_sft_solve_rate"]
    cand = df[(df.val_solve >= lo) & (df.val_solve <= hi)]
    target = (lo + hi) / 2
    if len(cand):
        pick = cand.iloc[(cand.val_solve - target).abs().argsort().iloc[0]]
        rule = f"val solve rate in [{lo}, {hi}], closest to midpoint {target}"
    else:
        pick = df.iloc[(df.val_solve - target).abs().argsort().iloc[0]]
        rule = f"no checkpoint in [{lo}, {hi}]; took the one closest to {target}"
    shutil.copy(ck / f"step_{int(pick.step):06d}.pt", ck / "sft_weak.pt")
    sel = {"weak_step": int(pick.step), "weak_val_solve": float(pick.val_solve), "rule": rule,
           "full_step": max_steps, "full_val_solve": float(df.iloc[-1].val_solve),
           "eval_sampler": f"{ev_strategy}, S={ev_steps}, greedy", "n_params": n_params(model),
           "wall_clock_s": wall, "overrides": a.set, "env": env_record(dev), "seed": seed, "ckpt_dir": str(ck)}
    json.dump(sel, open(out / "sft_selection.json", "w"), indent=2)
    print(json.dumps(sel, indent=2))


if __name__ == "__main__":
    main()
