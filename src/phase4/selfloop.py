"""Step 4.8b: self-training from a rule checker, no stored solutions (docs/SELF_LEARNING_FROM_CHECKER.md, stage A2).

Starts from the 4.8a model (trained on the givens only). Each round: (1) harvest: greedy 16-step rollouts on train
puzzles, the grids that pass the rule check go into a buffer keyed by puzzle; (2) `round_steps` training steps on
batches that mix buffer grids and the givens loss on puzzles not yet in the buffer. The stored solutions of the train
set are replaced by zeros; a buffer grid is only ever a rollout that `batch_validity` accepted.

Arms:
  givens     control: no harvest, the 4.8a loss continued
  harvest    buffer grids with the puzzle's own mask (givens visible, blanks masked with probability t)
  hindsight  half of the buffer rows as above, half with hindsight masks (any of the 81 cells masked with probability t)

    PYTHONPATH=src python -m phase4.selfloop --arm hindsight --launch-minutes 100
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from phase2.data import PuzzleSet, digits_to_tokens
from phase2.evaluate import run_sampler
from phase2.model import build_model, n_params
from phase2.utils import apply_overrides, env_record, get_device, load_cfg, seed_all

from .data import load_sets
from .generate import cache_dir
from .selflearn import without_solutions
from .train import band_eval


def mixed_loss(model, x0: torch.Tensor, always: torch.Tensor, maskable: torch.Tensor, t_min: float,
               gen: torch.Generator | None = None) -> torch.Tensor:
    """Masked cross-entropy / t. Per row: `x0` targets (digits, 0 = unknown), `always` cells that are always masked
    and carry no loss, `maskable` cells masked with probability t ~ U(t_min, 1) and scored when masked."""
    B, L = x0.shape
    t = t_min + (1 - t_min) * torch.rand(B, 1, device=x0.device, generator=gen)
    mask = maskable & (torch.rand(B, L, device=x0.device, generator=gen) < t)
    logits = model(digits_to_tokens(torch.where(mask | always, torch.zeros_like(x0), x0)))
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), (x0 - 1).clamp_min(0).reshape(-1), reduction="none").view(B, L)
    return ((ce * mask).sum(1) / t.squeeze(1) / L).mean()


def rows(puzzles: torch.Tensor, grids: torch.Tensor | None, kind: str):
    """(x0, always, maskable) for `givens` rows (targets = the givens), `own` rows (targets = a buffer grid, the
    puzzle's blanks maskable) and `hindsight` rows (targets = a buffer grid, every cell maskable)."""
    blank = puzzles == 0
    if kind == "givens":
        return puzzles, blank, ~blank
    if kind == "own":
        return grids, torch.zeros_like(blank), blank
    return grids, torch.zeros_like(blank), torch.ones_like(blank)


def batch_rows(arm: str, P: torch.Tensor, found: torch.Tensor, has: torch.Tensor, batch: int, givens_share: float):
    dev = P.device
    pick = lambda pool, n: pool[torch.randint(0, len(pool), (n,), device=dev)]           # noqa: E731
    solved, unsolved = has.nonzero().squeeze(1), (~has).nonzero().squeeze(1)
    if arm == "givens" or len(solved) == 0:
        return rows(P[torch.randint(0, len(P), (batch,), device=dev)], None, "givens")
    n_g = int(round(batch * givens_share)) if len(unsolved) else 0
    n_h = (batch - n_g) // 2 if arm == "hindsight" else 0
    parts = []
    for kind, n, pool in (("givens", n_g, unsolved), ("own", batch - n_g - n_h, solved), ("hindsight", n_h, solved)):
        if n:
            i = pick(pool, n)
            parts.append(rows(P[i], found[i], kind))
    return tuple(torch.cat(c) for c in zip(*parts))


@torch.no_grad()
def harvest(model, P: torch.Tensor, idx: torch.Tensor, found: torch.Tensor, has: torch.Tensor, steps: int) -> int:
    """Rollouts on puzzles `idx`; grids accepted by the rule check are written to the buffer. Returns how many."""
    ps = PuzzleSet(P[idx], torch.zeros_like(P[idx]), (P[idx] == 0).sum(1))
    out = run_sampler(model, ps, steps, "confidence", batch=1024)
    ok = out["solved"].to(P.device)
    found[idx[ok]] = out["grid"].to(P.device)[ok]
    has[idx[ok]] = True
    return int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase4.yaml")
    ap.add_argument("--size", default="medium")
    ap.add_argument("--arm", required=True, choices=["givens", "harvest", "hindsight"])
    ap.add_argument("--exp", default="4.8b")
    ap.add_argument("--name", default=None, help="run name for results and checkpoints; default: the arm")
    ap.add_argument("--init", default="4.8a/a1/state.pt", help="training state to start from, under the phase 4 cache")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--round-steps", type=int, default=1000)
    ap.add_argument("--harvest-steps", type=int, default=16)
    ap.add_argument("--harvest-block", type=int, default=15000, help="unsolved puzzles tried per round after the first")
    ap.add_argument("--givens-share", type=float, default=0.25)
    ap.add_argument("--launch-minutes", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = apply_overrides(load_cfg(a.config), a.set)
    cfg["model"].update(cfg["sizes"][a.size])
    tc = cfg["diffusion_training"]
    dev = get_device(a.device or cfg["runtime"]["device"])
    name = a.name or a.arm
    out, ck = Path("results/phase4") / a.exp, cache_dir(cfg) / a.exp / name
    out.mkdir(parents=True, exist_ok=True); ck.mkdir(parents=True, exist_ok=True)
    log_path, state_path, done_path = out / f"{name}.csv", ck / "state.pt", out / f"{name}.json"
    if done_path.exists():
        print(f"skip {a.arm}: already finished"); return
    sets, bands, names = load_sets(cfg, dev, ("train", "val"))
    P, val = without_solutions(sets["train"]).puzzles, sets["val"]
    tband = bands["train"].to(dev)
    fields = (["round", "step", "train_loss", "tried", "accepted"] + [f"buffer_{n}" for n in names]
              + [f"val_solve_{n}" for n in names] + [f"val4_solve_{n}" for n in names] + ["val_cell_acc", "elapsed_s"])
    seed_all(tc["seed"])
    model = build_model(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / tc["warmup_steps"]))
    found, has = torch.zeros_like(P), torch.zeros(len(P), dtype=torch.bool, device=dev)
    rnd, elapsed0 = 0, 0.0
    resumed = state_path.exists()
    st = torch.load(state_path if resumed else cache_dir(cfg) / a.init, map_location=dev)
    model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
    step = step_init = st["step"]
    if resumed:
        torch.set_rng_state(st["rng_cpu"].cpu())
        if dev.type == "mps":
            torch.mps.set_rng_state(st["rng_dev"].cpu())
        rnd, elapsed0, found, has, step_init = st["round"], st["elapsed_s"], st["found"], st["has"], st["step_init"]
        print(f"resumed {a.arm} at round {rnd}, step {step}", flush=True)
    else:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(fields)
    print(f"{a.arm}: params={n_params(model)} device={dev} train={len(P)} start step={step_init}", flush=True)
    t0 = time.time()
    while rnd < a.rounds:
        tried = accepted = 0
        if a.arm != "givens":
            idx = (~has).nonzero().squeeze(1)
            if rnd > 0 and len(idx) > a.harvest_block:
                idx = idx[torch.randperm(len(idx), device="cpu")[:a.harvest_block].to(dev)]
            tried, accepted = len(idx), harvest(model, P, idx, found, has, a.harvest_steps)
        model.train()
        run_loss = 0.0
        for _ in range(a.round_steps):
            loss = mixed_loss(model, *batch_rows(a.arm, P, found, has, tc["batch_size"], a.givens_share), tc["t_min"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step(); sched.step()
            run_loss += loss.item(); step += 1
        rnd += 1
        e16 = band_eval(model, val, bands["val"], names, 16, "confidence")
        e4 = band_eval(model, val, bands["val"], names, 4, "confidence")
        elapsed = elapsed0 + time.time() - t0
        row = {"round": rnd, "step": step, "train_loss": run_loss / a.round_steps, "tried": tried, "accepted": accepted,
               **{f"buffer_{n}": has[tband == b].float().mean().item() for b, n in enumerate(names)},
               **{f"val_solve_{n}": e16[f"val_solve_{n}"] for n in names}, **{f"val4_solve_{n}": e4[f"val_solve_{n}"] for n in names},
               "val_cell_acc": e16["val_cell_acc"], "elapsed_s": elapsed}
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fields).writerow(row)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                    "step_init": step_init, "round": rnd, "elapsed_s": elapsed, "found": found, "has": has,
                    "rng_cpu": torch.get_rng_state(), "rng_dev": torch.mps.get_rng_state() if dev.type == "mps" else None}, state_path)
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
        if a.launch_minutes and rnd < a.rounds and time.time() - t0 >= a.launch_minutes * 60:
            print(f"paused {a.arm} after round {rnd}; relaunch to resume", flush=True)
            return
    torch.save(model.state_dict(), ck / "final.pt")
    json.dump({"arm": a.arm, "rounds": rnd, "final_step": step, "start_step": step_init, "final": row, "wall_clock_s": elapsed,
               "args": vars(a), "seed": tc["seed"], "env": env_record(dev), "ckpt_dir": str(ck)}, open(done_path, "w"), indent=1)
    print(f"finished {a.arm} at step {step}", flush=True)


if __name__ == "__main__":
    main()
