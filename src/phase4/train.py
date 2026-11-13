"""Step 4.1: masked diffusion training of the 9x9 model. Same objective as phase 2 (phase2.train_diffusion.diffusion_loss):
givens visible, t ~ U(t_min, 1), each non-given cell masked with probability t, cross-entropy on masked cells / t.

Resumable: state.pt at every evaluation, one CSV row per evaluation. Stops at the first of: `--max-minutes` of wall-clock
(summed over resumed launches), the plateau rule of the config, `max_steps`.

    PYTHONPATH=src python -m phase4.train --name size_small --size small --max-minutes 20 --no-plateau
    PYTHONPATH=src python -m phase4.train --name sft --launch-minutes 110
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from phase2.evaluate import run_sampler
from phase2.model import build_model, n_params
from phase2.train_diffusion import diffusion_loss, eval_loss
from phase2.utils import apply_overrides, env_record, get_device, load_cfg, seed_all

from .data import load_sets
from .generate import cache_dir

FIELDS = ["step", "train_loss", "val_loss", "val_solve", "val_cell_acc", "lr", "elapsed_s"]


@torch.no_grad()
def band_eval(model, ps, band: torch.Tensor, names: list[str], steps: int, strategy: str, batch: int = 1024) -> dict:
    out = run_sampler(model, ps, steps, strategy, batch=batch)
    hidden = (ps.puzzles == 0).cpu()
    ok = (out["grid"] == ps.solutions.cpu()) & hidden
    row = {"val_solve": out["solved"].float().mean().item(), "val_cell_acc": (ok.sum() / hidden.sum()).item()}
    for b, name in enumerate(names):
        row[f"val_solve_{name}"] = out["solved"][band == b].float().mean().item()
    return row


def plateau(history: list[tuple[int, float]], step: int, window: int, rel: float) -> bool:
    """True if the best val loss in the last `window` steps is not `rel` below the best before them."""
    old = [v for s, v in history if s <= step - window]
    new = [v for s, v in history if s > step - window]
    return bool(old) and bool(new) and min(new) > (1 - rel) * min(old)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase4.yaml")
    ap.add_argument("--name", required=True)
    ap.add_argument("--size", default=None, help="key of `sizes` in the config")
    ap.add_argument("--max-minutes", type=float, default=None, help="total training budget over all launches")
    ap.add_argument("--launch-minutes", type=float, default=None, help="stop this launch after this long (resume later)")
    ap.add_argument("--no-plateau", action="store_true")
    ap.add_argument("--loss", default="diffusion", choices=["diffusion", "givens"],
                    help="givens (step 4.8): train on masked givens only, stored solutions of the train set zeroed")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None, help="4.9e: learning rate set after loading the state (default: keep it)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/phase4/sft")
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = apply_overrides(load_cfg(a.config), a.set)
    if a.size:
        cfg["model"].update(cfg["sizes"][a.size])
    tc = cfg["diffusion_training"]
    max_steps = a.max_steps or tc["max_steps"]
    dev = get_device(a.device or cfg["runtime"]["device"])
    out, ck = Path(a.out), Path(a.ckpt_dir).expanduser() if a.ckpt_dir else cache_dir(cfg) / "sft" / a.name
    out.mkdir(parents=True, exist_ok=True)
    ck.mkdir(parents=True, exist_ok=True)
    log_path, state_path, done_path = out / f"{a.name}.csv", ck / "state.pt", out / f"{a.name}.json"
    if done_path.exists():
        print(f"skip {a.name}: already finished")
        return

    sets, bands, names = load_sets(cfg, dev, ("train", "val"))
    train, val = sets["train"], sets["val"]
    loss_fn = lambda ps, gen=None: diffusion_loss(model, ps, tc["t_min"], gen)          # noqa: E731
    val_loss = lambda: eval_loss(model, val, tc["t_min"])                              # noqa: E731
    if a.loss == "givens":
        from .selflearn import givens_loss, without_solutions
        train = without_solutions(train)
        loss_fn = lambda ps, gen=None: givens_loss(model, ps.puzzles, tc["t_min"], gen)  # noqa: E731

        @torch.no_grad()
        def val_loss(reps: int = 4) -> float:
            model.eval()
            gen = torch.Generator(device=dev).manual_seed(123)
            v = sum(loss_fn(val, gen).item() for _ in range(reps)) / reps
            model.train()
            return v
    fields = FIELDS[:4] + [f"val_solve_{n}" for n in names] + FIELDS[4:]
    seed_all(tc["seed"])
    model = build_model(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / tc["warmup_steps"]))
    step0, elapsed0, history = 0, 0.0, []
    if state_path.exists():
        st = torch.load(state_path, map_location=dev)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        torch.set_rng_state(st["rng_cpu"].cpu())
        if dev.type == "mps":
            torch.mps.set_rng_state(st["rng_dev"].cpu())
        step0, elapsed0, history = st["step"], st["elapsed_s"], st["history"]
        if a.lr is not None:
            from .selfloop import set_lr
            set_lr(opt, sched, a.lr)
        import pandas as pd
        pd.read_csv(log_path).query("step <= @step0").to_csv(log_path, index=False)
        print(f"resumed {a.name} at step {step0} ({elapsed0 / 60:.1f} min so far)", flush=True)
    else:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(fields)
    print(f"{a.name}: params={n_params(model)} device={dev} train={len(train)} val={len(val)}", flush=True)

    dense = set(range(tc["dense_eval_every"], tc["dense_eval_until"] + 1, tc["dense_eval_every"]))
    t0, run_loss, n_loss, reason = time.time(), 0.0, 0, None
    model.train()
    step = step0
    while reason is None:
        step += 1
        idx = torch.randint(0, len(train), (tc["batch_size"],), device=dev)
        loss = loss_fn(train.subset(idx))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        opt.step(); sched.step()
        run_loss += loss.item(); n_loss += 1
        elapsed = elapsed0 + time.time() - t0
        if step >= max_steps:
            reason = "max_steps"
        elif a.max_minutes and elapsed >= a.max_minutes * 60:
            reason = "time_budget"
        pause = a.launch_minutes and time.time() - t0 >= a.launch_minutes * 60
        if not (reason or pause or step in dense or step % tc["eval_every"] == 0):
            continue
        row = {"step": step, "train_loss": run_loss / n_loss, "val_loss": val_loss(),
               **band_eval(model, val, bands["val"], names, tc["eval_sampler_steps"], tc["eval_strategy"]),
               "lr": sched.get_last_lr()[0], "elapsed_s": elapsed}
        run_loss, n_loss = 0.0, 0
        history.append((step, row["val_loss"]))
        if not a.no_plateau and reason is None and plateau(history, step, tc["plateau_window"], tc["plateau_rel_improvement"]):
            reason = "plateau"
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fields).writerow(row)
        if step % tc["eval_every"] == 0 or reason:
            torch.save(model.state_dict(), ck / f"step_{step:06d}.pt")
        elapsed = elapsed0 + time.time() - t0   # evaluation time counts towards the budget of the next check
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                    "elapsed_s": elapsed, "history": history, "rng_cpu": torch.get_rng_state(),
                    "rng_dev": torch.mps.get_rng_state() if dev.type == "mps" else None}, state_path)
        print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
        if pause and reason is None:
            print(f"paused {a.name} at step {step}; relaunch to resume", flush=True)
            return
    best = min(history, key=lambda h: h[1])
    torch.save(model.state_dict(), ck / "final.pt")
    json.dump({"name": a.name, "stop_reason": reason, "final_step": step, "best_val_loss": best[1], "best_val_loss_step": best[0],
               "final": row, "n_params": n_params(model), "model": cfg["model"], "wall_clock_s": elapsed, "overrides": a.set,
               "size": a.size, "loss": a.loss, "seed": tc["seed"], "env": env_record(dev), "ckpt_dir": str(ck)}, open(done_path, "w"), indent=1)
    print(f"finished {a.name}: {reason} at step {step}", flush=True)


if __name__ == "__main__":
    main()
