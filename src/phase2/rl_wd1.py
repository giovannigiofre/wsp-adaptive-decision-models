"""Hand-written wd1 RL fine-tuning for the toy Sudoku dLLM.

Per RL step: B puzzles x G completions (T=1, confidence strategy, S steps); reward = validity.
A_i = r_i - mean(r_group); groups with all-equal rewards are skipped.
w+ = softmax(A/tau), w- = softmax(-A/tau) within each group (w- = 0 for the positive-only ablation).
log p_hat(o) = one-step mean-field estimate: all non-given cells masked (+ each given masked with
prob p_mask), one forward pass, sum of log p(o_k) over non-given cells k.
loss = mean over kept groups of sum_i (w-_i - w+_i) * log p_hat(o_i); mu inner gradient steps per batch.

PYTHONPATH=src python -m phase2.rl_wd1 --variant wd1 --tau 1.0 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import batch_validity, build_datasets, digits_to_tokens
from .evaluate import solve_rate
from .model import build_model
from .sampler import sample
from .utils import CACHE, env_record, get_device, load_cfg, seed_all


def group_weights(r: torch.Tensor, tau: float = 1.0, positive_only: bool = False):
    """r: (B,G) rewards. Returns advantages, w+, w-, keep mask (B,) of groups with reward variance."""
    A = r - r.mean(1, keepdim=True)
    keep = r.max(1).values != r.min(1).values
    w_pos = torch.softmax(A / tau, dim=1)
    w_neg = torch.zeros_like(w_pos) if positive_only else torch.softmax(-A / tau, dim=1)
    return A, w_pos, w_neg, keep


def one_step_loglik(model, puzzles: torch.Tensor, completions: torch.Tensor, p_mask: float = 0.0,
                    generator: torch.Generator | None = None) -> torch.Tensor:
    """(N,16) puzzles (0 = hidden) and completions (digits) -> (N,) one-step log p_hat."""
    given = puzzles != 0
    extra = given & (torch.rand(puzzles.shape, device=puzzles.device, generator=generator) < p_mask)
    inp = torch.where(given & ~extra, completions, torch.zeros_like(completions))
    logp = F.log_softmax(model(digits_to_tokens(inp)).float(), -1)
    lp = logp.gather(-1, (completions - 1).unsqueeze(-1)).squeeze(-1)
    return (lp * (~given)).sum(1)


@torch.no_grad()
def sequential_loglik(model, puzzles: torch.Tensor, completions: torch.Tensor) -> torch.Tensor:
    """'True' likelihood under a fixed order: reveal non-given cells one at a time in row-major
    order, summing log p of each cell given the givens and all earlier cells. 16 forward passes."""
    given = puzzles != 0
    total = torch.zeros(len(puzzles), device=puzzles.device)
    for i in range(16):
        rows = ~given[:, i]
        if not rows.any():
            continue
        visible = given.clone()
        visible[:, :i] |= ~given[:, :i]  # earlier non-given cells revealed
        inp = torch.where(visible, completions, torch.zeros_like(completions))
        logp = F.log_softmax(model(digits_to_tokens(inp)).float(), -1)[:, i]
        total += torch.where(rows, logp.gather(-1, (completions[:, i] - 1).unsqueeze(-1)).squeeze(-1),
                             torch.zeros_like(total))
    return total


def wd1_loss(logp: torch.Tensor, w_pos: torch.Tensor, w_neg: torch.Tensor) -> torch.Tensor:
    """logp, w_pos, w_neg: (n_groups, G) for kept groups. Mean over groups of sum_i (w- - w+) logp."""
    return ((w_neg - w_pos) * logp).sum(1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase2.yaml")
    ap.add_argument("--variant", choices=["wd1", "positive_only"], default="wd1")
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--init", default=str(CACHE / "sft" / "sft_weak.pt"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/phase2/rl")
    ap.add_argument("--ckpt-dir", default=str(CACHE / "rl"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    rc = cfg["rl"]
    tau = a.tau if a.tau is not None else rc["tau"]
    lr = a.lr if a.lr is not None else rc["lr"]
    steps = a.steps or rc["max_rl_steps"]
    B, G, S = rc["batch_puzzles"], rc["group_size"], rc["rollout_steps"]
    eval_every = rc.get("eval_every", 25)
    dev = get_device(a.device or cfg["runtime"]["device"])
    name = f"{a.variant}_tau{tau:g}_seed{a.seed}{a.tag}"
    out, ck = Path(a.out), Path(a.ckpt_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    ck.mkdir(parents=True, exist_ok=True)

    ds = {k: v.to(dev) for k, v in build_datasets(cfg, cfg["data"].get("split_seed", 0)).items()}
    train, val, test = ds["train"], ds["val"], ds["test"]
    model = build_model(cfg).to(dev)
    model.load_state_dict(torch.load(a.init, map_location=dev))
    model.eval()  # no dropout during RL (rollouts and gradient passes)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    log_path = out / f"{name}.csv"
    fields = ["step", "mean_reward", "frac_skipped", "loss", "w_entropy", "logp_correct", "logp_wrong",
              "val_solve", "elapsed_s"]
    start, best = 1, (-1.0, 0)
    state_path = ck / "state.pt"
    if state_path.exists():  # resume
        st = torch.load(state_path, map_location=dev)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        torch.set_rng_state(st["rng_cpu"].cpu()); start, best = st["step"] + 1, tuple(st["best"])
        if dev.type == "mps":
            torch.mps.set_rng_state(st["rng_dev"].cpu())
        import pandas as pd
        pd.read_csv(log_path).query("step < @start").to_csv(log_path, index=False)
        print(f"resumed {name} at step {start}", flush=True)
    else:
        seed_all(a.seed)
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(fields)
        v0 = solve_rate(model, val, S, "confidence")
        best = (v0, 0)
        torch.save(model.state_dict(), ck / "best.pt")
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fields).writerow({"step": 0, "val_solve": v0, "elapsed_s": 0.0})
    t0 = time.time()
    for step in range(start, steps + 1):
        idx = torch.randint(0, len(train), (B,), device=dev)
        P = train.puzzles[idx].repeat_interleave(G, 0)
        with torch.no_grad():
            O = sample(model, P, S, rc["rollout_strategy"], rc["rollout_temperature"]).grid
        r = batch_validity(O, P).float().view(B, G)
        _, w_pos, w_neg, keep = group_weights(r, tau, a.variant == "positive_only")
        row = {"step": step, "mean_reward": r.mean().item(), "frac_skipped": 1 - keep.float().mean().item()}
        if keep.any():
            Pk, Ok = P.view(B, G, 16)[keep].reshape(-1, 16), O.view(B, G, 16)[keep].reshape(-1, 16)
            wp, wn, rk = w_pos[keep], w_neg[keep], r[keep]
            losses = []
            for _ in range(rc["inner_steps"]):
                lp = one_step_loglik(model, Pk, Ok, rc["prompt_mask_prob"]).view(-1, G)
                loss = wd1_loss(lp, wp, wn)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), rc.get("grad_clip", 1.0))
                opt.step()
                losses.append(loss.item())
            lpd = lp.detach()
            row.update(loss=sum(losses) / len(losses),
                       w_entropy=-(wp * wp.clamp_min(1e-12).log()).sum(1).mean().item(),
                       logp_correct=lpd[rk == 1].mean().item(), logp_wrong=lpd[rk == 0].mean().item())
        if step % eval_every == 0 or step == steps:
            row["val_solve"] = solve_rate(model, val, S, "confidence")
            if row["val_solve"] > best[0]:
                best = (row["val_solve"], step)
                torch.save(model.state_dict(), ck / "best.pt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": list(best),
                        "rng_cpu": torch.get_rng_state(),
                        "rng_dev": torch.mps.get_rng_state() if dev.type == "mps" else None}, state_path)
            print(f"{name} step={step} reward={row['mean_reward']:.3f} skip={row['frac_skipped']:.2f} "
                  f"val={row['val_solve']:.3f}", flush=True)
        row["elapsed_s"] = time.time() - t0
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fields).writerow(row)
    torch.save(model.state_dict(), ck / "final.pt")
    summ = {"name": name, "variant": a.variant, "tau": tau, "seed": a.seed, "lr": lr, "steps": steps,
            "init": a.init, "best_val": best[0], "best_step": best[1],
            "final_val": solve_rate(model, val, S, "confidence"),
            "final_test": solve_rate(model, test, S, "confidence"), "wall_clock_s_this_session": time.time() - t0,
            "env": env_record(dev)}
    model.load_state_dict(torch.load(ck / "best.pt", map_location=dev))
    summ["best_test"] = solve_rate(model, test, S, "confidence")
    json.dump(summ, open(out / f"{name}.json", "w"), indent=2)
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
