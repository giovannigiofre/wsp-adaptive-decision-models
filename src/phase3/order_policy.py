"""Unmasking order as a decision problem on a frozen denoiser (phase 3, track D).

One cell is committed per step and its value is always the denoiser's greedy prediction, so the only
freedom is *which* masked cell to commit next.

oracle (3.4a): commit a masked cell whose greedy prediction equals the true solution (the most confident
such cell). If there is none the puzzle is lost: commit the most confident masked cell and carry on.
The oracle uses the solution, so it is an analysis tool, not a method. Its solve rate is reached by
some order, hence it is a lower bound on what the best possible order achieves.

policy (3.4b): a small head scores every masked cell from the frozen denoiser's outputs; one cell is
sampled from the softmax over those scores (greedy at evaluation). Training is REINFORCE with the
group-mean baseline on the validity reward. The log-probability of an order is exact here (a sum of
per-step log-softmaxes), unlike the denoiser's likelihood in rl.py. `--teacher oracle` instead trains
the same head with the solution (maximise the probability of picking a cell whose greedy prediction is
right): a supervised ceiling for what the features can express, not a method.

PYTHONPATH=src python -m phase3.order_policy oracle
PYTHONPATH=src python -m phase3.order_policy train --features probs --init zero --seed 0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn

from phase2.data import PuzzleSet, batch_validity, build_datasets, digits_to_tokens
from phase2.evaluate import bootstrap_ci, run_sampler
from phase2.model import build_model
from phase2.utils import CACHE as PHASE2_CACHE
from phase2.utils import env_record, get_device, load_cfg, seed_all

from .rl import CACHE, git_commit, sha256

OUT = Path("results/phase3")
HEURISTICS = ("random", "confidence", "margin", "entropy")


@torch.no_grad()
def oracle_order(model, ps: PuzzleSet) -> dict[str, torch.Tensor]:
    grid, masked = ps.puzzles.clone(), ps.puzzles == 0
    B = len(ps)
    stuck_at = torch.full((B,), -1, device=grid.device)       # first step with no correct candidate
    n_candidates = []                                         # mean number of correct masked cells per step
    for s in range(16):
        active = masked.any(1)
        if not active.any():
            break
        probs = model(digits_to_tokens(grid)).float().softmax(-1)
        conf, pred = probs.max(-1)
        pred = pred + 1
        ok = masked & (pred == ps.solutions)
        has_ok = ok.any(1)
        n_candidates.append(ok.sum(1)[active].float().mean().item())
        stuck_at = torch.where(active & ~has_ok & (stuck_at < 0), torch.full_like(stuck_at, s), stuck_at)
        score = torch.where(has_ok[:, None], conf.masked_fill(~ok, float("-inf")),
                            conf.masked_fill(~masked, float("-inf")))
        commit = torch.nn.functional.one_hot(score.argmax(1), 16).bool() & active[:, None]
        grid = torch.where(commit, pred, grid)
        masked = masked & ~commit
    return {"solved": batch_validity(grid, ps.puzzles).cpu(), "stuck_at": stuck_at.cpu(),
            "n_candidates": torch.tensor(n_candidates)}


def load_denoiser(base: dict, path: Path, dev):
    model = build_model(base).to(dev)
    model.load_state_dict(torch.load(path, map_location=dev))
    return model.eval()


def denoise(model, grid: torch.Tensor):
    """TinyDLLM forward that also returns the final hidden states: probs (B,16,4), hidden (B,16,d)."""
    h = model.tok(digits_to_tokens(grid)) + model.pos
    if model.structured:
        h = h + model.row(model.rcb[0]) + model.col(model.rcb[1]) + model.box(model.rcb[2])
    z = model.norm(model.enc(h))
    return model.head(z).float().softmax(-1), z.float()


def cell_features(probs: torch.Tensor) -> torch.Tensor:
    """(B,16,3): confidence, margin, 1 - normalised entropy. Each is one of the phase 2 heuristics."""
    top2 = probs.topk(2, dim=-1).values
    ent = -(probs * probs.clamp_min(1e-12).log()).sum(-1) / math.log(4)
    return torch.stack([top2[..., 0], top2[..., 0] - top2[..., 1], 1 - ent], -1)


class OrderPolicy(nn.Module):
    """score(cell) = linear(features) + MLP(features). features = 'probs' (3 heuristic scores) or
    'hidden' (the same 3 plus the denoiser's final hidden state)."""

    def __init__(self, features: str = "probs", d_model: int = 128, width: int = 64, init: str = "zero",
                 conf_scale: float = 10.0):
        super().__init__()
        self.features = features
        in_dim = 3 + (d_model if features == "hidden" else 0)
        self.lin = nn.Linear(in_dim, 1, bias=False)
        self.mlp = nn.Sequential(nn.Linear(in_dim, width), nn.Tanh(), nn.Linear(width, 1))
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)
        if init == "confidence":      # starts as the confidence rule: score = conf_scale * confidence
            with torch.no_grad():
                self.lin.weight[0, 0] = conf_scale
        elif init != "zero":          # 'zero' starts as the uniform random order
            raise ValueError(init)

    def forward(self, probs: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        x = cell_features(probs)
        if self.features == "hidden":
            x = torch.cat([x, hidden], -1)
        return (self.lin(x) + self.mlp(x)).squeeze(-1)


def rollout(model, policy, puzzles: torch.Tensor, solutions: torch.Tensor | None = None, greedy: bool = False):
    """One cell per step. Returns grid, the summed log-prob of the chosen order (B,), the teacher
    log-likelihood (B,) if solutions are given, and the share of steps agreeing with the confidence rule."""
    grid, masked = puzzles.clone(), puzzles == 0
    B = len(puzzles)
    logp, teach = torch.zeros(B, device=grid.device), torch.zeros(B, device=grid.device)
    agree, n_steps = 0.0, 0.0
    for _ in range(16):
        active = masked.any(1)
        if not active.any():
            break
        with torch.no_grad():
            probs, hidden = denoise(model, grid)
        conf, pred = probs.max(-1)
        pred = pred + 1
        scores = policy(probs, hidden).masked_fill(~masked, float("-inf"))
        scores = torch.where(active[:, None], scores, torch.zeros_like(scores))   # finished rows: dummy
        logpi = scores.log_softmax(-1)
        cell = scores.argmax(1) if greedy else torch.multinomial(logpi.exp(), 1).squeeze(1)
        logp = logp + torch.where(active, logpi.gather(1, cell[:, None]).squeeze(1), torch.zeros_like(logp))
        if solutions is not None:
            ok = masked & (pred == solutions)
            has = ok.any(1) & active
            ok_safe = ok | ~has[:, None]     # rows without a correct candidate: all cells, so lse = 0, no NaN
            lse = torch.logsumexp(logpi.masked_fill(~ok_safe, float("-inf")), 1)
            teach = teach + torch.where(has, lse, torch.zeros_like(lse))
        conf_cell = conf.masked_fill(~masked, float("-inf")).argmax(1)
        agree += ((cell == conf_cell) & active).sum().item()
        n_steps += active.sum().item()
        commit = nn.functional.one_hot(cell, 16).bool() & active[:, None]
        grid = torch.where(commit, pred, grid)
        masked = masked & ~commit
    return grid, logp, teach, agree / max(n_steps, 1.0)


@torch.no_grad()
def evaluate(model, policy, ps: PuzzleSet, batch: int = 2048) -> tuple[float, float]:
    solved, agree = [], []
    for i in range(0, len(ps), batch):
        p = ps.puzzles[i:i + batch]
        grid, _, _, ag = rollout(model, policy, p, greedy=True)
        solved.append(batch_validity(grid, p).float())
        agree.append(ag * len(p))
    return torch.cat(solved).mean().item(), sum(agree) / len(ps)


def train_cmd(a):
    cfg = load_cfg(a.config)
    base = load_cfg(cfg["base_config"])
    oc = cfg["order_policy"]
    dev = get_device(a.device or base["runtime"]["device"])
    lr = a.lr or oc["lr"][a.features]
    name = f"{a.teacher or 'reinforce'}_{a.features}_{a.init}_lr{lr:g}_seed{a.seed}{a.tag}"
    out, ck = OUT / a.exp, CACHE / a.exp / "ckpt"
    out.mkdir(parents=True, exist_ok=True)
    ck.mkdir(parents=True, exist_ok=True)
    if (out / f"{name}.json").exists() and not a.force:
        print(f"skip {name}: already finished", flush=True)
        return
    ds = {k: v.to(dev) for k, v in build_datasets(base, base["data"].get("split_seed", 0)).items()}
    train, val, test = ds["train"], ds["val"], ds["test"]
    den_path = Path(a.denoiser) if a.denoiser else PHASE2_CACHE / cfg["init"]["checkpoint"]
    model = load_denoiser(base, den_path, dev)
    for p in model.parameters():
        p.requires_grad_(False)
    seed_all(a.seed)
    policy = OrderPolicy(a.features, base["model"]["d_model"], oc["width"], a.init, oc["conf_scale"]).to(dev)
    steps, B, G = a.steps or oc["steps"], oc["batch_puzzles"], oc["group_size"]
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    fields = ["step", "train_reward", "loss", "val_solve", "val_agree_conf", "elapsed_s"]
    v, ag = evaluate(model, policy, val)
    best = (v, 0)
    torch.save(policy.state_dict(), ck / f"{name}_best.pt")
    with open(out / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerow({"step": 0, "val_solve": round(v, 4), "val_agree_conf": round(ag, 4), "elapsed_s": 0.0})
    t0, rew, losses = time.time(), [], []
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(train), (B,), device=dev)
        P = train.puzzles[idx].repeat_interleave(G, 0)
        sol = train.solutions[idx].repeat_interleave(G, 0) if a.teacher else None
        grid, logp, teach, _ = rollout(model, policy, P, sol)
        r = batch_validity(grid, P).float().view(B, G)
        if a.teacher:
            loss = -teach.mean()
        else:
            A = r - r.mean(1, keepdim=True)
            loss = -(A * logp.view(B, G)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        rew.append(r.mean().item()); losses.append(loss.item())
        if step % oc["eval_every"] == 0 or step == steps:
            v, ag = evaluate(model, policy, val)
            if v > best[0]:
                best = (v, step)
                torch.save(policy.state_dict(), ck / f"{name}_best.pt")
            with open(out / f"{name}.csv", "a", newline="") as f:
                csv.DictWriter(f, fields).writerow(
                    {"step": step, "train_reward": round(sum(rew) / len(rew), 4),
                     "loss": round(sum(losses) / len(losses), 5), "val_solve": round(v, 4),
                     "val_agree_conf": round(ag, 4), "elapsed_s": round(time.time() - t0, 1)})
            print(f"{name} step={step} reward={sum(rew) / len(rew):.3f} val={v:.3f} agree={ag:.2f}", flush=True)
            rew, losses = [], []
    torch.save(policy.state_dict(), ck / f"{name}_final.pt")
    final_val, final_agree = evaluate(model, policy, val)
    final_test, test_agree = evaluate(model, policy, test)
    summ = {"name": name, "method": a.teacher or "reinforce", "features": a.features, "init": a.init, "lr": lr,
            "seed": a.seed, "steps": steps, "denoiser": str(den_path), "denoiser_sha256": sha256(den_path),
            "code_commit": git_commit(), "best_val": best[0], "best_step": best[1], "final_val": final_val,
            "final_test": final_test, "final_val_agree_conf": final_agree, "final_test_agree_conf": test_agree,
            "lin_weight_first3": [round(x, 3) for x in policy.lin.weight[0, :3].tolist()],
            "wall_clock_s": time.time() - t0, "env": env_record(dev)}
    policy.load_state_dict(torch.load(ck / f"{name}_best.pt", map_location=dev))
    summ["best_test"] = evaluate(model, policy, test)[0]
    json.dump(summ, open(out / f"{name}.json", "w"), indent=2)
    print(json.dumps(summ), flush=True)


def transfer_cmd(a):
    """Apply the policies trained on the weak denoiser (best-val checkpoints) to other denoisers,
    next to the confidence rule and the oracle on the same denoiser. No training."""
    cfg = load_cfg(a.config)
    base = load_cfg(cfg["base_config"])
    oc = cfg["order_policy"]
    dev = get_device(a.device or base["runtime"]["device"])
    test = build_datasets(base, base["data"].get("split_seed", 0))["test"].to(dev)
    rows = []
    for spec in a.denoisers:
        label, path = spec.split("=", 1)
        model = load_denoiser(base, Path(path).expanduser(), dev)
        for h in ("confidence", "random"):
            rows.append({"denoiser": label, "order": f"{h} rule", "seed": -1,
                         "test_solve": run_sampler(model, test, 16, h)["solved"].float().mean().item()})
        rows.append({"denoiser": label, "order": "oracle", "seed": -1,
                     "test_solve": oracle_order(model, test)["solved"].float().mean().item()})
        for feats, init in (("probs", "zero"), ("hidden", "zero"), ("hidden", "confidence")):
            for ck in sorted((CACHE / "3.4b" / "ckpt").glob(f"reinforce_{feats}_{init}_lr*_seed*_best.pt")):
                policy = OrderPolicy(feats, base["model"]["d_model"], oc["width"], init, oc["conf_scale"]).to(dev)
                policy.load_state_dict(torch.load(ck, map_location=dev))
                solve, agree = evaluate(model, policy, test)
                rows.append({"denoiser": label, "order": f"policy {feats}, {init} start",
                             "seed": int(ck.stem.split("_seed")[1].split("_")[0]), "test_solve": solve,
                             "agree_conf": agree})
    df = pd.DataFrame(rows).round(4)
    df.to_csv(OUT / "3.4b" / "transfer.csv", index=False)
    summ = df.groupby(["denoiser", "order"], sort=False).agg(n=("seed", "count"), test_solve_mean=("test_solve", "mean"),
                                                            test_solve_std=("test_solve", "std")).reset_index()
    summ.round(4).to_csv(OUT / "3.4b" / "transfer_summary.csv", index=False)
    print(summ.round(3).to_string(index=False))


def oracle_cmd(a):
    cfg = load_cfg(a.config)
    base = load_cfg(cfg["base_config"])
    dev = get_device(a.device or base["runtime"]["device"])
    ds = {k: v.to(dev) for k, v in build_datasets(base, base["data"].get("split_seed", 0)).items()}
    model = load_denoiser(base, PHASE2_CACHE / cfg["init"]["checkpoint"], dev)
    out = OUT / "3.4a"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for split in ("val", "test"):
        ps = ds[split]
        for h in HEURISTICS:   # steps=16 > 12 hidden cells, so the phase 2 sampler commits one cell per step
            x = run_sampler(model, ps, 16, h)["solved"].float().numpy()
            rows.append({"split": split, "order": h, "n": len(x), "solve_rate": x.mean(),
                         **dict(zip(("ci_lo", "ci_hi"), bootstrap_ci(x)))})
        o = oracle_order(model, ps)
        x = o["solved"].float().numpy()
        rows.append({"split": split, "order": "oracle", "n": len(x), "solve_rate": x.mean(),
                     **dict(zip(("ci_lo", "ci_hi"), bootstrap_ci(x)))})
        if split == "test":
            k = ps.k.cpu()
            by_k = pd.DataFrame({"k": k.numpy(), "oracle": x,
                                 "confidence": run_sampler(model, ps, 16, "confidence")["solved"].float().numpy()})
            by_k.groupby("k").mean().round(4).reset_index().to_csv(out / "oracle_by_hidden_cells.csv", index=False)
            extra = {"never_stuck": float((o["stuck_at"] < 0).float().mean()),
                     "stuck_at_step_0": float((o["stuck_at"] == 0).float().mean()),
                     "mean_correct_candidates_by_step": [round(v, 3) for v in o["n_candidates"].tolist()],
                     "env": env_record(dev)}
            json.dump(extra, open(out / "oracle_details.json", "w"), indent=2)
    df = pd.DataFrame(rows).round(4)
    df.to_csv(out / "oracle.csv", index=False)
    print(df.to_string(index=False))
    print(pd.read_csv(out / "oracle_by_hidden_cells.csv").to_string(index=False))
    print(json.dumps({k: v for k, v in extra.items() if k != "env"}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["oracle", "train", "transfer"])
    ap.add_argument("--denoisers", nargs="*", default=[], help="transfer: label=path pairs")
    ap.add_argument("--config", default="configs/phase3.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--exp", default="3.4b")
    ap.add_argument("--features", choices=["probs", "hidden"], default="probs")
    ap.add_argument("--init", choices=["zero", "confidence"], default="zero")
    ap.add_argument("--teacher", choices=["oracle"], default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--denoiser", default=None, help="checkpoint path; default is init.checkpoint")
    ap.add_argument("--tag", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    {"oracle": oracle_cmd, "train": train_cmd, "transfer": transfer_cmd}[a.what](a)


if __name__ == "__main__":
    main()
