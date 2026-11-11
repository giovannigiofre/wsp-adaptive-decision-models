"""Step 4.6a: exact-gradient audit of the RL objectives, no training (docs/RL_DLLM_DIRECTIONS.md, direction 1).

On this task the reward is binary and the solution unique, so the RL objective at rollout temperature 1 is
J = E_puzzle[P(the rollout is valid)], and `phase3.rl.exact_trajectory_logp` gives the exact probability of a
rollout. So g* = E[r * grad log P(trajectory)] can be estimated without a likelihood proxy, and the update
direction of every objective of 4.4 / 4.5a can be compared with it on a fixed set of puzzles.

What is measured, per checkpoint:
  reference   g* from `--ref-rollouts` rollouts per puzzle, in two halves (their cosine is the noise of g*)
  directions  for each objective the descent direction of its loss, mean over `--batches` batches of 32 x 8
              rollouts: cosine with g*, cosine between the two halves of the batches, norm, noise; for some
              objectives again at inner step 4 after three real optimiser updates on the batch
  parts       the wd1 direction split into pull-up and push-down, and by cell type: right cells, wrong cells of
              the first-error step, wrong cells committed later
  eps         gap between each likelihood estimate and the exact log-likelihood, valid and invalid rollouts
  fd          the same puzzles re-sampled after a step of fixed length along each direction: change of the
              valid share (J) against the first-order prediction step * (g* . direction). This is the check
              that does not depend on the likelihood formula, and it sees the jumps of J that g* does not.
  cells       `--compare name=path`: change of the one-step log-probability of the solution's digit against
              the audited checkpoint, by commit step and cell type (section 4 of the memo)

    PYTHONPATH=src python -m phase4.grad_audit --size medium
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from phase2.data import PuzzleSet, batch_validity
from phase2.model import build_model
from phase2.sampler import sample
from phase2.train_diffusion import diffusion_loss
from phase2.utils import get_device, seed_all
from phase3.objectives import filtered, grpo_loss, reinforce, wd1, wd1_batch, weighted_loss
from phase3.rl import (cell_logp, exact_trajectory_logp, one_step_logprobs, prompt_mask, trajectory_cell_logp)

from .data import load_sets
from .generate import cache_dir

FLOOR, B, G, P_MASK = -75.6, 32, 8, 0.15
GROUPS = ("base", "grpo", "wd1_cells", "batch", "self_training", "pull", "supervised", "exact")
INNER = ("wd1", "positive_only", "wd1_floored", "filtered", "self_training")


def grad_vec(model, loss: torch.Tensor, retain: bool = False) -> torch.Tensor:
    ps = [p for p in model.parameters()]
    gs = torch.autograd.grad(loss, ps, retain_graph=retain, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for g, p in zip(gs, ps)]).detach()


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a @ b / (a.norm() * b.norm()).clamp_min(1e-30)).item()


@torch.no_grad()
def rollout(model, P: torch.Tensor, S: int):
    o = sample(model, P, S, "confidence", 1.0)
    return o.grid, o.commit_step, batch_validity(o.grid, P)


def reference(model, P: torch.Tensor, S: int, n_roll: int, chunk: int = 256):
    """g* = mean over rollouts of r * grad log P(trajectory), in two halves. Returns (g_half0, g_half1, valid share)."""
    g, n, n_valid = [0.0, 0.0], [0, 0], 0
    for rep in range(n_roll):
        O, CS, valid = rollout(model, P, S)
        n[rep % 2] += len(P); n_valid += int(valid.sum())
        idx = valid.nonzero().squeeze(1)
        for i in range(0, len(idx), chunk):
            j = idx[i:i + chunk]
            g[rep % 2] = g[rep % 2] + grad_vec(model, exact_trajectory_logp(model, P[j], O[j], CS[j]).sum())
    return g[0] / n[0], g[1] / n[1], n_valid / sum(n)


@torch.no_grad()
def valid_share(model, P: torch.Tensor, S: int, n_roll: int) -> float:
    return sum(rollout(model, P, S)[2].float().mean().item() for _ in range(n_roll)) / n_roll


def losses(model, P, O, CS, Sol, valid, t_min: float, want=None) -> dict:
    """{objective: loss} on one batch of B x G rollouts (rows ordered by puzzle). A loss is absent when the
    batch gives the objective nothing to do (no group with unequal rewards, no valid rollout). `want` is a set
    of GROUPS names; the caller asks for one group at a time so that only one set of graphs is alive."""
    out, r, given = {}, valid.float().view(B, G), P != 0
    extra = prompt_mask(P, P_MASK)
    logp, _ = one_step_logprobs(model, P, O, extra=extra)
    cl = cell_logp(logp, O) * (~given)
    lp = cl.sum(1).view(B, G)
    keep = r.max(1).values != r.min(1).values
    use = lambda name: want is None or name in want                                      # noqa: E731
    if keep.any():
        gw = wd1(r)
        pos, neg = gw.pos[keep], gw.neg[keep]
    if keep.any() and use("base"):
        out["positive_only"] = -(pos * lp[keep]).sum(1).mean()
        out["wd1_push_down"] = (neg * lp[keep]).sum(1).mean()
        out["wd1_floored"] = (neg * lp[keep].clamp_min(FLOOR) - pos * lp[keep]).sum(1).mean()
        f = filtered(r)
        out["filtered"] = -(f.pos[keep] * lp[keep]).sum(1).mean()
        rf = reinforce(r)
        out["reinforce_onestep"] = ((rf.neg - rf.pos)[keep] * lp[keep]).sum(1).mean()
    if keep.any():
        if use("grpo"):
            A = (r - r.mean(1, keepdim=True))[keep].reshape(-1)
            rows = keep.repeat_interleave(G)
            c = cell_logp(logp, O)[rows]
            out["grpo"] = grpo_loss(c, c.detach(), A, ~given[rows], 0.5)[0]
        if use("wd1_cells"):                               # wd1 by cell type; the three parts sum to wd1
            wrong = (O != Sol) & ~given
            first = torch.where(wrong, CS, torch.full_like(CS, 10 ** 6)).min(1, keepdim=True).values
            coef = (gw.neg - gw.pos).view(-1, 1) * keep.repeat_interleave(G).view(-1, 1) / keep.sum()
            for name, m in (("wd1_right_cells", ~wrong), ("wd1_first_error_cells", wrong & (CS == first)),
                            ("wd1_later_wrong_cells", wrong & (CS > first))):
                out[name] = (coef * cl * m).sum()
    if use("batch"):
        for name, gb in (("wd1_batch_floored", wd1_batch(r, floor=FLOOR)), ("wd1_batch_beta0", wd1_batch(r, beta=0.0))):
            out[name] = weighted_loss(lp, gb)
    if valid.any():
        if use("self_training"):
            out["self_training"] = diffusion_loss(model, PuzzleSet(P[valid], O[valid], torch.zeros(int(valid.sum()))), t_min)
        if use("pull"):                                    # raise the valid rollouts under three likelihoods
            out["pull_onestep_nomask"] = -(cell_logp(one_step_logprobs(model, P[valid], O[valid])[0], O[valid])
                                           * (~given[valid])).sum(1).sum() / len(P)
            out["pull_path_no_order_factor"] = -(trajectory_cell_logp(model, P[valid], O[valid], CS[valid])
                                                 * (~given[valid])).sum(1).sum() / len(P)
    if use("supervised"):
        out["supervised_all"] = diffusion_loss(model, PuzzleSet(P, Sol, torch.zeros(len(P))), t_min)
    if use("exact"):                                       # policy gradient with the exact likelihood
        ex = exact_trajectory_logp(model, P, O, CS)
        out["pg_exact"] = -(valid.float() * ex).mean()
        out["pg_exact_group_baseline"] = -((r - r.mean(1, keepdim=True)).view(-1) * ex).mean()
        out["_eps"] = pd.DataFrame({"valid": valid.cpu().numpy(), "exact": ex.detach().cpu().numpy(),
                                    "onestep_masked_prompt": lp.detach().view(-1).cpu().numpy()})
    return out


class Acc:
    """Running mean direction (two halves), squared norms and per-batch cosines with g*."""
    def __init__(self):
        self.h, self.n, self.sq, self.cb, self.used = [0.0, 0.0], 0, 0.0, [], 0

    def add(self, d: torch.Tensor | None, gstar: torch.Tensor):
        k = self.n % 2
        self.n += 1
        if d is None:
            return
        self.used += 1
        self.h[k] = self.h[k] + d
        self.sq += (d @ d).item(); self.cb.append(cos(d, gstar))

    def mean(self) -> torch.Tensor:
        return (self.h[0] + self.h[1]) / self.n

    def row(self, gstar: torch.Tensor) -> dict:
        m = self.mean()
        noise = max(self.sq / self.n - (m @ m).item(), 0.0) ** 0.5
        both = all(torch.is_tensor(x) for x in self.h)
        return {"batches": self.n, "batches_with_signal": self.used, "cos_with_gstar": cos(m, gstar),
                "cos_between_halves": cos(self.h[0], self.h[1]) if both else float("nan"),
                "mean_cos_per_batch": sum(self.cb) / max(1, len(self.cb)), "norm_of_mean": m.norm().item(),
                "batch_noise_over_norm": noise / max(m.norm().item(), 1e-30),
                "J_gain_per_unit_step": (m @ gstar / m.norm().clamp_min(1e-30)).item()}


def audit(model, train: PuzzleSet, cfg: dict, a, name: str, out: Path):
    dev, S, t_min = train.puzzles.device, a.steps, cfg["diffusion_training"]["t_min"]
    P0, S0 = train.puzzles[:a.puzzles], train.solutions[:a.puzzles]
    seed_all(a.seed)
    g0, g1, J = reference(model, P0, S, a.ref_rollouts)
    gstar = (g0 + g1) / 2
    ref = {"model": name, "J_valid_share": J, "gstar_norm": gstar.norm().item(), "gstar_cos_between_halves": cos(g0, g1),
           "puzzles": a.puzzles, "ref_rollouts_per_puzzle": a.ref_rollouts, "sampler_steps": S}
    print(ref, flush=True)

    acc, acc4, eps = {}, {}, []
    theta = copy.deepcopy(model.state_dict())
    for b in range(a.batches):
        idx = torch.randint(0, a.puzzles, (B,), device=dev)
        P, Sol = P0[idx].repeat_interleave(G, 0), S0[idx].repeat_interleave(G, 0)
        O, CS, valid = rollout(model, P, S)
        d = {}
        for grp in GROUPS:                                  # one group at a time: its graphs are freed before the next
            ls = losses(model, P, O, CS, Sol, valid, t_min, want={grp})
            if grp == "exact":
                e = ls.pop("_eps")
            keys = list(ls)
            for i, k in enumerate(keys):
                d[k] = -grad_vec(model, ls[k], retain=i < len(keys) - 1)
            del ls
        with torch.no_grad():
            given = P != 0
            e["onestep_no_mask"] = (cell_logp(one_step_logprobs(model, P, O)[0], O) * ~given).sum(1).cpu().numpy()
            e["path_no_order_factor"] = (trajectory_cell_logp(model, P, O, CS) * ~given).sum(1).cpu().numpy()
        eps.append(e)
        if "positive_only" in d:
            d["wd1"] = d["positive_only"] + d["wd1_push_down"]
        for k in NAMES:
            acc.setdefault(k, Acc()).add(d.get(k), gstar)
        del d
        for k in INNER:                                     # inner step 4, after three real updates on this batch
            opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
            d4 = None
            for i in range(4):
                ls = losses(model, P, O, CS, Sol, valid, t_min, want={"self_training" if k == "self_training" else "base"})
                l4 = ls["positive_only"] + ls["wd1_push_down"] if k == "wd1" and "positive_only" in ls else ls.get(k)
                if l4 is None:
                    break
                if i == 3:
                    d4 = -grad_vec(model, l4)
                    break
                opt.zero_grad(set_to_none=True); l4.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip); opt.step()
            model.load_state_dict(theta)
            acc4.setdefault(k, Acc()).add(d4, gstar)
        if (b + 1) % 10 == 0:
            print(f"{name}: batch {b + 1}/{a.batches}", flush=True)

    rows = [{"model": name, "objective": k, "inner_step": 1, **v.row(gstar)} for k, v in acc.items()]
    rows += [{"model": name, "objective": k, "inner_step": 4, **v.row(gstar)} for k, v in acc4.items()]
    e = pd.concat(eps)
    erow = []
    for v, grp in e.groupby("valid"):
        for col in ("onestep_masked_prompt", "onestep_no_mask", "path_no_order_factor"):
            gap = grp[col] - grp.exact
            erow.append({"model": name, "rollouts": "valid" if v else "invalid", "n": len(grp), "estimate": col,
                         "exact_logp_mean": grp.exact.mean(), "gap_mean": gap.mean(), "gap_std": gap.std(),
                         "gap_abs_mean": gap.abs().mean(), "corr_with_exact": grp[col].corr(grp.exact)})

    # finite steps along each direction: does the valid share move as g* says?
    fd, eta = [], [x / max(gstar.norm().item(), 1e-30) for x in a.fd_gain]
    seed_all(a.seed + 1)
    J0 = valid_share(model, P0, S, a.fd_rollouts)
    dirs = {"gstar": gstar, **{k: acc[k].mean() for k in a.fd_dirs if k in acc and acc[k].used}}
    params = [p for p in model.parameters()]
    for k, d in dirs.items():
        u = d / d.norm().clamp_min(1e-30)
        for step in eta:
            with torch.no_grad():
                i = 0
                for p in params:
                    p += step * u[i:i + p.numel()].view_as(p); i += p.numel()
            fd.append({"model": name, "direction": k, "step_norm": step, "J_before": J0,
                       "J_after": valid_share(model, P0, S, a.fd_rollouts), "predicted_change": step * (u @ gstar).item()})
            fd[-1]["measured_change"] = fd[-1]["J_after"] - J0
            model.load_state_dict(theta)
            print(fd[-1], flush=True)
    return ref, rows, erow, fd


@torch.no_grad()
def cell_changes(model, other, train: PuzzleSet, a, name: str, other_name: str) -> list[dict]:
    """Change of the one-step log p (answer fully masked, no prompt mask) from `model` to `other`, on rollouts
    of `model`: of the solution's digit, by commit step and cell type, and of the wrong digit that was written."""
    P = train.puzzles[:a.puzzles].repeat_interleave(G, 0); Sol = train.solutions[:a.puzzles].repeat_interleave(G, 0)
    seed_all(a.seed + 2)
    O, CS, valid = rollout(model, P, a.steps)
    given, wrong = P != 0, (O != Sol) & (P != 0).logical_not()
    first = torch.where(wrong, CS, torch.full_like(CS, 10 ** 6)).min(1, keepdim=True).values
    lp = lambda m, X: cell_logp(one_step_logprobs(m, P, X)[0], X)                         # noqa: E731
    d_sol, d_own = lp(other, Sol) - lp(model, Sol), lp(other, O) - lp(model, O)
    v = valid[:, None].expand_as(O)
    types = {"valid rollout": v & ~given, "invalid, right, before the first error": ~v & ~given & ~wrong & (CS < first),
             "invalid, right, at or after the first error": ~v & ~given & ~wrong & (CS >= first),
             "invalid, wrong, first-error step": wrong & (CS == first), "invalid, wrong, later": wrong & (CS > first)}
    rows = []
    for t, m in types.items():
        for s in [None] + list(range(a.steps)):
            mm = m if s is None else m & (CS == s)
            if mm.any():
                rows.append({"from": name, "to": other_name, "cell_type": t, "commit_step": "all" if s is None else s,
                             "cells": int(mm.sum()), "dlogp_solution_digit": d_sol[mm].mean().item(),
                             "dlogp_written_digit": d_own[mm].mean().item()})
    return rows


NAMES = ["pg_exact", "pg_exact_group_baseline", "pull_path_no_order_factor", "pull_onestep_nomask", "self_training",
         "supervised_all", "filtered", "positive_only", "wd1", "wd1_push_down", "wd1_floored", "wd1_batch_floored",
         "wd1_batch_beta0", "reinforce_onestep", "grpo", "wd1_right_cells", "wd1_first_error_cells", "wd1_later_wrong_cells"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase4.yaml")
    ap.add_argument("--size", default="medium")
    ap.add_argument("--band", default="hard")
    ap.add_argument("--steps", type=int, default=4, help="sampler steps of the rollouts")
    ap.add_argument("--puzzles", type=int, default=256)
    ap.add_argument("--ref-rollouts", type=int, default=64)
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--fd-rollouts", type=int, default=128)
    ap.add_argument("--fd-gain", type=float, nargs="*", default=[0.02, 0.08], help="step sizes, as predicted gain along g*")
    ap.add_argument("--fd-dirs", nargs="*", default=["pg_exact_group_baseline", "self_training", "supervised_all", "filtered",
                                                     "positive_only", "wd1", "wd1_push_down", "wd1_batch_floored"])
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpts", nargs="*", default=[], help="name=path of further checkpoints to audit")
    ap.add_argument("--compare", nargs="*", default=[], help="name=path: per-cell log-prob change against the start")
    ap.add_argument("--out", default="results/phase4/4.6a")
    ap.add_argument("--device", default=None, help="override runtime.device (cpu for a smoke run)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    cfg["model"].update(cfg["sizes"][a.size])
    dev = get_device(a.device or cfg["runtime"]["device"])
    sets, bands, names = load_sets(cfg, dev, splits=("train",))
    train = sets["train"].subset((bands["train"] == names.index(a.band)).to(dev))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    start = cache_dir(cfg) / "sft" / "sft" / "final.pt"
    model = build_model(cfg).to(dev)
    refs, rows, eps, fd = [], [], [], []
    suffix = "" if a.steps == 4 else f"_s{a.steps}"
    for name, path in [("start", start)] + [tuple(x.split("=", 1)) for x in a.ckpts]:
        model.load_state_dict(torch.load(path, map_location=dev)); model.eval()
        r, d, e, f = audit(model, train, cfg, a, name, out)
        refs.append(r); rows += d; eps += e; fd += f
        for df, fn in ((refs, "reference"), (rows, "directions"), (eps, "eps"), (fd, "fd")):
            pd.DataFrame(df).round(5).to_csv(out / f"{fn}{suffix}.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
    for df in (rows, eps, fd):
        print(pd.DataFrame(df).round(4).to_string(index=False), flush=True)
    if a.compare:
        model.load_state_dict(torch.load(start, map_location=dev)); model.eval()
        other, cells = build_model(cfg).to(dev), []
        for x in a.compare:
            n, path = x.split("=", 1)
            other.load_state_dict(torch.load(path, map_location=dev)); other.eval()
            cells += cell_changes(model, other, train, a, "start", n)
        pd.DataFrame(cells).round(4).to_csv(out / "cells.csv", index=False)
        print(pd.DataFrame(cells).round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
