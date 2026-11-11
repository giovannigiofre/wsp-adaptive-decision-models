"""Step 4.5b: remasking at inference ("eraser"), no training (docs/RL_DLLM_SURVEY.md, proposal 6).

4.4b showed that a failed grid starts from about two wrong cells and that the rest follows. Here a committed
cell can be masked again and re-predicted. Every arm is compared with the plain sampler at the same number
of forward passes.

Arms (N = forward passes):
  plain          phase2.sampler.sample with N steps
  rcr            running-confidence remasking: at every step all hidden cells compete, a committed cell with
                 its confidence at commit time, a masked cell with its current confidence; the lowest are masked
  fix_lowconf    fill in N - R steps, then R rounds: mask the 20% of filled cells with the lowest commit
                 confidence and re-predict them in one pass
  fix_conflict   same, but mask the filled cells that share their digit with a peer (uses the Sudoku rules,
                 not the solution: a reference for a detector that knows the rules)
  fix_oracle     same, but mask exactly the wrong cells (uses the solution: upper reference for any detector)

Also: how well four scores separate right from wrong filled cells on the final grid of the plain sampler.

    PYTHONPATH=src python -m phase4.eraser --size medium
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from phase2.data import batch_validity, digits_to_tokens
from phase2.evaluate import auroc, bootstrap_ci
from phase2.model import build_model
from phase2.sampler import sample, select_topk
from phase2.utils import get_device, seed_all

from .data import load_sets
from .generate import cache_dir
from .pre_rl_checks import peers9

FRAC = 0.2


def conflicts(grid: torch.Tensor) -> torch.Tensor:
    """(B,81) filled grid -> (B,81) bool: the cell shares its digit with a peer."""
    P = torch.from_numpy(peers9()).to(grid.device)
    return ((grid[:, :, None] == grid[:, None, :]) & P).any(2)


@torch.no_grad()
def predict(model, grid: torch.Tensor):
    """Greedy digit and its probability for every cell, given the grid (0 = masked)."""
    p = model(digits_to_tokens(grid)).float().softmax(-1)
    conf, chosen = p.max(-1)
    return chosen + 1, conf, p


@torch.no_grad()
def sample_rcr(model, puzzles: torch.Tensor, steps: int):
    """Running-confidence remasking. Returns (grid, mean number of remask events per puzzle)."""
    hidden = puzzles == 0
    n_hidden = hidden.sum(1)
    grid, masked = puzzles.clone(), hidden.clone()
    conf = torch.zeros(puzzles.shape, device=puzzles.device)
    events = torch.zeros(len(puzzles), device=puzzles.device)
    for s in range(steps):
        digit, c, _ = predict(model, grid)
        score = torch.where(masked, c, conf)
        value = torch.where(masked, digit, grid)
        target = torch.ceil(n_hidden.float() * (s + 1) / steps).long()
        keep = select_topk(score, hidden, target)
        events += (hidden & ~masked & ~keep).sum(1)
        grid = torch.where(keep, value, puzzles)
        conf = torch.where(keep, score, torch.zeros_like(score))
        masked = hidden & ~keep
    return grid, events.mean().item()


@torch.no_grad()
def sample_fix(model, puzzles: torch.Tensor, solutions: torch.Tensor, fill_steps: int, rounds: int, detector: str):
    """Fill with the plain sampler, then `rounds` passes of mask-and-re-predict. Returns (grid, mean cells masked per round)."""
    hidden = puzzles == 0
    o = sample(model, puzzles, fill_steps, "confidence", 0.0)
    grid, conf = o.grid.clone(), o.commit_conf.clone()
    n = 0.0
    for _ in range(rounds):
        if detector == "lowconf":
            k = torch.ceil(hidden.sum(1).float() * FRAC).long()
            rem = select_topk(-conf.nan_to_num(float("inf")), hidden, k)
        elif detector == "conflict":
            rem = conflicts(grid) & hidden
        elif detector == "oracle":
            rem = (grid != solutions) & hidden
        else:
            raise ValueError(detector)
        digit, c, _ = predict(model, torch.where(rem, torch.zeros_like(grid), grid))
        grid = torch.where(rem, digit, grid)
        conf = torch.where(rem, c, conf)
        n += rem.sum(1).float().mean().item() / rounds
    return grid, n


@torch.no_grad()
def detector_scores(model, puzzles: torch.Tensor, steps: int, parts: int = 4, seed: int = 0) -> dict:
    """Scores (higher = more likely right) for every hidden cell of the plain sampler's final grid."""
    hidden = puzzles == 0
    o = sample(model, puzzles, steps, "confidence", 0.0)
    grid = o.grid
    own = lambda p: p.gather(-1, (grid - 1).unsqueeze(-1)).squeeze(-1)     # prob of the digit that is written
    visible = own(predict(model, grid)[2])                                  # one pass, nothing masked
    g = torch.Generator(device="cpu").manual_seed(seed)
    part = torch.randint(0, parts, puzzles.shape, generator=g).to(puzzles.device)
    leave = torch.zeros_like(visible)
    for j in range(parts):                                                  # each hidden cell masked in one pass
        m = hidden & (part == j)
        leave = torch.where(m, own(predict(model, torch.where(m, torch.zeros_like(grid), grid))[2]), leave)
    return {"grid": grid, "hidden": hidden, "commit_conf": o.commit_conf, "visible_prob": visible,
            "leave_out_prob": leave, "no_conflict": (~conflicts(grid)).float()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase4.yaml")
    ap.add_argument("--size", default="medium")
    ap.add_argument("--band", default="hard")
    ap.add_argument("--budgets", type=int, nargs="*", default=[4, 8, 16, 32])
    ap.add_argument("--ckpts", nargs="*", default=[], help="name=path of extra checkpoints (state dicts)")
    ap.add_argument("--out", default="results/phase4/4.5b")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    cfg["model"].update(cfg["sizes"][a.size])
    dev = get_device(cfg["runtime"]["device"])
    sets, bands, names = load_sets(cfg, dev, splits=("test",))
    test = sets["test"].subset((bands["test"] == names.index(a.band)).to(dev))
    P, S = test.puzzles, test.solutions
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    models = {"start": cache_dir(cfg) / "sft" / "sft" / "final.pt"}
    models.update(dict(x.split("=", 1) for x in a.ckpts))
    model = build_model(cfg).to(dev)
    rows, det = [], []
    for name, path in models.items():
        model.load_state_dict(torch.load(path, map_location=dev)); model.eval()
        seed_all(0)
        for N in a.budgets:
            R = max(1, N // 4)
            base = batch_validity(sample(model, P, N, "confidence", 0.0).grid, P).float().cpu().numpy()
            arms = {"plain": (base, 0.0)}
            g, n = sample_rcr(model, P, N)
            arms["rcr"] = (batch_validity(g, P).float().cpu().numpy(), n)
            for d in ("lowconf", "conflict", "oracle"):
                g, n = sample_fix(model, P, S, N - R, R, d)
                arms[f"fix_{d}"] = (batch_validity(g, P).float().cpu().numpy(), n)
            for arm, (v, n) in arms.items():
                lo, hi = bootstrap_ci(v - base)
                rows.append({"model": name, "forward_passes": N, "fix_rounds": 0 if arm in ("plain", "rcr") else R,
                             "arm": arm, "solve": v.mean(), "minus_plain": (v - base).mean(), "lo": lo, "hi": hi,
                             "cells_remasked": n})
                print(rows[-1], flush=True)
        for N in (4, 16):
            if name != "start":
                break
            d = detector_scores(model, P, N)
            h = d["hidden"].cpu().numpy()
            right = (d["grid"] == S).cpu().numpy()
            failed = ~right.all(1)
            for score in ("commit_conf", "visible_prob", "leave_out_prob", "no_conflict"):
                s = d[score].cpu().numpy()
                det.append({"sampler_steps": N, "score": score, "wrong_cells": int((~right & h).sum()),
                            "auroc_all_cells": auroc(s[h], right[h]),
                            "auroc_failed_grids": auroc(s[failed][h[failed]], right[failed][h[failed]])})
            c = (d["no_conflict"] == 0).cpu().numpy() & h
            det.append({"sampler_steps": N, "score": "conflict_flag", "wrong_cells": int((~right & h).sum()),
                        "flag_recall_of_wrong": (c & ~right).sum() / max(1, (~right & h).sum()),
                        "flag_precision": (c & ~right).sum() / max(1, c.sum()),
                        "failed_grids_with_a_flag": c[failed].any(1).mean()})
            print(pd.DataFrame(det[-5:]).round(3).to_string(index=False), flush=True)
    pd.DataFrame(rows).round(4).to_csv(out / "remask.csv", index=False)
    pd.DataFrame(det).round(4).to_csv(out / "detectors.csv", index=False)


if __name__ == "__main__":
    main()
