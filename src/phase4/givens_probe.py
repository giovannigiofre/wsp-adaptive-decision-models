"""Step 4.9a: what did the givens-only model learn? Measurements on saved models, no training.

Three final models at about 43,500 steps: supervised (4.1), givens-only (4.8e), hindsight loop (4.8d).
The stored solutions of the val set are read here for measurement only.

  m1  one forward pass on the puzzle as given: per blank cell, accuracy, probability on digits that a visible
      cell of the same row, column or box already holds ("excluded"), by number of digits left by that elimination
  m1b the same pass with a quarter of the givens hidden as well: accuracy on hidden givens against blanks
  m2  states with more visible cells than the givens (a share r of the blanks shown with the true digit):
      accuracy on the blanks still hidden
  m3  rollouts with 4 and 16 sampler steps: wrong share of the cells committed at each step, and the step of
      the first wrong cell per puzzle

    PYTHONPATH=src python -m phase4.givens_probe        # results/phase4/4.9a/*.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from phase2.data import digits_to_tokens
from phase2.evaluate import bootstrap_ci
from phase2.model import build_model
from phase2.sampler import sample
from phase2.utils import get_device, seed_all

from .data import load_sets
from .generate import cache_dir
from .sudoku9 import B as BOX, C as COL, CELLS, R as ROW

MODELS = {"supervised": ("sft", "sft"), "givens_long": ("4.8e", "givens_long"), "hindsight_long": ("4.8d", "hindsight_long")}
BUCKETS = ("1", "2", "3", "4+")


def peer_matrix(device) -> torch.Tensor:
    r, c, b = (torch.tensor(v, device=device) for v in (ROW, COL, BOX))
    same = (r[:, None] == r[None]) | (c[:, None] == c[None]) | (b[:, None] == b[None])
    return (same & ~torch.eye(CELLS, dtype=torch.bool, device=device)).float()


def excluded(grid: torch.Tensor, peers: torch.Tensor) -> torch.Tensor:
    """(B,81,9) bool: digit d+1 is held by a visible cell of the same row, column or box."""
    onehot = torch.nn.functional.one_hot(grid, 10)[..., 1:].float()
    return torch.einsum("ij,bjd->bid", peers, onehot) > 0


@torch.no_grad()
def forward(model, grid: torch.Tensor, batch: int = 512) -> torch.Tensor:
    return torch.cat([model(digits_to_tokens(grid[i:i + batch])).float().softmax(-1) for i in range(0, len(grid), batch)])


def cell_rows(model_name, state, probs, sol, target, kind, band, names, peers, tag):
    """One row per (band, candidate bucket): accuracy and rule statistics over the `target` cells."""
    used = excluded(state, peers)
    ncand = 9 - used.sum(-1)
    top = probs.argmax(-1)
    ok = (top + 1 == sol)
    mass = (probs * used).sum(-1)
    top_ex = used.gather(-1, top.unsqueeze(-1)).squeeze(-1)
    bucket = ncand.clamp(max=4)
    rows = []
    for b, bn in list(enumerate(names)) + [(-1, "all")]:
        pm = torch.ones(len(band), dtype=torch.bool) if b < 0 else band == b
        pm = pm.to(state.device)[:, None]
        for k, kn in list(zip((1, 2, 3, 4), BUCKETS)) + [(0, "any")]:
            m = target & pm & ((bucket == k) if k else torch.ones_like(target))
            n = int(m.sum())
            rows.append({"model": model_name, "state": tag, "cells": kind, "band": bn, "candidates": kn, "n": n,
                         "acc": ok[m].float().mean().item() if n else float("nan"),
                         "excluded_mass": mass[m].mean().item() if n else float("nan"),
                         "top1_excluded": top_ex[m].float().mean().item() if n else float("nan")})
    return rows, ok


def per_puzzle(ok: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    return (((ok & target).sum(1).float()) / target.sum(1).clamp_min(1)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase4.yaml")
    ap.add_argument("--size", default="medium")
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    cfg["model"].update(cfg["sizes"][a.size])
    dev = get_device(a.device or cfg["runtime"]["device"])
    sets, bands, names = load_sets(cfg, dev, (a.split,))
    ps, band = sets[a.split], bands[a.split]
    P, S = ps.puzzles, ps.solutions
    blank, given = P == 0, P != 0
    peers = peer_matrix(dev)
    hard = (band == names.index("hard")).numpy()

    # states, fixed once so that every model sees the same ones
    g = torch.Generator(device="cpu").manual_seed(0)
    hide = given & (torch.rand(P.shape, generator=g) < 0.25).to(dev)
    state_b = torch.where(hide, torch.zeros_like(P), P)
    shown = {r: blank & (torch.rand(P.shape, generator=g) < r).to(dev) for r in (0.25, 0.5, 0.75)}

    cells, reveal, steps, pp = [], [], [], {}
    for name, (exp, run) in MODELS.items():
        model = build_model(cfg).to(dev)
        model.load_state_dict(torch.load(cache_dir(cfg) / exp / run / "final.pt", map_location=dev)); model.eval()
        seed_all(0)
        # m1
        rows, ok = cell_rows(name, P, forward(model, P), S, blank, "blank", band, names, peers, "puzzle")
        cells += rows; pp[(name, "m1")] = per_puzzle(ok, blank)
        # m1b
        pr = forward(model, state_b)
        for kind, tgt in (("hidden given", hide), ("blank", blank)):
            rows, ok = cell_rows(name, state_b, pr, S, tgt, kind, band, names, peers, "quarter of givens hidden")
            cells += rows; pp[(name, f"m1b {kind}")] = per_puzzle(ok, tgt)
        # m2
        for r, sh in [(0.0, torch.zeros_like(blank))] + list(shown.items()):
            state = torch.where(sh, S, P)
            tgt = blank & ~sh
            ok = forward(model, state).argmax(-1) + 1 == S
            pp[(name, f"m2 {r}")] = per_puzzle(ok, tgt)
            for b, bn in list(enumerate(names)) + [(-1, "all")]:
                m = tgt & (torch.ones(len(band), dtype=torch.bool) if b < 0 else band == b).to(dev)[:, None]
                reveal.append({"model": name, "shown_share": r, "band": bn, "n": int(m.sum()), "acc": ok[m].float().mean().item()})
        # m3
        for N in (4, 16):
            o = sample(model, P, N, "confidence")
            wrong = (o.grid != S) & blank
            first = torch.where(wrong, o.commit_step, torch.full_like(o.commit_step, 10 ** 6)).min(1).values.cpu().numpy()
            pp[(name, f"m3 first {N}")] = first
            for b, bn in enumerate(names):
                pm = (band == b).to(dev)[:, None]
                nb = int((band == b).sum())
                for s in range(N):
                    at = (o.commit_step == s) & blank & pm
                    f = first[(band == b).numpy()]
                    steps.append({"model": name, "sampler_steps": N, "band": bn, "step": s + 1, "cells": int(at.sum()),
                                  "wrong_share": (wrong & at).sum().item() / max(int(at.sum()), 1),
                                  "mean_conf": o.commit_conf[at].mean().item() if at.any() else float("nan"),
                                  "puzzles_first_error_here": float((f == s).mean()),
                                  "puzzles_no_error": float((f == 10 ** 6).mean()), "puzzles": nb})
        del model

    out = Path("results/phase4/4.9a"); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cells).round(4).to_csv(out / "cells.csv", index=False)
    pd.DataFrame(reveal).round(4).to_csv(out / "reveal.csv", index=False)
    pd.DataFrame(steps).round(4).to_csv(out / "steps.csv", index=False)

    # paired differences per puzzle on the hard band, 95% bootstrap over puzzles
    pairs = []
    for key in ["m1", "m1b hidden given", "m1b blank"] + [f"m2 {r}" for r in (0.0, 0.25, 0.5, 0.75)]:
        for x, y in (("givens_long", "supervised"), ("hindsight_long", "givens_long")):
            d = (pp[(x, key)] - pp[(y, key)])[hard].astype(float)
            lo, hi = bootstrap_ci(d)
            pairs.append({"measure": key, "a": x, "b": y, "a_acc": pp[(x, key)][hard].mean(), "b_acc": pp[(y, key)][hard].mean(),
                          "a_minus_b": d.mean(), "lo": lo, "hi": hi})
    for x in MODELS:  # gap blank - hidden given inside one model
        d = (pp[(x, "m1b blank")] - pp[(x, "m1b hidden given")])[hard].astype(float)
        lo, hi = bootstrap_ci(d)
        pairs.append({"measure": "m1b blank minus hidden given", "a": x, "b": x, "a_acc": pp[(x, "m1b blank")][hard].mean(),
                      "b_acc": pp[(x, "m1b hidden given")][hard].mean(), "a_minus_b": d.mean(), "lo": lo, "hi": hi})
    for N in (4, 16):
        for x, y in (("givens_long", "supervised"), ("hindsight_long", "givens_long")):
            d = ((pp[(x, f"m3 first {N}")] == 0).astype(float) - (pp[(y, f"m3 first {N}")] == 0).astype(float))[hard]
            lo, hi = bootstrap_ci(d)
            pairs.append({"measure": f"m3 puzzles with an error at step 1, {N} steps", "a": x, "b": y,
                          "a_acc": (pp[(x, f"m3 first {N}")] == 0)[hard].mean(), "b_acc": (pp[(y, f"m3 first {N}")] == 0)[hard].mean(),
                          "a_minus_b": d.mean(), "lo": lo, "hi": hi})
    df = pd.DataFrame(pairs).round(4)
    df.to_csv(out / "pairs_hard.csv", index=False)
    pd.set_option("display.width", 250)
    c = pd.DataFrame(cells).round(4)
    print(c[(c.band == "hard")].to_string(index=False))
    r = pd.DataFrame(reveal).round(4)
    print(r[r.band == "hard"].to_string(index=False))
    s = pd.DataFrame(steps).round(4)
    print(s[(s.band == "hard") & (s.sampler_steps == 4)].to_string(index=False))
    print(s[(s.band == "hard") & (s.sampler_steps == 16)].to_string(index=False))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
