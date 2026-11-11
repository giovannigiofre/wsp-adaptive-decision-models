"""Step 4.6b: how good must a detector of wrong cells be, and is the information in the frozen model?
No model training (docs/RL_DLLM_DIRECTIONS.md, direction 5, the gate before any detector is trained with RL).

4.5b: with exactly the wrong cells masked, the model repairs every hard test grid in two rounds; no score the
model or the rules give detects them (AUROC 0.41 to 0.69). Two measurements here:

  degraded oracle  fill-then-fix as in 4.5b, with a detector that masks each wrong cell with probability rho
                   (recall) and each right filled cell with probability phi (false alarms). Which (rho, phi)
                   still beat the plain sampler at the same number of forward passes?
  probes           logistic regression on the frozen model's last-layer features of a filled cell (grid fully
                   visible, or the cell masked), trained on rollouts of hard TRAIN puzzles with right/wrong
                   labels, tested on the hard test band: AUROC on all cells and inside failed grids, its
                   (rho, phi) at threshold 0.5, and the solve rate of fill-then-fix with the probe as detector.

    PYTHONPATH=src python -m phase4.detector_gate --size medium
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
from phase2.sampler import sample
from phase2.utils import get_device, seed_all

from .data import load_sets
from .generate import cache_dir


@torch.no_grad()
def features(model, grid: torch.Tensor) -> torch.Tensor:
    """Per cell: last-layer features (after the final norm), the nine digit probabilities and the probability
    of the digit written in `grid` (0 where the cell is empty): (B,81,d+10)."""
    got = {}
    h = model.norm.register_forward_hook(lambda m, i, o: got.__setitem__("x", o))
    p = model(digits_to_tokens(grid)).float().softmax(-1)
    h.remove()
    own = p.gather(-1, (grid - 1).clamp_min(0).unsqueeze(-1)) * (grid > 0).unsqueeze(-1)
    return torch.cat([got["x"].float(), p, own], -1)


@torch.no_grad()
def cell_features(model, grid: torch.Tensor, hidden: torch.Tensor, kind: str, parts: int = 4, seed: int = 0) -> torch.Tensor:
    """`visible`: one pass, nothing masked. `leave_out`: `parts` passes, each hidden cell masked in one of them."""
    if kind == "visible":
        return features(model, grid)
    g = torch.Generator(device="cpu").manual_seed(seed)
    part = torch.randint(0, parts, grid.shape, generator=g).to(grid.device)
    out = None
    for j in range(parts):
        m = hidden & (part == j)
        f = features(model, torch.where(m, torch.zeros_like(grid), grid))
        f[..., -1] = f[..., -10:-1].gather(-1, (grid - 1).unsqueeze(-1)).squeeze(-1)     # written digit, cell masked
        out = f if out is None else torch.where(m[..., None], f, out)
    return out


@torch.no_grad()
def fix(model, puzzles: torch.Tensor, fill_steps: int, rounds: int, detect):
    """Fill with the plain greedy sampler, then `rounds` passes of mask-and-re-predict; `detect(grid) -> (B,81) bool`."""
    hidden = puzzles == 0
    grid = sample(model, puzzles, fill_steps, "confidence", 0.0).grid.clone()
    for _ in range(rounds):
        rem = detect(grid) & hidden
        digit = model(digits_to_tokens(torch.where(rem, torch.zeros_like(grid), grid))).argmax(-1) + 1
        grid = torch.where(rem, digit, grid)
    return grid


def degraded(solutions: torch.Tensor, rho: float, phi: float, gen: torch.Generator):
    def detect(grid):
        u = torch.rand(grid.shape, generator=gen).to(grid.device)
        wrong = grid != solutions
        return (wrong & (u < rho)) | (~wrong & (u < phi))
    return detect


class Probe:
    """Standardised logistic regression; score = P(cell is wrong)."""
    def __init__(self, X: np.ndarray, wrong: np.ndarray):
        from sklearn.linear_model import LogisticRegression
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.clf = LogisticRegression(C=0.1, max_iter=300, class_weight="balanced").fit((X - self.mu) / self.sd, wrong)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba((X - self.mu) / self.sd)[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase4.yaml")
    ap.add_argument("--size", default="medium")
    ap.add_argument("--band", default="hard")
    ap.add_argument("--budgets", type=int, nargs="*", default=[4, 8])
    ap.add_argument("--rho", type=float, nargs="*", default=[0.5, 0.7, 0.9, 1.0])
    ap.add_argument("--phi", type=float, nargs="*", default=[0.0, 0.1, 0.3, 0.6])
    ap.add_argument("--probe-puzzles", type=int, default=3000, help="hard train puzzles used to fit each probe")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/phase4/4.6b")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    cfg["model"].update(cfg["sizes"][a.size])
    dev = get_device(a.device or cfg["runtime"]["device"])
    sets, bands, names = load_sets(cfg, dev, splits=("train", "test"))
    cut = {k: sets[k].subset((bands[k] == names.index(a.band)).to(dev)) for k in ("train", "test")}
    P, S = cut["test"].puzzles, cut["test"].solutions
    Ptr, Str = cut["train"].puzzles[:a.probe_puzzles], cut["train"].solutions[:a.probe_puzzles]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model = build_model(cfg).to(dev)
    model.load_state_dict(torch.load(cache_dir(cfg) / "sft" / "sft" / "final.pt", map_location=dev)); model.eval()
    seed_all(0)
    gen = torch.Generator(device="cpu").manual_seed(0)
    solve = lambda g: batch_validity(g, P).float().cpu().numpy()                        # noqa: E731
    base = {N: solve(sample(model, P, N, "confidence", 0.0).grid) for N in a.budgets}

    rows = []
    def add(N, R, arm, v, **kw):
        lo, hi = bootstrap_ci(v - base[N])
        rows.append({"forward_passes": N, "fix_rounds": R, "arm": arm, **kw, "solve": v.mean(), "plain": base[N].mean(),
                     "minus_plain": (v - base[N]).mean(), "lo": lo, "hi": hi})
        print(rows[-1], flush=True)

    for N in a.budgets:                                                                  # degraded oracle
        R = max(1, N // 4)
        for rho in a.rho:
            for phi in a.phi:
                add(N, R, "degraded_oracle", solve(fix(model, P, N - R, R, degraded(S, rho, phi, gen))), rho=rho, phi=phi)
    pd.DataFrame(rows).round(4).to_csv(out / "degraded_oracle.csv", index=False)

    prow = []
    for N in a.budgets:                                                                  # probes on frozen features
        R = max(1, N // 4)
        fill = N - R
        gtr = sample(model, Ptr, fill, "confidence", 0.0).grid
        gte = sample(model, P, fill, "confidence", 0.0).grid
        htr, hte = (Ptr == 0).cpu().numpy(), (P == 0).cpu().numpy()
        wtr, wte = (gtr != Str).cpu().numpy(), (gte != S).cpu().numpy()
        failed = wte.any(1)
        for kind in ("visible", "leave_out"):
            Xtr = cell_features(model, gtr, Ptr == 0, kind).cpu().numpy()
            Xte = cell_features(model, gte, P == 0, kind, seed=1).cpu().numpy()
            for feats, sl in (("features+probs", slice(None)), ("probs only", slice(-10, None))):
                pr = Probe(Xtr[htr][:, sl], wtr[htr])
                s = pr(Xte[hte][:, sl])
                full = np.zeros(hte.shape); full[hte] = s
                flag = (full > 0.5) & hte
                prow.append({"fill_steps": fill, "pass": kind, "inputs": feats, "train_cells": int(htr.sum()),
                             "wrong_share_test": wte[hte].mean(),
                             "auroc_all_cells": auroc(-s, ~wte[hte]),
                             "auroc_failed_grids": auroc(-full[failed][hte[failed]], ~wte[failed][hte[failed]]),
                             "recall_rho": (flag & wte).sum() / max(1, wte.sum()),
                             "false_alarm_phi": (flag & ~wte).sum() / max(1, (hte & ~wte).sum())})
                print(prow[-1], flush=True)
                if kind == "visible" and feats == "features+probs":                     # the probe as a real detector
                    def detect(grid, pr=pr):
                        X = features(model, grid).cpu().numpy()
                        return torch.from_numpy(pr(X.reshape(-1, X.shape[-1])).reshape(grid.shape) > 0.5).to(grid.device)
                    add(N, R, "fix_probe_visible", solve(fix(model, P, fill, R, detect)))
        # one round costs one detector pass too; the same arm with that pass charged to the budget
        if fill - R >= 1:
            add(N, R, "fix_probe_visible_detector_pass_charged", solve(fix(model, P, fill - R, R, detect)))
    pd.DataFrame(rows).round(4).to_csv(out / "degraded_oracle.csv", index=False)
    pd.DataFrame(prow).round(4).to_csv(out / "probes.csv", index=False)


if __name__ == "__main__":
    main()
