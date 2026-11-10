"""Step 4.4b: three measurements on the plateau model, no training (docs/RL_DLLM_SURVEY.md, proposal 1).

a) Do wrong cells follow an earlier mistake, and are first errors conflicts between cells committed together?
b) Does a sampled unmasking order (Gumbel noise on log confidence) give more groups with unequal rewards?
c) Are valid rollouts on train puzzles just the stored solution?

    PYTHONPATH=src python -m phase4.pre_rl_checks --size medium
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from phase2.data import batch_validity
from phase2.evaluate import run_sampler
from phase2.model import build_model
from phase2.sampler import gumbel_like, sample  # noqa: F401 (gumbel_like re-exported for the tests)
from phase2.utils import get_device, seed_all

from .data import load_sets
from .generate import cache_dir


def peers9() -> np.ndarray:
    """(81, 81) bool: the two cells share a row, a column or a box (a cell is not its own peer)."""
    i = np.arange(81)
    r, c = i // 9, i % 9
    b = (r // 3) * 3 + c // 3
    p = (r[:, None] == r[None]) | (c[:, None] == c[None]) | (b[:, None] == b[None])
    return p & ~np.eye(81, dtype=bool)


@torch.no_grad()
def sample_gumbel_order(model, puzzles: torch.Tensor, steps: int, order_temp: float, temperature: float = 1.0):
    """phase2.sampler.sample with the `confidence` order replaced by a draw without replacement from a
    Plackett-Luce distribution with weights confidence^(1/order_temp) (Gumbel top-k)."""
    return sample(model, puzzles, steps, "confidence", temperature, order_temp=order_temp).grid


def error_structure(grid, step, puzzle, sol, S: int) -> tuple[dict, list[dict]]:
    """grid/step/puzzle/sol: (n, 81) numpy. Returns summary numbers and per-step conditional error rates."""
    P = peers9()
    hidden = puzzle == 0
    wrong = hidden & (grid != sol)
    failed = wrong.any(1)
    first = np.where(wrong, step, 10 ** 6).min(1)                       # step of the first wrong commit
    n_first = (wrong & (step == first[:, None])).sum(1)[failed]
    n_after = (wrong & (step > first[:, None])).sum(1)[failed]
    kinds = {"visible_conflict": 0, "same_step_conflict": 0, "no_conflict": 0}
    puzzles_with = dict.fromkeys(kinds, 0)
    for i in np.flatnonzero(failed):
        seen = set()
        visible = (~hidden[i]) | (step[i] < first[i])                  # shown to the model when it chose
        together = hidden[i] & (step[i] == first[i])
        for cell in np.flatnonzero(wrong[i] & together):
            same = P[cell] & (grid[i] == grid[i, cell])
            kind = ("visible_conflict" if (same & visible).any() else
                    "same_step_conflict" if (same & together).any() else "no_conflict")
            kinds[kind] += 1; seen.add(kind)
        for kind in seen:
            puzzles_with[kind] += 1
    n_cells = sum(kinds.values())
    summ = {"sampler_steps": S, "n_puzzles": int(len(grid)), "failed": int(failed.sum()),
            "wrong_cells": int(wrong.sum()), "wrong_cells_per_failed_puzzle": float(wrong.sum() / max(failed.sum(), 1)),
            "wrong_at_first_error_step_mean": float(n_first.mean()), "wrong_after_first_error_step_mean": float(n_after.mean()),
            "share_of_wrong_cells_after_first_error_step": float(n_after.sum() / max(wrong.sum(), 1)),
            "first_error_step_mean": float(first[failed].mean()),
            **{f"first_step_wrong_cells_{k}": v / max(n_cells, 1) for k, v in kinds.items()},
            **{f"failed_puzzles_with_{k}": v / max(failed.sum(), 1) for k, v in puzzles_with.items()}}
    rows = []
    for s in range(S):
        at = hidden & (step == s)
        before = (wrong & (step < s)).any(1)                           # an earlier wrong commit exists
        for label, m in (("no_earlier_error", ~before), ("after_earlier_error", before)):
            n = int(at[m].sum())
            rows.append({"sampler_steps": S, "step": s, "condition": label, "puzzles": int(m.sum()), "cells": n,
                         "error_rate": float(wrong[m][at[m]].mean()) if n else float("nan")})
    return summ, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase4.yaml")
    ap.add_argument("--size", default="medium")
    ap.add_argument("--band", default="hard")
    ap.add_argument("--n-puzzles", type=int, default=512)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--rollout-steps", type=int, default=4)
    ap.add_argument("--order-temps", type=float, nargs="*", default=[0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--out", default="results/phase4/4.4b")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    cfg["model"].update(cfg["sizes"][a.size])
    dev = get_device(cfg["runtime"]["device"])
    sets, bands, names = load_sets(cfg, dev, splits=("train", "test"))
    b = names.index(a.band)
    model = build_model(cfg).to(dev)
    model.load_state_dict(torch.load(cache_dir(cfg) / "sft" / "sft" / "final.pt", map_location=dev))
    model.eval()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # --- a) error structure, greedy decoding of the hard test band ------------------------------------------
    test = sets["test"].subset((bands["test"] == b).to(dev))
    summ, steps = [], []
    for S in (a.rollout_steps, 16):
        seed_all(0)
        o = run_sampler(model, test, S, "confidence", 0.0, batch=1024)
        s, r = error_structure(o["grid"].numpy(), o["commit_step"].numpy(), test.puzzles.cpu().numpy(),
                               test.solutions.cpu().numpy(), S)
        s["solved_but_not_stored_solution"] = int((o["solved"].numpy() & (o["grid"].numpy() != test.solutions.cpu().numpy()).any(1)).sum())
        summ.append(s); steps += r
        print(json.dumps(s, indent=1), flush=True)
    pd.DataFrame(summ).round(4).to_csv(out / "error_structure.csv", index=False)
    pd.DataFrame(steps).round(4).to_csv(out / "error_rate_by_step.csv", index=False)
    print(pd.DataFrame(steps).round(3).to_string(index=False))

    # --- b) sampled order and c) valid rollouts vs stored solution, train puzzles ----------------------------
    seed_all(0)
    idx = (bands["train"] == b).nonzero().squeeze(1)
    idx = idx[torch.randperm(len(idx))[:a.n_puzzles]].to(dev)
    p, sol = sets["train"].puzzles[idx], sets["train"].solutions[idx]
    rep, rsol = p.repeat_interleave(a.group, 0), sol.repeat_interleave(a.group, 0)
    rows, same = [], {}
    for tau in [0.0] + a.order_temps:                                  # 0.0 = the deterministic order used so far
        grids = torch.cat([sample(model, rep[i:i + 1024], a.rollout_steps, "confidence", 1.0).grid if tau == 0 else
                           sample_gumbel_order(model, rep[i:i + 1024], a.rollout_steps, tau)
                           for i in range(0, len(rep), 1024)])
        v = batch_validity(grids, rep)
        r = v.float().view(-1, a.group).cpu()
        n = r.sum(1)
        rows.append({"order_temp": tau, "valid_rollouts": r.mean().item(),
                     "groups_unequal": ((n > 0) & (n < a.group)).float().mean().item(),
                     "groups_all_valid": (n == a.group).float().mean().item(),
                     "groups_none_valid": (n == 0).float().mean().item(),
                     "valid_equal_to_stored_solution": (grids[v] == rsol[v]).all(1).float().mean().item()})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).round(4).to_csv(out / "order_noise.csv", index=False)


if __name__ == "__main__":
    main()
