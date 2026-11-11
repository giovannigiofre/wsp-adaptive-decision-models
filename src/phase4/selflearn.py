"""Step 4.8: can the model learn Sudoku with no stored solutions? (docs/SELF_LEARNING_FROM_CHECKER.md, stage A)

A1, cold start on the givens: the givens of a puzzle are correct labels for their own cells. Blanks stay masked,
each given is masked with probability t ~ U(t_min, 1), cross-entropy on the masked givens weighted 1/t. The loss
takes the puzzles only; `phase4.train --loss givens` also replaces the stored solutions of the train set by zeros.

    PYTHONPATH=src python -m phase4.train --name a1 --size medium --loss givens --max-minutes 60 --no-plateau \
        --out results/phase4/4.8a --ckpt-dir ~/dllm-cache/phase4/4.8a/a1
    PYTHONPATH=src python -m phase4.selflearn --name a1          # valid share per band at 4, 16, 81 sampler steps
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from phase2.data import PuzzleSet, digits_to_tokens
from phase2.evaluate import bootstrap_ci, run_sampler
from phase2.model import build_model
from phase2.utils import get_device, seed_all

from .data import load_sets
from .generate import cache_dir


def givens_loss(model, puzzles: torch.Tensor, t_min: float, gen: torch.Generator | None = None) -> torch.Tensor:
    """puzzles: (B,L) digits, 0 = blank. No solution enters this function."""
    given = puzzles != 0
    B, L = puzzles.shape
    t = t_min + (1 - t_min) * torch.rand(B, 1, device=puzzles.device, generator=gen)
    mask = given & (torch.rand(B, L, device=puzzles.device, generator=gen) < t)
    logits = model(digits_to_tokens(torch.where(mask, torch.zeros_like(puzzles), puzzles)))
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), (puzzles - 1).clamp_min(0).reshape(-1),
                         reduction="none").view(B, L)
    return ((ce * mask).sum(1) / t.squeeze(1) / L).mean()


def without_solutions(ps: PuzzleSet) -> PuzzleSet:
    return PuzzleSet(ps.puzzles, torch.zeros_like(ps.solutions), ps.k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase4.yaml")
    ap.add_argument("--size", default="medium")
    ap.add_argument("--name", default="a1")
    ap.add_argument("--exp", default="4.8a")
    ap.add_argument("--split", default="val")
    ap.add_argument("--budgets", type=int, nargs="*", default=[4, 16, 81])
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    cfg["model"].update(cfg["sizes"][a.size])
    dev = get_device(a.device or cfg["runtime"]["device"])
    sets, bands, names = load_sets(cfg, dev, (a.split,))
    ps, band = without_solutions(sets[a.split]), bands[a.split]          # validity is checked from the rules only
    model = build_model(cfg).to(dev)
    model.load_state_dict(torch.load(cache_dir(cfg) / a.exp / a.name / "final.pt", map_location=dev)); model.eval()
    seed_all(0)
    rows = []
    for N in a.budgets:
        ok = run_sampler(model, ps, N, "confidence", batch=1024)["solved"].float().numpy()
        for b, name in enumerate(names):
            v = ok[(band == b).numpy()]
            lo, hi = bootstrap_ci(v)
            rows.append({"split": a.split, "sampler_steps": N, "band": name, "n": len(v), "valid": v.mean(), "lo": lo, "hi": hi})
            print(rows[-1], flush=True)
    out = Path("results/phase4") / a.exp
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(out / f"{a.name}_valid_{a.split}.csv", index=False)


if __name__ == "__main__":
    main()
