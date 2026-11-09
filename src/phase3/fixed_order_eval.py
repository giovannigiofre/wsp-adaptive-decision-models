"""Solve rate of saved checkpoints when cells are filled in a fixed row-major order, one per step (greedy
values), next to the usual confidence order. Used by 3.6c to ask whether a model trained with the
fixed-order likelihood became good along that order only.

PYTHONPATH=src python -m phase3.fixed_order_eval
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from phase2.data import batch_validity, build_datasets, digits_to_tokens
from phase2.model import build_model
from phase2.utils import CACHE as PHASE2_CACHE
from phase2.utils import get_device, load_cfg

from .rl import CACHE, solve_rate


@torch.no_grad()
def row_major_solve_rate(model, ps) -> float:
    grid = ps.puzzles.clone()
    for i in range(16):
        hidden = grid[:, i] == 0
        if hidden.any():
            pred = model(digits_to_tokens(grid)).float()[:, i].argmax(-1) + 1
            grid[:, i] = torch.where(hidden, pred, grid[:, i])
    return batch_validity(grid, ps.puzzles).float().mean().item()


def main():
    cfg = load_cfg("configs/phase3.yaml")
    base = load_cfg(cfg["base_config"])
    dev = get_device(base["runtime"]["device"])
    test = build_datasets(base, base["data"].get("split_seed", 0))["test"].to(dev)
    model = build_model(base).to(dev).eval()
    ckpts = {"start (SFT step 120)": [PHASE2_CACHE / cfg["init"]["checkpoint"]]}
    for label, exp, name in [("wd1, one-step", "3.1b", "wd1_beta1_tau1"), ("wd1, sequential", "3.6a", "wd1_beta1_tau1_seqlik"),
                             ("wd1 floored, one-step", "3.1c", "wd1_beta1_floor-12_tau1"),
                             ("wd1 floored, sequential", "3.6a", "wd1_beta1_floor-9_tau1_seqlik"),
                             ("positive-only, one-step", "3.1b", "wd1_beta0_tau1"),
                             ("positive-only, sequential", "3.6a", "wd1_beta0_tau1_seqlik")]:
        ckpts[label] = [CACHE / exp / "ckpt" / f"{name}_seed{s}" / "final.pt" for s in range(3)]
    rows = []
    for label, paths in ckpts.items():
        for p in paths:
            model.load_state_dict(torch.load(p, map_location=dev))
            rows.append({"model": label, "checkpoint": p.parent.name if p.name == "final.pt" else p.name,
                         "test_confidence_order_8_steps": solve_rate(model, test, 8, "confidence"),
                         "test_confidence_order_16_steps": solve_rate(model, test, 16, "confidence"),
                         "test_row_major_16_steps": row_major_solve_rate(model, test)})
    df = pd.DataFrame(rows)
    out = Path("results/phase3/3.6c")
    out.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(out / "fixed_order_eval.csv", index=False)
    print(df.groupby("model", sort=False).mean(numeric_only=True).round(3).to_string())


if __name__ == "__main__":
    main()
