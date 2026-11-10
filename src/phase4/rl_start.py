"""Step 4.3: the two quantities that decided things in phase 3, measured at candidate RL starts.

For the plateau model and each (band, sampler steps): G = 8 sampled rollouts (temperature 1, `confidence` order) per
train puzzle, then the share of valid rollouts and the share of groups with unequal rewards (binary reward).

    PYTHONPATH=src python -m phase4.rl_start --size medium
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml

from phase2.data import batch_validity
from phase2.model import build_model
from phase2.sampler import sample
from phase2.utils import get_device, seed_all

from .data import load_sets
from .generate import cache_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase4.yaml")
    ap.add_argument("--size", default="medium")
    ap.add_argument("--n-puzzles", type=int, default=512)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--steps", type=int, nargs="*", default=[4, 8, 16])
    ap.add_argument("--out", default="results/phase4/4.3")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    cfg["model"].update(cfg["sizes"][a.size])
    dev = get_device(cfg["runtime"]["device"])
    sets, bands, names = load_sets(cfg, dev, splits=("train",))
    model = build_model(cfg).to(dev)
    model.load_state_dict(torch.load(cache_dir(cfg) / "sft" / "sft" / "final.pt", map_location=dev))
    model.eval()
    seed_all(0)
    rows = []
    for b, name in enumerate(names):
        idx = (bands["train"] == b).nonzero().squeeze(1)
        idx = idx[torch.randperm(len(idx))[:a.n_puzzles]]
        p = sets["train"].puzzles[idx.to(dev)]
        rep = p.repeat_interleave(a.group, 0)
        for S in a.steps:
            greedy = batch_validity(sample(model, p, S, "confidence", 0.0).grid, p).float().mean().item()
            r = torch.cat([batch_validity(sample(model, rep[i:i + 1024], S, "confidence", 1.0).grid, rep[i:i + 1024])
                           for i in range(0, len(rep), 1024)]).float().view(-1, a.group).cpu()
            n_valid = r.sum(1)
            rows.append({"band": name, "sampler_steps": S, "n_puzzles": len(p), "group": a.group,
                         "greedy_solve": greedy, "valid_rollouts": r.mean().item(),
                         "groups_unequal": ((n_valid > 0) & (n_valid < a.group)).float().mean().item(),
                         "groups_all_valid": (n_valid == a.group).float().mean().item(),
                         "groups_none_valid": (n_valid == 0).float().mean().item()})
            print(rows[-1], flush=True)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(out / "rl_start.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
