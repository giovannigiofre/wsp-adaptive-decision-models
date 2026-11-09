"""Unmasking order as a decision problem on a frozen denoiser (phase 3, track D).

One cell is committed per step and its value is always the denoiser's greedy prediction, so the only
freedom is *which* masked cell to commit next.

oracle (3.4a): commit a masked cell whose greedy prediction equals the true solution (the most confident
such cell). If there is none the puzzle is lost: commit the most confident masked cell and carry on.
The oracle uses the solution, so it is an analysis tool, not a method. Its solve rate is reached by
some order, hence it is a lower bound on what the best possible order achieves.

PYTHONPATH=src python -m phase3.order_policy oracle
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from phase2.data import PuzzleSet, batch_validity, build_datasets, digits_to_tokens
from phase2.evaluate import bootstrap_ci, run_sampler
from phase2.model import build_model
from phase2.utils import CACHE as PHASE2_CACHE
from phase2.utils import env_record, get_device, load_cfg

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
    ap.add_argument("what", choices=["oracle"])
    ap.add_argument("--config", default="configs/phase3.yaml")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    {"oracle": oracle_cmd}[a.what](a)


if __name__ == "__main__":
    main()
