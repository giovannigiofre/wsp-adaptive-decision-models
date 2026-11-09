"""Solve rates and calibration metrics (ECE, AUROC, reliability bins)."""
from __future__ import annotations

import numpy as np
import torch

from .data import PuzzleSet, batch_validity
from .sampler import sample


def bootstrap_ci(x: np.ndarray, n: int = 2000, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


@torch.no_grad()
def run_sampler(model, ps: PuzzleSet, steps: int, strategy: str = "confidence",
                temperature: float = 0.0, batch: int = 2048) -> dict[str, torch.Tensor]:
    was_training = model.training
    model.eval()
    outs = {"grid": [], "commit_step": [], "commit_conf": [], "solved": []}
    for i in range(0, len(ps), batch):
        p = ps.puzzles[i:i + batch]
        o = sample(model, p, steps, strategy, temperature)
        outs["grid"].append(o.grid); outs["commit_step"].append(o.commit_step)
        outs["commit_conf"].append(o.commit_conf); outs["solved"].append(batch_validity(o.grid, p))
    model.train(was_training)
    return {k: torch.cat(v).cpu() for k, v in outs.items()}


def solve_rate(model, ps: PuzzleSet, steps: int = 8, strategy: str = "confidence") -> float:
    return run_sampler(model, ps, steps, strategy)["solved"].float().mean().item()


def reliability_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10):
    """Equal-width bins on [0,1]. Returns (bin_centers, accuracy, mean_conf, counts)."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    acc, mc, cnt = np.full(n_bins, np.nan), np.full(n_bins, np.nan), np.zeros(n_bins, int)
    for b in range(n_bins):
        sel = idx == b
        cnt[b] = sel.sum()
        if cnt[b]:
            acc[b], mc[b] = correct[sel].mean(), conf[sel].mean()
    return (edges[:-1] + edges[1:]) / 2, acc, mc, cnt


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    _, acc, mc, cnt = reliability_bins(conf, correct, n_bins)
    ok = cnt > 0
    return float((cnt[ok] / cnt.sum() * np.abs(acc[ok] - mc[ok])).sum())


def auroc(conf: np.ndarray, correct: np.ndarray) -> float:
    """AUROC of confidence for separating correct from wrong; nan if only one class present."""
    from sklearn.metrics import roc_auc_score
    if correct.min() == correct.max():
        return float("nan")
    return float(roc_auc_score(correct.astype(int), conf))


def calibration_data(out: dict, ps: PuzzleSet):
    """Cell level: commit confidence vs (cell == unique solution). Puzzle level: mean conf vs solved."""
    hidden = (ps.puzzles == 0).cpu()
    cell_conf = out["commit_conf"][hidden].numpy()
    cell_ok = (out["grid"] == ps.solutions.cpu())[hidden].numpy()
    conf = torch.nan_to_num(out["commit_conf"], nan=0.0)
    puz_conf = ((conf * hidden).sum(1) / hidden.sum(1)).numpy()
    puz_ok = out["solved"].numpy()
    return cell_conf, cell_ok, puz_conf, puz_ok
