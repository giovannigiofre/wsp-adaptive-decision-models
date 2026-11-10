"""Splits by solution, unique-solution puzzle generation, token tensors, batched validity."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .sudoku import GROUPS, solutions_array

PAD, MASK = 0, 1
DIGIT_OFFSET = 1  # digit d (1..4) -> token d + 1 (2..5)
VOCAB = 6
GROUPS_T = torch.tensor(GROUPS, dtype=torch.long)  # (12, 4)


def digits_to_tokens(grid: torch.Tensor) -> torch.Tensor:
    """0 (empty) -> MASK, d -> d + 1."""
    return torch.where(grid == 0, torch.full_like(grid, MASK), grid + DIGIT_OFFSET)


@dataclass
class PuzzleSet:
    puzzles: torch.Tensor    # (n, 16) digits, 0 = hidden
    solutions: torch.Tensor  # (n, 16) digits
    k: torch.Tensor          # (n,) number of hidden cells

    def __len__(self) -> int:
        return len(self.k)

    def to(self, device) -> "PuzzleSet":
        return PuzzleSet(self.puzzles.to(device), self.solutions.to(device), self.k.to(device))

    def subset(self, idx) -> "PuzzleSet":
        return PuzzleSet(self.puzzles[idx], self.solutions[idx], self.k[idx])


def split_solutions(seed: int = 0, n_train: int = 200, n_val: int = 44, n_test: int = 44):
    sols = solutions_array()
    assert n_train + n_val + n_test == len(sols)
    perm = np.random.default_rng(seed).permutation(len(sols))
    return {"train": sols[perm[:n_train]], "val": sols[perm[n_train:n_train + n_val]],
            "test": sols[perm[n_train + n_val:]]}


def make_puzzles(split_sols: np.ndarray, per_solution_per_k: int, ks=range(6, 13), seed: int = 0,
                 max_tries: int = 200) -> PuzzleSet:
    """For each solution and each k, draw up to `per_solution_per_k` distinct unique-solution puzzles."""
    rng = np.random.default_rng(seed)
    all_sols = solutions_array()
    P, S, K = [], [], []
    for sol in split_sols:
        for k in ks:
            seen: set[bytes] = set()
            for _ in range(max_tries):
                if len(seen) >= per_solution_per_k:
                    break
                hidden = rng.choice(16, size=k, replace=False)
                p = sol.copy()
                p[hidden] = 0
                key = p.tobytes()
                if key in seen:
                    continue
                given = p != 0
                if int(np.all(all_sols[:, given] == p[given], axis=1).sum()) == 1:
                    seen.add(key)
                    P.append(p); S.append(sol); K.append(k)
    return PuzzleSet(torch.tensor(np.array(P)), torch.tensor(np.array(S)), torch.tensor(K))


def build_datasets(cfg: dict, seed: int = 0) -> dict[str, PuzzleSet]:
    d = cfg["data"]
    splits = split_solutions(seed, d["n_train_solutions"], d["n_val_solutions"], d["n_test_solutions"])
    ks = d["hidden_cells"]
    per = {"train": d.get("train_per_solution_per_k", 20), "val": d.get("val_per_solution_per_k", 1),
           "test": d.get("test_per_solution_per_k", 4)}
    return {name: make_puzzles(sols, per[name], ks, seed=seed + 1000 * i)
            for i, (name, sols) in enumerate(splits.items())}


_GROUPS: dict[tuple[int, str], torch.Tensor] = {}


def groups_t(n: int, device) -> torch.Tensor:
    """(3n, n) cell indices of the rows, columns and boxes of an n x n grid (n = 4 or 9)."""
    key = (n, str(device))
    if key not in _GROUPS:
        if n == 4:
            g = GROUPS_T
        else:
            b = int(round(n ** 0.5))
            idx = torch.arange(n * n)
            box = (idx // (n * b)) * b + (idx % n) // b
            g = torch.stack([idx.view(n, n)[r] for r in range(n)] + [idx.view(n, n)[:, c] for c in range(n)]
                            + [idx[box == k] for k in range(n)])
        _GROUPS[key] = g.to(device)
    return _GROUPS[key]


def batch_validity(grid: torch.Tensor, puzzle: torch.Tensor) -> torch.Tensor:
    """Vectorised `sudoku.validity`. grid, puzzle: (B, n*n) digits, n = 4 or 9. Returns (B,) bool."""
    g = grid.long()
    n = int(round(g.shape[1] ** 0.5))
    in_range = ((g >= 1) & (g <= n)).all(1)
    oh = torch.nn.functional.one_hot((g - 1).clamp(0, n - 1), n)      # (B,L,n)
    counts = oh[:, groups_t(n, g.device)].sum(2)                        # (B,3n,n)
    groups_ok = (counts == 1).all(2).all(1)
    givens_ok = ((puzzle == 0) | (puzzle == g)).all(1)
    return in_range & groups_ok & givens_ok


def main():  # PYTHONPATH=src python -m phase2.data --config configs/phase2.yaml
    import argparse
    import time
    from pathlib import Path

    import pandas as pd
    import yaml

    from .sudoku import enumerate_solutions

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase2.yaml")
    ap.add_argument("--out", default="results/phase2/data_stats.csv")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    t0 = time.time()
    n_sol = len(enumerate_solutions())
    ds = build_datasets(cfg, seed=cfg["data"].get("split_seed", 0))
    rows = [{"split": n, "k": int(k), "n_puzzles": int((d.k == k).sum()),
             "n_solutions": len({tuple(s) for s in d.solutions[d.k == k].tolist()})}
            for n, d in ds.items() for k in sorted(set(d.k.tolist()))]
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"solutions enumerated: {n_sol}; datasets built in {time.time() - t0:.1f}s")
    print(df.pivot(index="k", columns="split", values="n_puzzles").to_string())
    print({n: len(d) for n, d in ds.items()})


if __name__ == "__main__":
    main()
