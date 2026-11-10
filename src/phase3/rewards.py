"""Reward functions for RL rollouts: (grid, puzzle) -> (N,) float. None of them uses the solution."""
from __future__ import annotations

from typing import Callable

import torch

from phase2.data import GROUPS_T, batch_validity


def binary(grid: torch.Tensor, puzzle: torch.Tensor) -> torch.Tensor:
    """1 if the grid is a valid completion of the puzzle, else 0 (the phase 2 reward)."""
    return batch_validity(grid, puzzle).float()


def graded(grid: torch.Tensor, puzzle: torch.Tensor) -> torch.Tensor:
    """Share of the 12 units (4 rows, 4 columns, 4 boxes) that contain each digit once; 0 if a given
    cell was changed. Equals 1 exactly when `binary` is 1."""
    g = grid.long()
    oh = torch.nn.functional.one_hot((g - 1).clamp(0, 3), 4)
    units_ok = (oh[:, GROUPS_T.to(g.device)].sum(2) == 1).all(2).float().mean(1)
    givens_ok = ((puzzle == 0) | (puzzle == g)).all(1) & ((g >= 1) & (g <= 4)).all(1)
    return units_ok * givens_ok


def lenient(grid: torch.Tensor, puzzle: torch.Tensor) -> torch.Tensor:
    """1 if every row and column contains each digit once and the givens are kept; boxes are ignored.
    A weaker checker than correctness: a Latin square that breaks the 2x2 boxes is rewarded."""
    g = grid.long()
    oh = torch.nn.functional.one_hot((g - 1).clamp(0, 3), 4)
    rows_cols_ok = (oh[:, GROUPS_T[:8].to(g.device)].sum(2) == 1).all(2).all(1)
    givens_ok = ((puzzle == 0) | (puzzle == g)).all(1) & ((g >= 1) & (g <= 4)).all(1)
    return (rows_cols_ok & givens_ok).float()


def cell_accuracy(grid: torch.Tensor, puzzle: torch.Tensor, solution: torch.Tensor) -> torch.Tensor:
    """Share of the hidden cells that equal the stored solution (phase 4.6). Unlike the rewards above it
    uses the solution; this is the Sudoku reward of the wd1 paper (arXiv 2507.08838). Any board size.
    Equals 1 exactly when the grid is the stored solution."""
    hidden = puzzle == 0
    return ((grid == solution) & hidden).sum(1).float() / hidden.sum(1).clamp_min(1)


REWARDS: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {"binary": binary, "graded": graded,
                                                                            "lenient": lenient}
