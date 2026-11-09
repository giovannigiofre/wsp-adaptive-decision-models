"""Reward functions for RL rollouts: (grid, puzzle) -> (N,) float. None of them uses the solution."""
from __future__ import annotations

from typing import Callable

import torch

from phase2.data import batch_validity


def binary(grid: torch.Tensor, puzzle: torch.Tensor) -> torch.Tensor:
    """1 if the grid is a valid completion of the puzzle, else 0 (the phase 2 reward)."""
    return batch_validity(grid, puzzle).float()


REWARDS: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {"binary": binary}
