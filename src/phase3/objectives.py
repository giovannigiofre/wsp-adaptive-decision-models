"""RL objectives for the toy Sudoku dLLM: how a group of rewards becomes a loss on log p_hat.

An objective maps rewards r (B, G) to `GroupWeights`. The loss is then
    mean over kept groups of sum_i coef_i * log p_hat(o_i)
so a negative coefficient raises a completion's likelihood and a positive one lowers it.

wd1(tau, beta): A = r - group mean, w+ = softmax(A / tau), w- = softmax(-A / tau), coef = beta * w- - w+.
  beta = 1 is the phase 2 wd1 loss; beta = 0 is its positive-only ablation (same numbers as
  phase2.rl_wd1.group_weights / wd1_loss, checked in tests/test_objectives_sign.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class GroupWeights:
    coef: torch.Tensor   # (B, G) coefficient of log p_hat in the loss
    w_pos: torch.Tensor  # (B, G) positive weights, kept for diagnostics
    keep: torch.Tensor   # (B,) bool, groups whose rewards are not all equal


def wd1(r: torch.Tensor, tau: float = 1.0, beta: float = 1.0) -> GroupWeights:
    A = r - r.mean(1, keepdim=True)
    keep = r.max(1).values != r.min(1).values
    w_pos = torch.softmax(A / tau, dim=1)
    w_neg = torch.softmax(-A / tau, dim=1)
    return GroupWeights(beta * w_neg - w_pos, w_pos, keep)


OBJECTIVES: dict[str, Callable[..., GroupWeights]] = {"wd1": wd1}


def weighted_loglik_loss(logp: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
    """logp, coef: (n_groups, G) for kept groups."""
    return (coef * logp).sum(1).mean()


def weight_diagnostics(w_pos: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    """w_pos (n, G), valid (n, G) bool, kept groups only. Means over groups."""
    return {
        "w_entropy": -(w_pos * w_pos.clamp_min(1e-12).log()).sum(1).mean().item(),
        "w_ess": (1.0 / (w_pos ** 2).sum(1)).mean().item(),          # effective sample size, max G
        "w_pos_on_invalid": (w_pos * (~valid)).sum(1).mean().item(),  # share of w+ mass on invalid samples
    }
