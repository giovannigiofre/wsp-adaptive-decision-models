"""Iterative unmasking sampler for the toy dLLM: strategies x step budgets.

Strategy scores (higher = commit earlier), same definitions as phase 1 (docs/PHASE1_LLADA_ON_MAC.md 1.2):
  random     uniform random
  confidence probability of the value being committed (= max softmax prob when greedy; this is
             LLaDA's `low_confidence` remasking, which scores the sampled token's probability)
  margin     top-1 prob - top-2 prob
  entropy    -entropy of the predictive distribution
`score_positions` / `select_topk` are the shared selection logic; they only need (B, L, V) probs.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .data import digits_to_tokens

STRATEGIES = ("random", "confidence", "margin", "entropy")


def score_positions(probs: torch.Tensor, chosen: torch.Tensor, strategy: str,
                    generator: torch.Generator | None = None) -> torch.Tensor:
    """probs (B,L,V), chosen (B,L) class indices -> (B,L) scores, higher = commit first."""
    if strategy == "random":
        return torch.rand(probs.shape[:2], generator=generator, device=probs.device)
    if strategy == "confidence":
        return probs.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
    if strategy == "margin":
        top2 = probs.topk(2, dim=-1).values
        return top2[..., 0] - top2[..., 1]
    if strategy == "entropy":
        return (probs * probs.clamp_min(1e-12).log()).sum(-1)  # = -entropy
    raise ValueError(f"unknown strategy {strategy!r}")


def select_topk(scores: torch.Tensor, masked: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Pick, per row, the k[b] highest-scoring masked positions. Returns a (B,L) bool mask."""
    s = scores.masked_fill(~masked, float("-inf"))
    order = s.argsort(dim=1, descending=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(s.shape[1], device=s.device).expand_as(order))
    return (rank < k.unsqueeze(1)) & masked


@dataclass
class SampleOut:
    grid: torch.Tensor         # (B,16) digits
    commit_step: torch.Tensor  # (B,16) step index at which each cell was committed, -1 for givens
    commit_conf: torch.Tensor  # (B,16) prob of the committed value at commit time, nan for givens
    n_forward: int


@torch.no_grad()
def sample(model, puzzles: torch.Tensor, steps: int, strategy: str = "confidence",
           temperature: float = 0.0, generator: torch.Generator | None = None) -> SampleOut:
    """puzzles: (B,16) digits with 0 = hidden. Commits ceil(remaining / steps_left) cells per step."""
    device = puzzles.device
    grid = puzzles.clone()
    masked = puzzles == 0
    commit_step = torch.full_like(puzzles, -1)
    commit_conf = torch.full(puzzles.shape, float("nan"), device=device)
    n_fwd = 0
    for s in range(steps):
        remaining = masked.sum(1)
        if remaining.max() == 0:
            break
        k = torch.ceil(remaining.float() / (steps - s)).long()
        logits = model(digits_to_tokens(grid)).float()
        n_fwd += 1
        probs = logits.softmax(-1)
        if temperature > 0:
            flat = (logits / temperature).softmax(-1).reshape(-1, logits.shape[-1])
            chosen = torch.multinomial(flat, 1, generator=generator).view(logits.shape[:2])
        else:
            chosen = probs.argmax(-1)
        scores = score_positions(probs, chosen, strategy, generator)
        commit = select_topk(scores, masked, k)
        conf = probs.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
        grid = torch.where(commit, chosen + 1, grid)
        commit_step = torch.where(commit, torch.full_like(commit_step, s), commit_step)
        commit_conf = torch.where(commit, conf, commit_conf)
        masked = masked & ~commit
    return SampleOut(grid, commit_step, commit_conf, n_fwd)

