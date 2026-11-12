"""RL objectives for the toy Sudoku dLLM: how a group of rewards becomes a loss on log p_hat.

Weighted-likelihood objectives map rewards r (B, G) to `GroupWeights` (pos, neg >= 0). The loss is
    mean over kept groups of sum_i [neg_i * log p_hat(o_i) - pos_i * log p_hat(o_i)]
so `pos` raises a completion's likelihood and `neg` lowers it. A = r - group mean throughout.

  wd1(tau, beta, floor)  pos = softmax(A / tau), neg = beta * softmax(-A / tau).
                         beta = 1 is wd1 as in the upstream code (github.com/xiaohangt/wd1, commit 1a2e254,
                         wd1/trainers/rev_grpo_trainer.py, which uses tau = 1); beta = 0 is its positive-only
                         variant. Same numbers as phase2.rl_wd1 (checked in tests/test_objectives_sign.py).
                         floor (not in the paper): the negative term uses max(log p_hat, floor), so a
                         completion is no longer pushed down once its log p_hat is below the floor.
  wd1_batch(...)         wd1 with weights normalised over the whole batch, see the function (phase 4.5a).
  reinforce(normalize)   pos = relu(A), neg = relu(-A): the plain policy gradient with the group-mean
                         baseline, i.e. linear instead of softmax weights. normalize divides A by the group std.
  filtered()             pos = uniform over the best-reward completions of the group, neg = 0: the
                         tau -> 0 limit of positive-only ("imitate your own best attempts").

Two objectives are not weighted likelihoods and are handled in rl.py:
  grpo(epsilon, kl)      d1-style clipped-ratio objective, see `grpo_loss`.
  self_diffusion, supervised   the SFT masked-diffusion loss on the model's own valid rollouts / on the
                         true solutions (the latter is a reference that uses labels).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


@dataclass
class GroupWeights:
    pos: torch.Tensor    # (B, G) >= 0, weight of -log p_hat (raise likelihood)
    neg: torch.Tensor    # (B, G) >= 0, weight of +log p_hat (lower likelihood)
    keep: torch.Tensor   # (B,) bool, groups whose rewards are not all equal
    floor: float | None = None

    @property
    def coef(self) -> torch.Tensor:
        """Net coefficient of log p_hat in the loss when no floor is used."""
        return self.neg - self.pos

    @property
    def w_pos(self) -> torch.Tensor:
        """Positive weights normalised to sum to 1 per group, for diagnostics."""
        return self.pos / self.pos.sum(1, keepdim=True).clamp_min(1e-12)


def advantages(r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return r - r.mean(1, keepdim=True), r.max(1).values != r.min(1).values


def wd1(r: torch.Tensor, tau: float = 1.0, beta: float = 1.0, floor: float | None = None) -> GroupWeights:
    A, keep = advantages(r)
    return GroupWeights(torch.softmax(A / tau, dim=1), beta * torch.softmax(-A / tau, dim=1), keep, floor)


def wd1_batch(r: torch.Tensor, tau: float = 1.0, beta: float = 1.0, floor: float | None = None) -> GroupWeights:
    """wd1 with the softmax taken over all B*G completions of the batch instead of within each group
    (phase 4.5a). Weights are multiplied by B, so they sum to 1 per group on average as in `wd1`. A group
    whose rewards are all equal now gets a net weight (pull-up if they are above the batch mean,
    push-down if below), so no group is skipped. With B = 1 it equals `wd1`."""
    B = r.shape[0]
    gw = wd1(r.reshape(1, -1), tau, beta, floor)
    return GroupWeights(gw.pos.view_as(r) * B, gw.neg.view_as(r) * B, torch.ones(B, dtype=torch.bool, device=r.device), floor)


def reinforce(r: torch.Tensor, normalize: bool = False) -> GroupWeights:
    A, keep = advantages(r)
    if normalize:
        A = A / (r.std(1, keepdim=True) + 1e-4)
    return GroupWeights(torch.relu(A), torch.relu(-A), keep)


def filtered(r: torch.Tensor) -> GroupWeights:
    _, keep = advantages(r)
    best = (r == r.max(1, keepdim=True).values).float()
    return GroupWeights(best / best.sum(1, keepdim=True), torch.zeros_like(r), keep)


WEIGHTED: dict[str, Callable[..., GroupWeights]] = {"wd1": wd1, "reinforce": reinforce, "filtered": filtered}
KIND = {**{k: "weighted" for k in WEIGHTED}, "grpo": "grpo", "self_diffusion": "diffusion",
        "supervised": "diffusion", "givens": "diffusion"}   # givens (phase 4.9b): control, see rl.py


def weighted_loss(logp: torch.Tensor, gw: GroupWeights) -> torch.Tensor:
    """logp (n_groups, G) and gw restricted to the kept groups."""
    down = logp if gw.floor is None else logp.clamp_min(gw.floor)
    return (gw.neg * down - gw.pos * logp).sum(1).mean()


def grpo_loss(cell_logp: torch.Tensor, old_cell_logp: torch.Tensor, adv: torch.Tensor, cells: torch.Tensor,
              epsilon: float, kl: float = 0.0, ref_cell_logp: torch.Tensor | None = None):
    """d1-style diffu-GRPO loss, adapted from DiffuGRPOTrainer.compute_loss in the d1 code as shipped in
    github.com/xiaohangt/wd1 (commit 1a2e254, wd1/trainers/diffu_grpo_trainer.py; Apache-2.0 / MIT, see
    that repository). Per-token ratio against the pre-update policy under the same prompt masks,
    clipped to [1 - epsilon, 1 + epsilon], advantage = r - group mean (not divided by the std), optional
    per-token KL estimate to a reference policy with weight `kl`.
    cell_logp, old_cell_logp, ref_cell_logp: (N, 16); adv: (N,); cells: (N, 16) bool, the non-given cells.
    Returns (loss, share of cells where the clipped term is the smaller one, i.e. where the cell
    contributes no gradient; upstream logs the opposite comparison as `clip_ratio`)."""
    ratio = torch.exp(cell_logp - old_cell_logp)
    l1 = ratio * adv[:, None]
    l2 = ratio.clamp(1 - epsilon, 1 + epsilon) * adv[:, None]
    per_cell = -torch.min(l1, l2)
    if kl:
        d = ref_cell_logp - cell_logp
        per_cell = per_cell + kl * (torch.exp(d) - d - 1)
    n = cells.sum()
    return (per_cell * cells).sum() / n, (((l2 < l1).float() * cells).sum() / n).item()


def weight_diagnostics(w_pos: torch.Tensor, valid: torch.Tensor) -> dict[str, float]:
    """w_pos (n, G) normalised positive weights, valid (n, G) bool, kept groups only. Means over groups."""
    return {
        "w_entropy": -(w_pos * w_pos.clamp_min(1e-12).log()).sum(1).mean().item(),
        "w_ess": (1.0 / (w_pos ** 2).sum(1)).mean().item(),          # effective sample size, max G
        "w_pos_on_invalid": (w_pos * (~valid)).sum(1).mean().item(),  # share of positive mass on invalid samples
    }
