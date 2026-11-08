import torch

from phase2.sampler import STRATEGIES, sample, score_positions, select_topk


class Fixed(torch.nn.Module):
    """Fixed logits regardless of input: position i prefers digit (i % 4) + 1 with logit 0.1 * i."""

    def forward(self, tokens):
        logits = torch.zeros(tokens.shape[0], 16, 4)
        for i in range(16):
            logits[:, i, i % 4] = 0.1 * i
        return logits


def test_scores_definitions():
    probs = torch.tensor([[[0.7, 0.1, 0.1, 0.1], [0.4, 0.4, 0.1, 0.1]]])
    chosen = probs.argmax(-1)
    assert torch.allclose(score_positions(probs, chosen, "confidence"), torch.tensor([[0.7, 0.4]]))
    assert torch.allclose(score_positions(probs, chosen, "margin"), torch.tensor([[0.6, 0.0]]))
    ent = score_positions(probs, chosen, "entropy")
    assert ent[0, 0] > ent[0, 1]  # peaked distribution has lower entropy -> higher score


def test_select_topk_per_row_k_and_only_masked():
    scores = torch.tensor([[5., 4., 3., 2.], [1., 2., 3., 4.]])
    masked = torch.tensor([[False, True, True, True], [True, True, True, False]])
    sel = select_topk(scores, masked, torch.tensor([2, 1]))
    assert sel.tolist() == [[False, True, True, False], [False, False, True, False]]


def test_budgets_commit_counts_and_order():
    puzzles = torch.zeros(3, 16, dtype=torch.long)
    puzzles[:, :4] = torch.tensor([1, 2, 3, 4])  # 12 hidden cells
    for S in (1, 2, 4, 8, 16):
        for strat in STRATEGIES:
            out = sample(Fixed(), puzzles, S, strat)
            assert (out.grid[:, :4] == puzzles[:, :4]).all() and (out.grid > 0).all()
            assert (out.commit_step[:, :4] == -1).all()
            assert out.commit_step[:, 4:].max().item() == min(S, 12) - 1
            counts = torch.bincount(out.commit_step[0, 4:], minlength=min(S, 12))
            assert counts.sum() == 12 and counts.max() - counts.min() <= 1
    out = sample(Fixed(), puzzles, 16, "confidence")
    # one cell per step, highest-probability (largest index) first
    assert out.commit_step[0, 4:].tolist() == list(range(11, -1, -1))


def test_temperature_sampling_varies():
    torch.manual_seed(0)
    out = sample(Fixed(), torch.zeros(64, 16, dtype=torch.long), 4, "confidence", temperature=1.0)
    assert len({tuple(g) for g in out.grid.tolist()}) > 1
