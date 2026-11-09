import torch

from phase2.data import PuzzleSet
from phase2.sudoku import enumerate_solutions
from phase3.order_policy import oracle_order


class _Fixed(torch.nn.Module):
    """Denoiser stub: always predicts the digits of `guess`, with confidence growing along the grid."""

    def __init__(self, guess: torch.Tensor):
        super().__init__()
        self.guess = guess

    def forward(self, tokens):
        logits = torch.full((tokens.shape[0], 16, 4), -5.0)
        logits.scatter_(2, (self.guess - 1).expand(tokens.shape[0], 16).unsqueeze(-1), 5.0)
        return logits + torch.arange(16).view(1, 16, 1) * 0.01 * torch.nn.functional.one_hot(
            (self.guess - 1), 4).unsqueeze(0)


def _puzzle():
    sol = torch.tensor(enumerate_solutions()[7])
    puzzle = sol.clone()
    puzzle[[0, 5, 10, 15]] = 0
    return PuzzleSet(puzzle.unsqueeze(0), sol.unsqueeze(0), torch.tensor([4])), sol


def test_oracle_solves_when_predictions_are_right():
    ps, sol = _puzzle()
    o = oracle_order(_Fixed(sol), ps)
    assert o["solved"].tolist() == [True] and o["stuck_at"].tolist() == [-1]
    assert o["n_candidates"].tolist() == [4.0, 3.0, 2.0, 1.0]


def test_oracle_reports_stuck_when_no_prediction_is_right():
    ps, sol = _puzzle()
    wrong = sol % 4 + 1
    o = oracle_order(_Fixed(wrong), ps)
    assert o["solved"].tolist() == [False] and o["stuck_at"].tolist() == [0]
