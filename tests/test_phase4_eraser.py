import numpy as np
import torch

from phase2.data import batch_validity
from phase4.eraser import conflicts, sample_fix, sample_rcr


def solved_grid() -> torch.Tensor:
    r, c = np.divmod(np.arange(81), 9)
    return torch.from_numpy(((3 * r + r // 3 + c) % 9 + 1).reshape(1, 81))


class Knows(torch.nn.Module):
    """Predicts the stored solution at every cell with probability close to 1."""
    def __init__(self, sol):
        super().__init__()
        self.sol = sol

    def forward(self, tokens):
        return torch.nn.functional.one_hot(self.sol - 1, 9).float().expand(len(tokens), -1, -1) * 20.0


def test_conflicts_flags_both_cells_of_a_clash_and_nothing_on_a_valid_grid():
    g = solved_grid()
    assert not conflicts(g).any()
    bad = g.clone(); bad[0, 0] = bad[0, 1]
    f = conflicts(bad)[0]
    assert f[0] and f[1]


def test_rcr_with_a_perfect_model_solves_and_never_remasks():
    sol = solved_grid()
    puzzle = sol.clone(); puzzle[0, ::2] = 0
    grid, events = sample_rcr(Knows(sol), puzzle, 4)
    assert batch_validity(grid, puzzle).all() and events == 0


def test_fix_rounds_repair_what_the_detector_masks():
    sol = solved_grid()
    puzzle = sol.clone(); puzzle[0, ::2] = 0

    class WrongFirst(Knows):                       # first pass writes a wrong digit in cell 0, later passes are right
        calls = 0

        def forward(self, tokens):
            out = super().forward(tokens)
            if self.calls == 0:
                out[:, 0] = out[:, 0].roll(1, -1)
            self.calls += 1
            return out

    for det in ("oracle", "conflict"):
        m = WrongFirst(sol)
        grid, n = sample_fix(m, puzzle, sol, 1, 1, det)
        assert batch_validity(grid, puzzle).all() and n >= 1
    grid, _ = sample_fix(WrongFirst(sol), puzzle, sol, 1, 0, "oracle")
    assert not batch_validity(grid, puzzle).all()
