import torch

from phase2.data import batch_validity
from phase4.detector_gate import degraded, fix
from test_phase4_eraser import Knows, solved_grid


class WrongFirst(Knows):                           # first pass writes a wrong digit in cell 0, later passes are right
    calls = 0

    def forward(self, tokens):
        out = super().forward(tokens)
        if self.calls == 0:
            out[:, 0] = out[:, 0].roll(1, -1)
        self.calls += 1
        return out


def test_degraded_oracle_limits():
    sol = solved_grid()
    grid = sol.clone(); grid[0, :5] = grid[0, :5] % 9 + 1
    gen = torch.Generator().manual_seed(0)
    assert (degraded(sol, 1.0, 0.0, gen)(grid) == (grid != sol)).all()      # perfect detector
    assert not degraded(sol, 0.0, 0.0, gen)(grid).any()                     # blind detector
    assert degraded(sol, 1.0, 1.0, gen)(grid).all()                         # flags everything


def test_fix_repairs_with_full_recall_and_not_with_none():
    sol = solved_grid()
    puzzle = sol.clone(); puzzle[0, ::2] = 0
    gen = torch.Generator().manual_seed(0)
    assert batch_validity(fix(WrongFirst(sol), puzzle, 1, 1, degraded(sol, 1.0, 0.0, gen)), puzzle).all()
    assert not batch_validity(fix(WrongFirst(sol), puzzle, 1, 1, degraded(sol, 0.0, 0.0, gen)), puzzle).all()
