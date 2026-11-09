import torch

from phase2.sudoku import enumerate_solutions
from phase3.rewards import REWARDS


def _cases():
    sols = enumerate_solutions()
    sol, other = torch.tensor(sols[3]), torch.tensor(sols[40])
    puzzle = sol.clone()
    puzzle[[1, 2, 5, 8, 11, 14]] = 0
    broken = sol.clone()
    broken[1] = sol[1] % 4 + 1          # a hidden cell changed: row/column/box now repeat a digit
    assert not torch.equal(other, sol) and (other != puzzle)[puzzle != 0].any()
    return puzzle, sol, broken, other    # `other` is a valid grid that contradicts the givens


def test_binary_reward():
    puzzle, sol, broken, other = _cases()
    r = REWARDS["binary"](torch.stack([sol, broken, other]), puzzle.repeat(3, 1))
    assert r.tolist() == [1.0, 0.0, 0.0] and r.dtype == torch.float32


def test_graded_reward():
    puzzle, sol, broken, other = _cases()
    r = REWARDS["graded"](torch.stack([sol, broken, other]), puzzle.repeat(3, 1))
    assert r[0] == 1.0 and r[2] == 0.0             # a changed given gives 0 even if the grid is valid
    assert abs(r[1].item() - 9 / 12) < 1e-6        # one wrong cell breaks its row, column and box
    assert ((REWARDS["graded"](torch.stack([sol, broken]), puzzle.repeat(2, 1)) == 1.0)
            == REWARDS["binary"](torch.stack([sol, broken]), puzzle.repeat(2, 1)).bool()).all()
