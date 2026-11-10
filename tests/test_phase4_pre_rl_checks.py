import numpy as np
import torch

from phase2.sampler import select_topk
from phase4.pre_rl_checks import error_structure, gumbel_like, peers9


def test_gumbel_noise_is_finite_with_the_right_moments():
    torch.manual_seed(0)
    g = gumbel_like(torch.zeros(200_000))
    assert torch.isfinite(g).all()
    assert abs(g.mean().item() - 0.5772) < 0.02 and abs(g.std().item() - 1.2825) < 0.02


def test_small_order_temperature_gives_the_deterministic_order():
    torch.manual_seed(0)
    conf = torch.rand(16, 81) * 0.9 + 0.05
    masked = torch.rand(16, 81) < 0.7
    k = torch.ceil(masked.sum(1).float() / 4).long()
    noisy = select_topk(conf.log() / 1e-4 + gumbel_like(conf), masked, k)
    assert (noisy == select_topk(conf, masked, k)).all()


def test_peers():
    P = peers9()
    assert P.sum(1).tolist() == [20] * 81 and not P.diagonal().any()


def test_error_structure_names_a_same_step_conflict():
    r, c = np.divmod(np.arange(81), 9)
    sol = ((3 * r + r // 3 + c) % 9 + 1).reshape(1, 81)        # a valid grid; first row is 1..9
    hid = [0, 1, 27]                                            # cell 0 and the two peers that hold a 2
    puzzle = sol.copy(); puzzle[0, hid] = 0
    grid = sol.copy(); grid[0, 0] = 2                           # cell 0 wrong, equal to cell 1 (row) and cell 27 (column)
    step = np.full((1, 81), -1)

    def kind(steps):
        step[0, hid] = steps
        s, _ = error_structure(grid, step, puzzle, sol, 2)
        assert s["failed"] == 1
        return max(("visible_conflict", "same_step_conflict", "no_conflict"), key=lambda k: s[f"first_step_wrong_cells_{k}"])

    assert kind([0, 0, 1]) == "same_step_conflict"
    assert kind([1, 0, 1]) == "visible_conflict"               # cell 1 was already on the board
    assert kind([0, 1, 1]) == "no_conflict"                    # both peers still hidden when cell 0 was chosen
