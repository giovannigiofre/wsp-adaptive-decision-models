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


def test_wd1_batch_equals_wd1_for_one_group_and_uses_equal_groups():
    from phase3.objectives import wd1, wd1_batch
    r = torch.tensor([[1., 0, 0, 1]])
    a, b = wd1(r, floor=-5.0), wd1_batch(r, floor=-5.0)
    assert torch.allclose(a.pos, b.pos) and torch.allclose(a.neg, b.neg) and b.floor == -5.0
    r = torch.tensor([[1., 1, 1, 1], [0., 0, 0, 0], [1., 0, 1, 0]])
    g = wd1_batch(r)
    assert g.keep.all() and torch.allclose(g.pos.sum(), torch.tensor(3.0)) and torch.allclose(g.neg.sum(), torch.tensor(3.0))
    assert (g.coef[0] < 0).all() and (g.coef[1] > 0).all()      # all-valid group pulled up, all-invalid pushed down
    assert torch.allclose(g.pos[0, 0], g.pos[2, 0]) and torch.allclose(g.neg[1, 0], g.neg[2, 1])


def test_sampler_order_temp_zero_is_the_old_path_and_small_is_the_same_order():
    from phase2.sampler import sample

    class Fixed(torch.nn.Module):
        def forward(self, tokens):
            g = torch.Generator().manual_seed(0)
            return torch.randn(1, tokens.shape[1], 9, generator=g).expand(tokens.shape[0], -1, -1) * 3

    p = torch.zeros(8, 81, dtype=torch.long)
    a = sample(Fixed(), p, 4, "confidence")
    b = sample(Fixed(), p, 4, "confidence", order_temp=1e-4)
    c = sample(Fixed(), p, 4, "confidence", order_temp=5.0)
    assert torch.equal(a.commit_step, b.commit_step) and torch.equal(a.grid, b.grid)
    assert not torch.equal(a.commit_step, c.commit_step) and (c.grid > 0).all()
