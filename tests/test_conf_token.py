import numpy as np
import torch

from phase2.model import TinyDLLM
from phase2.sudoku import enumerate_solutions
from phase3.conf_token import ConfHead, brier_reward, expected_brier, fit_platt
from phase3.order_policy import denoise


def _setup():
    torch.manual_seed(0)
    base = TinyDLLM(n_layers=2, d_model=32, n_heads=2, dropout=0.0, pos_embedding="structured").eval()
    sols = torch.tensor(enumerate_solutions()[:5])
    puzzles = sols.clone()
    puzzles[:, [1, 4, 6, 11, 13]] = 0
    return base, ConfHead(base, 4), puzzles


def test_grid_positions_do_not_see_the_confidence_query():
    base, conf, puzzles = _setup()
    with torch.no_grad():
        z = conf.hidden(puzzles)
        _, ref = denoise(base, puzzles)
        assert z.shape == (5, 17, 32) and torch.allclose(z[:, :16], ref, atol=1e-5)
        conf.query.add_(3.0)                       # changing the query must not move the grid states
        assert torch.allclose(conf.hidden(puzzles)[:, :16], ref, atol=1e-5)


def test_starts_uniform_and_reads_the_grid():
    base, conf, puzzles = _setup()
    with torch.no_grad():
        assert torch.allclose(conf(puzzles).softmax(-1), torch.full((5, 4), 0.25))
        torch.nn.init.normal_(conf.head.weight)
        a, b = conf(puzzles), conf(torch.tensor(enumerate_solutions()[:5]))
    assert not torch.allclose(a, b)                # the query attends to the grid contents


def test_brier_reward_is_maximised_by_the_true_probability():
    buckets = torch.tensor([0.125, 0.375, 0.625, 0.875])
    for p_true, best in ((0.1, 0), (0.4, 1), (0.6, 2), (0.95, 3)):
        r = (torch.rand(200_000, generator=torch.Generator().manual_seed(1)) < p_true).float()
        expected = torch.stack([brier_reward(r, q).mean() for q in buckets])
        assert expected.argmax().item() == best
    assert brier_reward(torch.tensor(1.0), torch.tensor(1.0)) == 1.0
    assert brier_reward(torch.tensor(0.0), torch.tensor(1.0)) == -1.0


def test_expected_brier_gradient_moves_mass_towards_the_outcome():
    buckets = torch.tensor([0.125, 0.375, 0.625, 0.875])
    logits = torch.zeros(2, 4, requires_grad=True)
    loss = expected_brier(logits, torch.tensor([1.0, 0.0]), buckets)
    assert abs(loss.item() - ((buckets - 1) ** 2).mean().item()) < 1e-6   # symmetric buckets, uniform p
    loss.backward()
    step = -logits.grad                             # gradient descent direction
    assert step[0, 3] > step[0, 0] and step[1, 0] > step[1, 3]


def test_platt_recalibration_reduces_overconfidence():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 4000)
    solved = (rng.uniform(size=4000) < p).astype(float)
    over = 1 / (1 + np.exp(-3 * np.log(p / (1 - p))))    # an over-confident version of p
    f, (a, b) = fit_platt(over, solved)
    assert abs(a - 1 / 3) < 0.05 and abs(b) < 0.1
    assert ((f(over) - solved) ** 2).mean() < ((over - solved) ** 2).mean()
    g, (a1, b1) = fit_platt(over, solved, slope_only=True)
    assert b1 == 0.0 and abs(a1 - 1 / 3) < 0.05
