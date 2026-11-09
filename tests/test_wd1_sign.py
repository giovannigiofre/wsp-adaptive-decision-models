import torch

from phase2.model import TinyDLLM
from phase2.rl_wd1 import group_weights, one_step_loglik, sequential_loglik, wd1_loss
from phase2.sudoku import enumerate_solutions


def _group():
    sol = torch.tensor(enumerate_solutions()[5])
    puzzle = sol.clone()
    puzzle[[0, 3, 6, 9, 12, 15]] = 0
    wrong = sol.clone()
    wrong[[0, 3]] = sol[[3, 0]] if sol[0] != sol[3] else (sol[[0, 3]] % 4) + 1
    assert not torch.equal(wrong, sol)
    return puzzle.repeat(2, 1), torch.stack([sol, wrong]), torch.tensor([[1.0, 0.0]])


def _one_step(positive_only: bool):
    torch.manual_seed(0)
    model = TinyDLLM(n_layers=2, d_model=32, n_heads=2, dropout=0.0)
    P, O, r = _group()
    _, wp, wn, keep = group_weights(r, tau=1.0, positive_only=positive_only)
    assert keep.all()
    before = one_step_loglik(model, P, O).detach()
    loss = wd1_loss(one_step_loglik(model, P, O, p_mask=0.0).view(1, 2), wp, wn)
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            p -= 1e-3 * p.grad
    after = one_step_loglik(model, P, O).detach()
    return before, after


def test_wd1_step_raises_correct_and_lowers_wrong():
    before, after = _one_step(positive_only=False)
    assert after[0] > before[0], "log p(correct) must increase"
    assert after[1] < before[1], "log p(wrong) must decrease"


def test_positive_only_raises_correct():
    before, after = _one_step(positive_only=True)
    assert after[0] > before[0]


def test_weights_and_skipping():
    r = torch.tensor([[1., 0., 0., 1.], [1., 1., 1., 1.], [0., 0., 0., 0.]])
    A, wp, wn, keep = group_weights(r, tau=0.5)
    assert keep.tolist() == [True, False, False]
    assert torch.allclose(wp.sum(1), torch.ones(3)) and torch.allclose(wn.sum(1), torch.ones(3))
    assert wp[0, 0] > wp[0, 1] and wn[0, 0] < wn[0, 1]
    assert torch.allclose(A[1:], torch.zeros(2, 4))
    _, wp2, wn2, _ = group_weights(r, tau=0.5, positive_only=True)
    assert (wn2 == 0).all() and torch.allclose(wp2, wp)


def test_sequential_equals_one_step_when_one_cell_hidden():
    torch.manual_seed(1)
    model = TinyDLLM(n_layers=1, d_model=32, n_heads=2, dropout=0.0).eval()
    sol = torch.tensor(enumerate_solutions()[9]).unsqueeze(0)
    puzzle = sol.clone()
    puzzle[0, 7] = 0
    with torch.no_grad():
        assert torch.allclose(one_step_loglik(model, puzzle, sol), sequential_loglik(model, puzzle, sol))
