import torch

from phase2.model import TinyDLLM
from phase2.rl_wd1 import group_weights, one_step_loglik, wd1_loss
from phase2.sudoku import enumerate_solutions
from phase3.objectives import OBJECTIVES, weight_diagnostics, weighted_loglik_loss
from phase3.rl import loglik, one_step_logprobs, predictive_entropy, run_name


def _group():
    """One puzzle, two completions: the solution and a wrong grid. Rewards [1, 0]."""
    sol = torch.tensor(enumerate_solutions()[5])
    puzzle = sol.clone()
    puzzle[[0, 3, 6, 9, 12, 15]] = 0
    wrong = sol.clone()
    wrong[[0, 3]] = sol[[3, 0]] if sol[0] != sol[3] else (sol[[0, 3]] % 4) + 1
    assert not torch.equal(wrong, sol)
    return puzzle.repeat(2, 1), torch.stack([sol, wrong]), torch.tensor([[1.0, 0.0]])


def _phat(model, P, O):
    logp, given = one_step_logprobs(model, P, O)
    return loglik(logp, O, given)


def _one_step(**params):
    torch.manual_seed(0)
    model = TinyDLLM(n_layers=2, d_model=32, n_heads=2, dropout=0.0)
    P, O, r = _group()
    gw = OBJECTIVES["wd1"](r, **params)
    assert gw.keep.all()
    before = _phat(model, P, O).detach()
    weighted_loglik_loss(_phat(model, P, O).view(1, 2), gw.coef).backward()
    with torch.no_grad():
        for p in model.parameters():
            p -= 1e-3 * p.grad
    return before, _phat(model, P, O).detach()


def test_wd1_raises_valid_and_lowers_invalid():
    before, after = _one_step(tau=1.0, beta=1.0)
    assert after[0] > before[0] and after[1] < before[1]


def test_positive_only_raises_valid():
    before, after = _one_step(tau=1.0, beta=0.0)
    assert after[0] > before[0]


def test_wd1_matches_phase2_weights_and_loss():
    torch.manual_seed(0)
    r = (torch.rand(6, 8) < 0.4).float()
    logp = -10 * torch.rand(6, 8)
    for beta, positive_only in [(1.0, False), (0.0, True)]:
        for tau in (0.5, 1.0, 2.0):
            _, wp, wn, keep = group_weights(r, tau, positive_only)
            gw = OBJECTIVES["wd1"](r, tau=tau, beta=beta)
            assert torch.equal(gw.keep, keep)
            assert torch.allclose(gw.coef, wn - wp) and torch.allclose(gw.w_pos, wp)
            assert torch.allclose(weighted_loglik_loss(logp[keep], gw.coef[keep]),
                                  wd1_loss(logp[keep], wp[keep], wn[keep]))


def test_coefficients_sum_to_beta_minus_one():
    r = torch.tensor([[1., 0., 0., 1.]])
    for beta in (0.0, 0.5, 1.0):
        assert torch.allclose(OBJECTIVES["wd1"](r, tau=1.0, beta=beta).coef.sum(1), torch.tensor([beta - 1.0]))


def test_loglik_matches_phase2_with_same_random_masks():
    torch.manual_seed(1)
    model = TinyDLLM(n_layers=1, d_model=32, n_heads=2, dropout=0.0).eval()
    P, O, _ = _group()
    with torch.no_grad():
        g = torch.Generator().manual_seed(7)
        ref = one_step_loglik(model, P, O, p_mask=0.5, generator=g)
        g = torch.Generator().manual_seed(7)
        logp, given = one_step_logprobs(model, P, O, p_mask=0.5, generator=g)
    assert torch.allclose(loglik(logp, O, given), ref)
    assert 0.0 < predictive_entropy(logp, given).item() <= torch.log(torch.tensor(4.0)).item() + 1e-6


def test_weight_diagnostics():
    r = torch.tensor([[1., 1., 0., 0., 0., 0., 0., 0.]])
    valid = r.bool()
    flat = weight_diagnostics(OBJECTIVES["wd1"](r, tau=1.0).w_pos, valid)
    sharp = weight_diagnostics(OBJECTIVES["wd1"](r, tau=0.1).w_pos, valid)
    e = torch.e
    assert abs(flat["w_pos_on_invalid"] - 6 / (6 + 2 * e)) < 1e-5   # 52% at tau = 1
    assert sharp["w_pos_on_invalid"] < 1e-3 and abs(sharp["w_ess"] - 2.0) < 1e-2
    assert flat["w_ess"] > sharp["w_ess"] and flat["w_entropy"] > sharp["w_entropy"]


def test_run_name():
    rc = {"objective": "wd1", "objective_params": {"tau": 1.0, "beta": 0}, "reward": "binary"}
    assert run_name(rc, 2) == "wd1_beta0_tau1_seed2"
    assert run_name({**rc, "reward": "graded"}, 0, "_long") == "wd1_beta0_tau1_graded_seed0_long"
