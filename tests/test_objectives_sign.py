import torch

from phase2.model import TinyDLLM
from phase2.rl_wd1 import group_weights, one_step_loglik, wd1_loss
from phase2.sudoku import enumerate_solutions
from phase3.objectives import WEIGHTED, GroupWeights, grpo_loss, weight_diagnostics, weighted_loss
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


def _one_step(objective="wd1", **params):
    torch.manual_seed(0)
    model = TinyDLLM(n_layers=2, d_model=32, n_heads=2, dropout=0.0)
    P, O, r = _group()
    gw = WEIGHTED[objective](r, **params)
    assert gw.keep.all()
    before = _phat(model, P, O).detach()
    weighted_loss(_phat(model, P, O).view(1, 2), gw).backward()
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
            gw = WEIGHTED["wd1"](r, tau=tau, beta=beta)
            assert torch.equal(gw.keep, keep)
            assert torch.allclose(gw.coef, wn - wp) and torch.allclose(gw.w_pos, wp)
            kept = GroupWeights(gw.pos[keep], gw.neg[keep], gw.keep[keep])
            assert torch.allclose(weighted_loss(logp[keep], kept), wd1_loss(logp[keep], wp[keep], wn[keep]))


def test_coefficients_sum_to_beta_minus_one():
    r = torch.tensor([[1., 0., 0., 1.]])
    for beta in (0.0, 0.5, 1.0):
        assert torch.allclose(WEIGHTED["wd1"](r, tau=1.0, beta=beta).coef.sum(1), torch.tensor([beta - 1.0]))


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
    flat = weight_diagnostics(WEIGHTED["wd1"](r, tau=1.0).w_pos, valid)
    sharp = weight_diagnostics(WEIGHTED["wd1"](r, tau=0.1).w_pos, valid)
    e = torch.e
    assert abs(flat["w_pos_on_invalid"] - 6 / (6 + 2 * e)) < 1e-5   # 52% at tau = 1
    assert sharp["w_pos_on_invalid"] < 1e-3 and abs(sharp["w_ess"] - 2.0) < 1e-2
    assert flat["w_ess"] > sharp["w_ess"] and flat["w_entropy"] > sharp["w_entropy"]


DEFAULTS = {"wd1": {"tau": 1.0, "beta": 1.0, "floor": None}, "reinforce": {"normalize": False},
            "filtered": {}, "grpo": {"epsilon": 0.5, "kl": 0.04}}


def test_run_name():
    rc = {"objective": "wd1", "objective_params": {"beta": 0}, "objective_defaults": DEFAULTS, "reward": "binary"}
    assert run_name(rc, 2) == "wd1_beta0_tau1_seed2"                      # same names as the 3.0 runs
    assert run_name({**rc, "reward": "graded"}, 0, "_long") == "wd1_beta0_tau1_graded_seed0_long"
    assert run_name({**rc, "objective_params": {"floor": -12}, "ref_mix": 0.2}, 1) == \
        "wd1_beta1_floor-12_tau1_ref0.2_seed1"
    assert run_name({**rc, "objective": "filtered", "objective_params": {}}, 3) == "filtered_seed3"


def test_reinforce_and_filtered_signs():
    for objective, params in (("reinforce", {}), ("reinforce", {"normalize": True}), ("filtered", {})):
        before, after = _one_step(objective, **params)
        assert after[0] > before[0], objective
    before, after = _one_step("reinforce")
    assert after[1] < before[1]


def test_reinforce_and_filtered_weights():
    r = torch.tensor([[1., 1., 0., 0., 0., 0., 0., 0.], [1., 1., 1., 1., 1., 1., 1., 1.]])
    gw = WEIGHTED["reinforce"](r)
    assert gw.keep.tolist() == [True, False] and torch.allclose(gw.coef[0], -(r[0] - 0.25))
    assert torch.allclose(gw.coef.sum(1), torch.zeros(2), atol=1e-6)
    f = WEIGHTED["filtered"](r)
    assert torch.allclose(f.pos[0], torch.tensor([.5, .5, 0, 0, 0, 0, 0, 0])) and (f.neg == 0).all()
    sharp = WEIGHTED["wd1"](r, tau=0.01, beta=0.0)          # filtered is the tau -> 0 limit of positive-only
    assert torch.allclose(sharp.pos[0], f.pos[0], atol=1e-6)


def test_floor_stops_the_negative_push_only():
    r = torch.tensor([[1., 0.]])
    logp = torch.tensor([[-5.0, -20.0]], requires_grad=True)
    gw = WEIGHTED["wd1"](r, tau=1.0, beta=1.0, floor=-12.0)
    weighted_loss(logp, gw).backward()
    free = WEIGHTED["wd1"](r, tau=1.0, beta=1.0)
    # below the floor the completion only keeps its (small) positive weight; above it nothing changes
    assert torch.allclose(logp.grad[0, 1], -gw.pos[0, 1]) and torch.allclose(logp.grad[0, 0], free.coef[0, 0])


def test_grpo_loss_sign_and_clipping():
    torch.manual_seed(0)
    model = TinyDLLM(n_layers=2, d_model=32, n_heads=2, dropout=0.0)
    P, O, r = _group()
    adv = (r - r.mean(1, keepdim=True)).reshape(-1)
    from phase3.rl import cell_logp
    logp, given = one_step_logprobs(model, P, O)
    old = cell_logp(logp, O).detach()
    before = _phat(model, P, O).detach()
    loss, clip = grpo_loss(cell_logp(logp, O), old, adv, ~given, epsilon=0.5)
    assert clip == 0.0 and torch.allclose(loss, torch.zeros(()), atol=1e-6)   # ratio 1, advantages sum to 0
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            p -= 1e-3 * p.grad
    after = _phat(model, P, O).detach()
    assert after[0] > before[0] and after[1] < before[1]
    # a ratio far above 1 + epsilon with a positive advantage is clipped (no gradient through it)
    far = old - 2.0
    _, clip = grpo_loss(old, far, torch.ones(2), ~given, epsilon=0.5)
    assert clip == 1.0
    # the KL term is zero when the policy equals the reference and positive otherwise
    same, _ = grpo_loss(old, old, torch.zeros(2), ~given, 0.5, kl=1.0, ref_cell_logp=old)
    diff, _ = grpo_loss(old, old, torch.zeros(2), ~given, 0.5, kl=1.0, ref_cell_logp=old - 1.0)
    assert same.abs() < 1e-6 and diff > 0


def test_run_name_with_reference_kl():
    rc = {"objective": "wd1", "objective_params": {}, "objective_defaults": DEFAULTS, "reward": "binary",
          "ref_kl": 0.04}
    assert run_name(rc, 0) == "wd1_beta1_tau1_refkl0.04_seed0"
    assert run_name({**rc, "ref_kl": 0.0}, 0) == "wd1_beta1_tau1_seed0"


def test_sequential_cell_logp_matches_phase2_and_one_step_with_one_hidden_cell():
    from phase2.rl_wd1 import sequential_loglik
    from phase3.rl import sequential_cell_logp
    torch.manual_seed(2)
    model = TinyDLLM(n_layers=1, d_model=32, n_heads=2, dropout=0.0).eval()
    P, O, _ = _group()
    with torch.no_grad():
        seq = (sequential_cell_logp(model, P, O) * (P == 0)).sum(1)
        assert torch.allclose(seq, sequential_loglik(model, P, O), atol=1e-5)
    one = P.clone()
    one[0, 7] = 0
    one[0, [0, 3, 6, 9, 12, 15]] = O[0, [0, 3, 6, 9, 12, 15]]
    with torch.no_grad():
        logp, given = one_step_logprobs(model, one[:1], O[:1])
        assert torch.allclose((sequential_cell_logp(model, one[:1], O[:1]) * (~given)).sum(1), loglik(logp, O[:1], given))
    lp = (sequential_cell_logp(model, P, O) * (P == 0)).sum(1)
    assert lp.requires_grad


def test_run_name_sequential():
    rc = {"objective": "wd1", "objective_params": {}, "objective_defaults": DEFAULTS, "reward": "binary",
          "likelihood": "sequential"}
    assert run_name(rc, 1) == "wd1_beta1_tau1_seqlik_seed1"
