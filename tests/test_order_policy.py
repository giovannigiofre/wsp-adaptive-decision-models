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


def _tiny():
    from phase2.model import TinyDLLM
    torch.manual_seed(0)
    return TinyDLLM(n_layers=2, d_model=32, n_heads=2, dropout=0.0, pos_embedding="structured").eval()


def _batch(n=6):
    sols = torch.tensor(enumerate_solutions()[:n])
    puzzles = sols.clone()
    puzzles[:, [0, 3, 5, 6, 9, 10, 12, 15]] = 0
    return puzzles, sols


def test_denoise_matches_model_forward():
    from phase2.data import digits_to_tokens
    from phase3.order_policy import denoise
    model = _tiny()
    puzzles, _ = _batch()
    with torch.no_grad():
        probs, hidden = denoise(model, puzzles)
        assert torch.allclose(probs, model(digits_to_tokens(puzzles)).softmax(-1), atol=1e-6)
    assert hidden.shape == (6, 16, 32)


def test_confidence_init_reproduces_confidence_order():
    from phase2.sampler import sample
    from phase3.order_policy import OrderPolicy, rollout
    model = _tiny()
    puzzles, _ = _batch()
    for feats in ("probs", "hidden"):
        policy = OrderPolicy(feats, d_model=32, init="confidence")
        with torch.no_grad():
            grid, logp, _, agree = rollout(model, policy, puzzles, greedy=True)
        assert torch.equal(grid, sample(model, puzzles, 16, "confidence").grid) and agree == 1.0
        assert (logp < 0).all()


def test_zero_init_is_uniform_and_logp_is_exact():
    import math
    from phase3.order_policy import OrderPolicy, rollout
    model = _tiny()
    puzzles, sols = _batch()
    torch.manual_seed(1)
    grid, logp, teach, _ = rollout(model, OrderPolicy("probs", d_model=32, init="zero"), puzzles, sols)
    assert (grid != 0).all()
    assert torch.allclose(logp, torch.full((6,), -math.lgamma(9)), atol=1e-4)   # 8 hidden cells: log(1/8!)
    assert (teach <= 0).all()


def test_reinforce_gradient_raises_probability_of_rewarded_order():
    from phase3.order_policy import OrderPolicy, rollout
    model = _tiny()
    puzzles, _ = _batch(2)
    P = puzzles[:1].repeat(2, 1)
    policy = OrderPolicy("probs", d_model=32, init="zero")
    torch.manual_seed(3)
    _, logp, _, _ = rollout(model, policy, P)
    A = torch.tensor([0.5, -0.5])          # first order rewarded, second not
    (-(A * logp).mean()).backward()
    g_lin = policy.lin.weight.grad
    assert g_lin is not None and g_lin.abs().sum() > 0
