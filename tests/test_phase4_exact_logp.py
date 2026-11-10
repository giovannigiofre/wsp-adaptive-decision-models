"""exact_trajectory_logp against brute-force enumeration of the sampler on a tiny board (6 cells, 4 digits)."""
import itertools
from collections import defaultdict

import torch

from phase2.data import digits_to_tokens
from phase2.sampler import sample
from phase3.rl import exact_trajectory_logp, trajectory_cell_logp

L, V = 6, 4
PUZZLE = torch.tensor([[2, 0, 0, 3, 0, 0]])
HID = [1, 2, 4, 5]


class Tiny(torch.nn.Module):
    """Any state-dependent model will do: logits from an MLP on the one-hot input tokens."""
    def __init__(self, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.a = torch.nn.Parameter(torch.randn(L * 8, 32, generator=g, dtype=torch.float64) * 0.7)
        self.b = torch.nn.Parameter(torch.randn(32, L * V, generator=g, dtype=torch.float64) * 0.7)

    def forward(self, tokens):
        x = torch.nn.functional.one_hot(tokens, 8).to(torch.float64).reshape(len(tokens), -1)
        return (torch.tanh(x @ self.a) @ self.b).view(len(tokens), L, V)


def enumerate_trajectories(model):
    """{(commit_step tuple, grid tuple): probability} for the 2-step confidence sampler at temperature 1,
    written from the description of the sampler, not from its code."""
    out = defaultdict(lambda: torch.zeros((), dtype=torch.float64))
    p1 = model(digits_to_tokens(PUZZLE))[0].softmax(-1)
    for draw in itertools.product(range(V), repeat=4):
        score = torch.stack([p1[c, d] for c, d in zip(HID, draw)])
        top = score.argsort(descending=True)[:2].tolist()              # 2 of 4 cells are committed in step 0
        grid, cs = PUZZLE[0].clone(), torch.full((L,), -1)
        for t in top:
            grid[HID[t]], cs[HID[t]] = draw[t] + 1, 0
        rest = [HID[t] for t in range(4) if t not in top]
        p2 = model(digits_to_tokens(grid[None]))[0].softmax(-1)
        for d2 in itertools.product(range(V), repeat=2):
            g2, c2 = grid.clone(), cs.clone()
            for c, d in zip(rest, d2):
                g2[c], c2[c] = d + 1, 1
            out[(tuple(c2.tolist()), tuple(g2.tolist()))] += score.prod() * p2[rest[0], d2[0]] * p2[rest[1], d2[1]]
    return out


def as_batch(keys):
    cs = torch.tensor([k[0] for k in keys]); grid = torch.tensor([k[1] for k in keys])
    return PUZZLE.expand(len(keys), -1), grid, cs


def test_exact_logp_equals_enumeration_and_sums_to_one():
    model = Tiny()
    with torch.no_grad():
        ref = enumerate_trajectories(model)
        keys = list(ref)
        lp = exact_trajectory_logp(model, *as_batch(keys), dtype=torch.float64)
        naive = (trajectory_cell_logp(model, *as_batch(keys)) * (PUZZLE == 0)).sum(1)
    want = torch.stack([ref[k] for k in keys])
    assert abs(want.sum().item() - 1) < 1e-9
    assert torch.allclose(lp.exp(), want, rtol=1e-8, atol=1e-12)
    assert abs(naive.exp().sum().item() - 1) > 0.05          # the path likelihood without the order factor is not normalised


def test_score_function_gradient_equals_finite_difference_of_expected_reward():
    model = Tiny(1)

    with torch.no_grad():
        ref = enumerate_trajectories(model)
        keys = list(ref)
    by_grid = defaultdict(float)
    for (cs, g), p in ref.items():
        by_grid[g] += p.item()
    target = max(by_grid, key=by_grid.get)                      # reward 1 for the most likely final grid
    assert 0.02 < by_grid[target] < 0.98
    TARGET = torch.tensor(target)

    def J():
        with torch.no_grad():
            return sum(p for (cs, g), p in enumerate_trajectories(model).items() if g == target)

    P, G, CS = as_batch(keys)
    lp = exact_trajectory_logp(model, P, G, CS, dtype=torch.float64)
    r = (G == TARGET).all(1).double()
    (lp.exp().detach() * r * lp).sum().backward()               # sum over trajectories of P * r * grad log P
    torch.manual_seed(0)
    for prm in (model.a, model.b):
        d = torch.randn_like(prm)
        analytic = (prm.grad * d).sum().item()
        eps = 1e-6
        with torch.no_grad():
            prm += eps * d; up = J(); prm -= 2 * eps * d; down = J(); prm += eps * d
        fd = (up - down).item() / (2 * eps)
        assert abs(analytic) > 1e-3 and abs(analytic - fd) < 1e-5 * abs(analytic), (analytic, fd)


def test_sampler_frequencies_match_exact_probabilities():
    model = Tiny(2)
    with torch.no_grad():
        ref = enumerate_trajectories(model)
        n = 40000
        o = sample(model, PUZZLE.expand(n, -1), 2, "confidence", 1.0, generator=torch.Generator().manual_seed(0))
    top = sorted(ref, key=lambda k: -ref[k].item())[:5]
    for cs, g in top:
        freq = ((o.commit_step == torch.tensor(cs)).all(1) & (o.grid == torch.tensor(g)).all(1)).float().mean().item()
        p = ref[(cs, g)].item()
        assert abs(freq - p) < 5 * (p * (1 - p) / n) ** 0.5 + 1e-4, (freq, p)
