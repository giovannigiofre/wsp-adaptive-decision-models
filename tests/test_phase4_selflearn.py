import torch

from phase2.data import PuzzleSet, digits_to_tokens
from phase4.selflearn import givens_loss, without_solutions


class Spy(torch.nn.Module):
    """Uniform logits plus a learnable offset per cell; remembers the tokens it was given."""
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(81, 9))

    def forward(self, tokens):
        self.seen = tokens
        return self.w.expand(len(tokens), -1, -1)


def puzzles():
    g = torch.Generator().manual_seed(0)
    p = torch.randint(1, 10, (16, 81), generator=g)
    p[:, :40] = 0                                             # cells 0..39 blank in every puzzle
    return p


def test_blanks_stay_masked_and_carry_no_loss():
    p, m = puzzles(), Spy()
    loss = givens_loss(m, p, 0.05, torch.Generator().manual_seed(1))
    mask_token = digits_to_tokens(torch.zeros(1, 1, dtype=torch.long)).item()
    assert (m.seen[:, :40] == mask_token).all()
    assert (m.seen[:, 40:] == mask_token).any() and (m.seen[:, 40:] != mask_token).any()
    loss.backward()
    assert m.w.grad[:40].abs().sum() == 0 and m.w.grad[40:].abs().sum() > 0


def test_loss_is_zero_for_a_model_that_knows_the_givens_and_positive_otherwise():
    p = puzzles()[:1]

    class Knows(torch.nn.Module):
        def forward(self, tokens):
            return torch.nn.functional.one_hot((p - 1).clamp_min(0), 9).float().expand(len(tokens), -1, -1) * 30.0

    gen = torch.Generator().manual_seed(2)
    assert givens_loss(Knows(), p, 0.5, gen).item() < 1e-6
    assert givens_loss(Spy(), p, 0.5, gen).item() > 0.1


def test_train_set_has_no_solutions():
    p = puzzles()
    ps = without_solutions(PuzzleSet(p, torch.randint(1, 10, p.shape), (p == 0).sum(1)))
    assert (ps.solutions == 0).all() and (ps.puzzles == p).all()


def test_mixed_loss_on_givens_rows_equals_givens_loss():
    from phase4.selfloop import mixed_loss, rows
    p, m = puzzles(), Spy()
    torch.nn.init.normal_(m.w)
    a = givens_loss(m, p, 0.05, torch.Generator().manual_seed(3))
    b = mixed_loss(m, *rows(p, None, "givens"), 0.05, torch.Generator().manual_seed(3))
    assert torch.allclose(a, b)


def test_own_rows_keep_givens_visible_and_hindsight_rows_can_mask_them():
    from phase4.selfloop import mixed_loss, rows
    p = puzzles()
    grid = torch.where(p == 0, torch.full_like(p, 5), p)
    mask_token = digits_to_tokens(torch.zeros(1, 1, dtype=torch.long)).item()
    m = Spy()
    mixed_loss(m, *rows(p, grid, "own"), 0.9, torch.Generator().manual_seed(4)).backward()
    assert (m.seen[:, 40:] != mask_token).all() and m.w.grad[40:].abs().sum() == 0 and m.w.grad[:40].abs().sum() > 0
    m = Spy()
    mixed_loss(m, *rows(p, grid, "hindsight"), 0.9, torch.Generator().manual_seed(4)).backward()
    assert (m.seen[:, 40:] == mask_token).any() and m.w.grad[40:].abs().sum() > 0


def test_batch_rows_draws_buffer_rows_only_from_accepted_puzzles():
    from phase4.selfloop import batch_rows
    torch.manual_seed(0)
    p = puzzles()
    found = torch.zeros_like(p)
    has = torch.zeros(len(p), dtype=torch.bool)
    has[:4] = True
    found[:4] = 7
    x0, always, maskable = batch_rows("hindsight", p, found, has, 64, 0.25)
    assert len(x0) == 64 and int(always.any(1).sum()) == 16                 # 16 givens rows, 48 buffer rows
    assert (x0[~always.any(1)] == 7).all()
    assert int(maskable.all(1).sum()) == 24                                 # half of the buffer rows are hindsight
    x0, always, _ = batch_rows("givens", p, found, has, 64, 0.25)
    assert always.any(1).all()
