import math

import torch

from src.phase1.llada_generate import (generate_instrumented, get_num_transfer_tokens, select_transfer,
                                       strategy_scores)


def _logits():
    # 4 positions, vocab 3. probs designed so each strategy prefers a different position.
    probs = torch.tensor([[[0.50, 0.49, 0.01],   # conf .50, margin .01, mid entropy
                           [0.45, 0.10, 0.45],   # conf .45, margin 0
                           [0.60, 0.20, 0.20],   # conf .60 (max), margin .40
                           [0.55, 0.45, 0.00001]]])  # conf .55, low entropy-ish
    return probs.log(), probs


def test_confidence_and_margin_and_entropy():
    logits, probs = _logits()
    x0 = logits.argmax(-1)
    s, conf = strategy_scores(logits, x0, "confidence")
    assert torch.allclose(conf, probs.max(-1).values / probs.sum(-1), atol=1e-4)
    assert s[0].argmax() == 2
    s, _ = strategy_scores(logits, x0, "margin")
    assert s[0].argmax() == 2 and s[0].argmin() == 1
    s, _ = strategy_scores(logits, x0, "entropy")
    p = probs / probs.sum(-1, keepdim=True)
    ent = -(p * p.log()).sum(-1)
    assert s[0].argmax() == ent[0].argmin()
    assert torch.allclose(s, -ent, atol=1e-4)


def test_random_reproducible():
    logits, _ = _logits()
    x0 = logits.argmax(-1)
    g1, g2 = torch.Generator().manual_seed(0), torch.Generator().manual_seed(0)
    a, _ = strategy_scores(logits, x0, "random", generator=g1)
    b, _ = strategy_scores(logits, x0, "random", generator=g2)
    assert torch.equal(a, b)


def test_select_transfer_respects_mask():
    scores = torch.tensor([[0.9, 0.1, 0.8, 0.7]])
    mask = torch.tensor([[False, True, True, True]])
    t = select_transfer(scores, mask, torch.tensor([2]))
    assert t.tolist() == [[False, False, True, True]]


def test_num_transfer_tokens():
    m = torch.ones(1, 32, dtype=torch.bool)
    n = get_num_transfer_tokens(m, 8)
    assert n.sum() == 32 and n.tolist()[0] == [4] * 8
    n = get_num_transfer_tokens(m, 5)
    assert n.sum() == 32


class FakeModel(torch.nn.Module):
    """Returns fixed random logits; vocab 10, mask id 9."""
    device = torch.device("cpu")

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.table = torch.randn(64, 10, generator=g)
        self.table[:, 9] = -1e9

    def forward(self, x, attention_mask=None):
        class O: pass
        o = O()
        o.logits = self.table[: x.shape[1]].unsqueeze(0).expand(x.shape[0], -1, -1)
        return o


def test_generate_instrumented_commit_bookkeeping():
    model = FakeModel()
    prompt = torch.tensor([[1, 2, 3]])
    for strat in ["random", "confidence", "margin", "entropy"]:
        out = generate_instrumented(model, prompt, steps=8, gen_length=16, block_length=8,
                                    remasking=strat, mask_id=9, record_frames=True, seed=1)
        cs = out["commit_step"][0]
        assert (cs >= 0).all() and cs.max() == 7
        # semi-AR: block 1 positions committed at steps >= 4, block 0 at steps < 4
        assert (cs[:8] < 4).all() and (cs[8:] >= 4).all()
        # 2 tokens per step
        assert torch.bincount(cs).tolist() == [2] * 8
        assert not torch.isnan(out["commit_conf"]).any()
        assert out["nfe"] == 8 and len(out["frames"]) == 8
        assert (out["x"][0, 3:] != 9).all()
    # confidence strategy commits highest-prob token first within block 0
    out = generate_instrumented(model, prompt, steps=8, gen_length=16, block_length=8, remasking="confidence", mask_id=9)
    p = torch.softmax(model.table[3:11], -1).max(-1).values
    first = (out["commit_step"][0, :8] == 0).nonzero().flatten()
    assert set(first.tolist()) == set(torch.topk(p, 2).indices.tolist())
