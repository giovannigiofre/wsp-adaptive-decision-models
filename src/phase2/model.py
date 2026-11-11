"""Tiny bidirectional transformer encoder for Sudoku grids (no causal mask). 4x4 by default; `grid_n=9`
for the 9x9 grids of phase 4 (the order in which parameters are created is unchanged for 4x4)."""
from __future__ import annotations

import torch
from torch import nn



class TinyDLLM(nn.Module):
    """Token + learned position embeddings -> pre-LN encoder -> logits over the 4 digits.

    The output head predicts only digits (class j = digit j + 1); PAD/MASK are never valid outputs,
    so restricting the head is equivalent to masking their logits to -inf.
    """

    def __init__(self, n_layers: int = 4, d_model: int = 128, n_heads: int = 4, dropout: float = 0.1,
                 seq_len: int | None = None, d_ff: int | None = None, pos_embedding: str = "learned",
                 grid_n: int = 4):
        super().__init__()
        n, b = grid_n, int(round(grid_n ** 0.5))
        assert b * b == n
        seq_len = seq_len or n * n
        self.tok = nn.Embedding(n + 2, d_model)  # PAD, MASK, digits 1..n
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.structured = pos_embedding == "structured"
        if self.structured:  # "2D" option of the spec: + learned row, column and 2x2-box embeddings
            idx = torch.arange(n * n)
            self.register_buffer("rcb", torch.stack([idx // n, idx % n, (idx // (n * b)) * b + (idx % n) // b]),
                                 persistent=False)
            self.row, self.col, self.box = (nn.Embedding(n, d_model) for _ in range(3))
        elif pos_embedding != "learned":
            raise ValueError(pos_embedding)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff or 4 * d_model, dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # (B,L) -> (B,L,n)
        h = self.tok(tokens) + self.pos
        if self.structured:
            h = h + self.row(self.rcb[0]) + self.col(self.rcb[1]) + self.box(self.rcb[2])
        return self.head(self.norm(self.enc(h)))


def build_model(cfg: dict) -> TinyDLLM:
    m = cfg["model"]
    return TinyDLLM(m["n_layers"], m["d_model"], m["n_heads"], m["dropout"],
                    pos_embedding=m.get("pos_embedding", "learned"), grid_n=m.get("grid_n", 4))


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
