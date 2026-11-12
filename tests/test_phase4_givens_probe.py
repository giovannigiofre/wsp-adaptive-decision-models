import torch

from phase4.givens_probe import excluded, peer_matrix, per_puzzle


def test_excluded_reads_row_column_and_box():
    g = torch.zeros(1, 81, dtype=torch.long)
    g[0, 0] = 5          # row 0, column 0, box 0
    ex = excluded(g, peer_matrix("cpu"))
    assert ex[0, 8, 4] and ex[0, 72, 4] and ex[0, 20, 4]      # same row, same column, same box
    assert not ex[0, 40].any() and not ex[0, 0].any()         # unrelated cell; a cell does not exclude itself
    assert ex[0].sum() == 20                                  # 20 peers, one digit each


def test_per_puzzle_share():
    ok = torch.tensor([[True, False, True, True]])
    tgt = torch.tensor([[True, True, False, False]])
    assert per_puzzle(ok, tgt)[0] == 0.5
