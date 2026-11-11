import random

from phase4.sudoku9 import (CELLS, canonical_digits, count_solutions, difficulty, is_unique, is_valid_solution,
                            logic_solve, make_puzzle, one_pass_fill, random_solution, solve, validity)


def g(s):
    return [int(ch) for ch in s.replace(".", "0")]


# Published puzzles. EASY and its solution: P. Norvig, "Solving Every Sudoku Puzzle" (grid1).
EASY = g("003020600900305001001806400008102900700000008006708200002609500800203009005010300")
EASY_SOL = g("483921657967345821251876493548132976729564138136798245372689514814253769695417382")
# A 17-given puzzle (first entry of G. Royle's list of 17-clue puzzles) and A. Inkala's 2012 puzzle.
SEVENTEEN = g("000000010400000000020000000000050407008000300001090000300400200050100000000806000")
INKALA = g("800000000003600000070090200050007000000045700000100030001000068008500010090000400")


def test_known_solution():
    assert is_valid_solution(EASY_SOL) and validity(EASY_SOL, EASY)
    assert solve(EASY, 2) == [tuple(EASY_SOL)]
    assert difficulty(EASY) <= 1 and logic_solve(EASY)[0] == EASY_SOL


def test_hard_published_puzzles_have_one_solution():
    for p in (SEVENTEEN, INKALA):
        assert sum(v != 0 for v in SEVENTEEN) == 17
        sols = solve(p, 2)
        assert len(sols) == 1 and validity(sols[0], p)
    assert difficulty(INKALA) == 2


def test_counts_two_solutions():
    # Remove a "deadly rectangle": two rows, two columns in one band of boxes, holding the same two digits.
    sol = EASY_SOL
    found = None
    for r1 in range(9):
        for r2 in range(r1 + 1, 9):
            if r1 // 3 != r2 // 3:
                continue
            for c1 in range(9):
                for c2 in range(c1 + 1, 9):
                    if sol[r1 * 9 + c1] == sol[r2 * 9 + c2] and sol[r1 * 9 + c2] == sol[r2 * 9 + c1]:
                        found = [r1 * 9 + c1, r1 * 9 + c2, r2 * 9 + c1, r2 * 9 + c2]
    assert found, "no rectangle in this grid"
    p = list(sol)
    for i in found:
        p[i] = 0
    assert count_solutions(p, 5) == 2 and not is_unique(p)


def test_invalid_inputs():
    bad = list(EASY_SOL)
    bad[0] = bad[1]
    assert not is_valid_solution(bad)
    assert not validity(EASY_SOL, [9 if i == 0 else 0 for i in range(CELLS)])  # given 9 where solution has 4
    clash = [0] * CELLS
    clash[0] = clash[1] = 5
    assert solve(clash) == []


def test_random_solution_and_puzzle():
    rng = random.Random(0)
    for k in (30, 45, 58):
        sol = random_solution(rng)
        assert is_valid_solution(sol)
        p = make_puzzle(sol, k, rng)
        hidden = sum(v == 0 for v in p)
        assert 24 <= hidden <= k and all(a == 0 or a == b for a, b in zip(p, sol))
        assert solve(p, 2) == [tuple(sol)]


def test_one_pass_and_logic():
    p = list(EASY_SOL)
    p[0] = p[40] = p[80] = 0
    assert one_pass_fill(p) == EASY_SOL and difficulty(p) == 0
    assert 0 in one_pass_fill(SEVENTEEN)


def test_canonical_digits():
    perm = [3, 1, 4, 5, 9, 2, 6, 8, 7]
    relabelled = [perm[v - 1] for v in EASY_SOL]
    assert canonical_digits(EASY_SOL) == canonical_digits(relabelled)
    assert canonical_digits(EASY, EASY_SOL)[:3] == (0, 0, 3)


def test_generator_is_deterministic_and_clean():
    import numpy as np
    from phase4.analysis import np_valid
    from phase4.generate import build

    cfg = dict(data=dict(seed=0, bands=dict(easy=[30, 40], hard=[51, 56]), per_band=dict(train=30, val=5, test=5),
                         max_tries_per_solution=20, chunk=10, workers=1))
    a, stats = build(cfg)
    b, _ = build(cfg, workers=2)
    seen = set()
    for split in ("train", "val", "test"):
        assert all((a[split][k] == b[split][k]).all() for k in a[split])
        d = a[split]
        assert np_valid(d["solution"]).all()
        assert (((d["puzzle"] == 0).sum(axis=1)) == d["k"]).all()
        assert ((d["puzzle"] == 0) | (d["puzzle"] == d["solution"])).all()
        for p, s in zip(d["puzzle"], d["solution"]):
            assert solve(p.tolist(), 2) == [tuple(s.tolist())]
            assert s.tobytes() not in seen
            seen.add(s.tobytes())
    assert stats["sizes"] == dict(train=60, val=10, test=10)


def test_torch_validity_and_model_shapes_9x9():
    import torch
    from phase2.data import batch_validity
    from phase2.model import TinyDLLM

    sol = torch.tensor([EASY_SOL, EASY_SOL, EASY_SOL])
    puz = torch.tensor([EASY, EASY, EASY])
    sol[1, 0], sol[1, 1] = sol[1, 1].item(), sol[1, 0].item()   # breaks columns and the givens stay fine or not
    sol[2, 2] = 9                                               # given 3 overwritten, row broken
    assert batch_validity(sol, puz).tolist() == [True, False, False]
    m = TinyDLLM(2, 32, 2, 0.0, pos_embedding="structured", grid_n=9)
    assert m(torch.ones(2, 81, dtype=torch.long)).shape == (2, 81, 9)
    assert m.rcb[2].view(9, 9)[4, 7].item() == 5 and m.rcb[2].view(9, 9)[8, 0].item() == 6
