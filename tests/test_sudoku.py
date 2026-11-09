import numpy as np
import torch

from phase2.data import batch_validity, make_puzzles, split_solutions
from phase2.sudoku import count_completions, enumerate_solutions, is_valid_solution, solve, validity


def test_exactly_288_solutions_all_valid_and_distinct():
    sols = enumerate_solutions()
    assert len(sols) == 288 and len(set(sols)) == 288
    assert all(is_valid_solution(s) for s in sols)


def test_random_grids_mostly_invalid():
    rng = np.random.default_rng(0)
    grids = rng.integers(1, 5, size=(2000, 16))
    assert np.mean([is_valid_solution(g) for g in grids]) < 0.01


def test_givens_respected():
    sol = enumerate_solutions()[17]
    puzzle = list(sol)
    puzzle[0] = puzzle[5] = puzzle[10] = 0
    assert validity(sol, puzzle)
    other = next(s for s in enumerate_solutions() if s[1] != sol[1])
    assert is_valid_solution(other) and not validity(other, puzzle)


def test_uniqueness_count_matches_backtracking():
    rng = np.random.default_rng(1)
    sols = enumerate_solutions()
    for _ in range(200):
        p = np.array(sols[rng.integers(288)])
        p[rng.choice(16, rng.integers(6, 13), replace=False)] = 0
        assert count_completions(p) == len(solve(p))


def test_splits_disjoint_and_puzzles_unique():
    sp = split_solutions(0)
    keys = {n: {tuple(s) for s in v} for n, v in sp.items()}
    assert sum(map(len, keys.values())) == 288
    assert not (keys["train"] & keys["val"] or keys["train"] & keys["test"] or keys["val"] & keys["test"])
    ps = make_puzzles(sp["val"], 1, seed=3)
    for p, s, k in zip(ps.puzzles.numpy(), ps.solutions.numpy(), ps.k.numpy()):
        assert (p == 0).sum() == k and 6 <= k <= 12
        assert tuple(s) in keys["val"] and count_completions(p) == 1


def test_batch_validity_matches_python():
    rng = np.random.default_rng(2)
    sols = np.array(enumerate_solutions())
    grids = np.concatenate([sols[:50], rng.integers(1, 5, size=(50, 16))])
    puzzles = grids.copy()
    puzzles[:, :8] = 0
    puzzles[10, 9] = (puzzles[10, 9] % 4) + 1  # a valid grid that contradicts a given
    got = batch_validity(torch.tensor(grids), torch.tensor(puzzles)).numpy()
    want = np.array([validity(g, p) for g, p in zip(grids, puzzles)])
    assert (got == want).all() and not got[10]
