"""4x4 Sudoku: enumeration, backtracking solver, uniqueness check, validity.

Grids are length-16 sequences in row-major order. Digits are 1..4, 0 means empty.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

N = 4
CELLS = 16


def _groups() -> list[list[int]]:
    rows = [[r * 4 + c for c in range(4)] for r in range(4)]
    cols = [[r * 4 + c for r in range(4)] for c in range(4)]
    boxes = [[(br + dr) * 4 + bc + dc for dr in range(2) for dc in range(2)]
             for br in (0, 2) for bc in (0, 2)]
    return rows + cols + boxes


GROUPS = _groups()  # 12 groups (4 rows, 4 cols, 4 boxes) of 4 cell indices
PEERS = [sorted({j for g in GROUPS if i in g for j in g} - {i}) for i in range(CELLS)]


def solve(grid, limit: int | None = None) -> list[tuple[int, ...]]:
    """All completions of `grid` (0 = empty), by backtracking. Stops after `limit` solutions."""
    g = list(grid)
    out: list[tuple[int, ...]] = []
    for i, v in enumerate(g):  # reject givens that already clash
        if v and any(g[j] == v for j in PEERS[i]):
            return out

    def bt() -> bool:
        try:
            i = g.index(0)
        except ValueError:
            out.append(tuple(g))
            return limit is not None and len(out) >= limit
        used = {g[j] for j in PEERS[i]}
        for v in range(1, 5):
            if v not in used:
                g[i] = v
                if bt():
                    return True
                g[i] = 0
        return False

    bt()
    return out


@lru_cache(maxsize=1)
def enumerate_solutions() -> tuple[tuple[int, ...], ...]:
    """All valid solved 4x4 grids (there are 288), in lexicographic order."""
    return tuple(solve([0] * CELLS))


def solutions_array() -> np.ndarray:
    return np.array(enumerate_solutions(), dtype=np.int64)


def is_valid_solution(grid) -> bool:
    if len(grid) != CELLS or any(v not in (1, 2, 3, 4) for v in grid):
        return False
    return all(len({grid[i] for i in g}) == 4 for g in GROUPS)


def validity(grid, puzzle) -> bool:
    """Full grid valid and consistent with the givens (nonzero cells) of `puzzle`."""
    return is_valid_solution(grid) and all(p == 0 or p == v for p, v in zip(puzzle, grid))


def count_completions(puzzle, sols: np.ndarray | None = None) -> int:
    """Number of valid completions, by matching against the enumerated 288 solutions.

    Exact (every completion is one of the 288) and much faster than backtracking.
    """
    sols = solutions_array() if sols is None else sols
    p = np.asarray(puzzle)
    given = p != 0
    return int(np.all(sols[:, given] == p[given], axis=1).sum())


def is_unique(puzzle, sols: np.ndarray | None = None) -> bool:
    return count_completions(puzzle, sols) == 1


def to_str(grid) -> str:
    rows = []
    for r in range(4):
        cells = ["." if v == 0 else str(v) for v in grid[r * 4:(r + 1) * 4]]
        rows.append(" ".join(cells[:2]) + " | " + " ".join(cells[2:]))
        if r == 1:
            rows.append("----+----")
    return "\n".join(rows)
