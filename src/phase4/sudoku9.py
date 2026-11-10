"""9x9 Sudoku: validity, a backtracking solver that counts solutions, random solved grids,
and a singles-only logic solver used as a difficulty measure.

Grids are length-81 sequences in row-major order. Digits are 1..9, 0 means empty.
The 4x4 code of phase 2 enumerates all 288 grids; that is impossible here, so this module is separate.
"""
from __future__ import annotations

import random

N = 9
CELLS = 81
R = [i // 9 for i in range(CELLS)]
C = [i % 9 for i in range(CELLS)]
B = [(i // 27) * 3 + (i % 9) // 3 for i in range(CELLS)]
ROWS = [[r * 9 + c for c in range(9)] for r in range(9)]
COLS = [[r * 9 + c for r in range(9)] for c in range(9)]
BOXES = [[i for i in range(CELLS) if B[i] == b] for b in range(9)]
GROUPS = ROWS + COLS + BOXES  # 27 units of 9 cells
ALL = 0x1FF
POP = [bin(m).count("1") for m in range(512)]
DIGITS = [[d + 1 for d in range(9) if m >> d & 1] for m in range(512)]


def is_valid_solution(grid) -> bool:
    if len(grid) != CELLS or any(v not in range(1, 10) for v in grid):
        return False
    return all(len({grid[i] for i in g}) == 9 for g in GROUPS)


def validity(grid, puzzle) -> bool:
    """Full grid valid and consistent with the givens (nonzero cells) of `puzzle`."""
    return is_valid_solution(grid) and all(p == 0 or p == v for p, v in zip(puzzle, grid))


def solve(grid, limit: int | None = 2, rng: random.Random | None = None) -> list[tuple[int, ...]]:
    """Completions of `grid`, by backtracking on the cell with fewest candidates. Stops after `limit`
    solutions (None = all). With `rng`, digits are tried in random order."""
    g = [int(v) for v in grid]
    rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9
    for i, v in enumerate(g):
        if v:
            bit = 1 << (v - 1)
            if (rows[R[i]] | cols[C[i]] | boxes[B[i]]) & bit:
                return []
            rows[R[i]] |= bit; cols[C[i]] |= bit; boxes[B[i]] |= bit
    empties = [i for i in range(CELLS) if g[i] == 0]
    out: list[tuple[int, ...]] = []

    def bt() -> bool:
        best, best_m, best_n = -1, 0, 10
        for i in empties:
            if g[i] == 0:
                m = ALL & ~(rows[R[i]] | cols[C[i]] | boxes[B[i]])
                n = POP[m]
                if n == 0:
                    return False
                if n < best_n:
                    best, best_m, best_n = i, m, n
                    if n == 1:
                        break
        if best < 0:
            out.append(tuple(g))
            return limit is not None and len(out) >= limit
        r, c, b = R[best], C[best], B[best]
        ds = DIGITS[best_m]
        if rng is not None:
            ds = ds[:]
            rng.shuffle(ds)
        for d in ds:
            bit = 1 << (d - 1)
            g[best] = d
            rows[r] |= bit; cols[c] |= bit; boxes[b] |= bit
            if bt():
                return True
            rows[r] ^= bit; cols[c] ^= bit; boxes[b] ^= bit
        g[best] = 0
        return False

    bt()
    return out


def count_solutions(grid, limit: int = 2) -> int:
    return len(solve(grid, limit))


def is_unique(grid) -> bool:
    return count_solutions(grid, 2) == 1


def random_solution(rng: random.Random) -> list[int]:
    """A random solved grid: randomized backtracking, then a random symmetry (digit relabelling, row and
    column permutations inside bands, band and stack permutations, transpose). The backtracking alone is
    not uniform over grids; the symmetry removes any bias towards particular digits or positions."""
    g = list(solve([0] * CELLS, 1, rng)[0])
    perm = list(range(1, 10))
    rng.shuffle(perm)
    g = [perm[v - 1] for v in g]

    def lines() -> list[int]:
        bands = [0, 1, 2]
        rng.shuffle(bands)
        out = []
        for b in bands:
            inner = [0, 1, 2]
            rng.shuffle(inner)
            out += [b * 3 + k for k in inner]
        return out

    rr, cc = lines(), lines()
    g = [g[rr[r] * 9 + cc[c]] for r in range(9) for c in range(9)]
    if rng.random() < 0.5:
        g = [g[c * 9 + r] for r in range(9) for c in range(9)]
    return g


def make_puzzle(solution, k: int, rng: random.Random) -> list[int]:
    """Hide up to k cells of `solution`, in random order, keeping the solution unique. A cell whose
    removal would allow a second solution stays given, so the result can have fewer than k hidden cells."""
    order = list(range(CELLS))
    rng.shuffle(order)
    p = list(solution)
    start = 0
    first = min(k, 24)  # removing few cells almost never breaks uniqueness: try them in one check
    for i in order[:first]:
        p[i] = 0
    if is_unique(p):
        start = first
    else:
        p = list(solution)
    hidden = start
    for i in order[start:]:
        if hidden >= k:
            break
        p[i] = 0
        if is_unique(p):
            hidden += 1
        else:
            p[i] = solution[i]
    return p


def candidates(g) -> list[int]:
    """Bitmask of digits allowed in each empty cell by its row, column and box (0 for filled cells)."""
    rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9
    for i, v in enumerate(g):
        if v:
            bit = 1 << (v - 1)
            rows[R[i]] |= bit; cols[C[i]] |= bit; boxes[B[i]] |= bit
    return [0 if g[i] else ALL & ~(rows[R[i]] | cols[C[i]] | boxes[B[i]]) for i in range(CELLS)]


def one_pass_fill(puzzle) -> list[int]:
    """Baseline: fill every cell that has exactly one candidate in the puzzle as given, once."""
    g = [int(v) for v in puzzle]
    cand = candidates(g)
    for i, m in enumerate(cand):
        if POP[m] == 1:
            g[i] = DIGITS[m][0]
    return g


def logic_solve(puzzle, hidden_singles: bool = True) -> tuple[list[int], bool]:
    """Repeat until stuck: naked singles (a cell with one candidate); if none and `hidden_singles`,
    hidden singles (a digit with one possible cell in a row, column or box). Returns (grid, used_hidden)."""
    g = [int(v) for v in puzzle]
    used_hidden = False
    while True:
        cand = candidates(g)
        naked = [(i, DIGITS[m][0]) for i, m in enumerate(cand) if POP[m] == 1]
        if naked:  # in a puzzle with a solution, each is forced, so filling them together is safe
            for i, d in naked:
                g[i] = d
            continue
        if not hidden_singles:
            break
        found = None
        for unit in GROUPS:
            for d in range(9):
                cells = [i for i in unit if cand[i] >> d & 1]
                if len(cells) == 1:
                    found = (cells[0], d + 1)
                    break
            if found:
                break
        if not found:
            break
        g[found[0]] = found[1]
        used_hidden = True
    return g, used_hidden


def difficulty(puzzle) -> int:
    """0: naked singles alone solve it. 1: naked and hidden singles solve it. 2: neither does."""
    g, _ = logic_solve(puzzle, hidden_singles=False)
    if 0 not in g:
        return 0
    g, _ = logic_solve(puzzle, hidden_singles=True)
    return 1 if 0 not in g else 2


def canonical_digits(grid, reference=None) -> tuple[int, ...]:
    """Relabel digits by order of first appearance in `reference` (default: the grid itself)."""
    ref = grid if reference is None else reference
    m: dict[int, int] = {}
    for v in ref:
        if v and v not in m:
            m[v] = len(m) + 1
    return tuple(m.get(v, 0) if v else 0 for v in grid)


def to_str(grid) -> str:
    rows = []
    for r in range(9):
        cells = ["." if v == 0 else str(int(v)) for v in grid[r * 9:(r + 1) * 9]]
        rows.append(" ".join(cells[:3]) + " | " + " ".join(cells[3:6]) + " | " + " ".join(cells[6:]))
        if r in (2, 5):
            rows.append("------+-------+------")
    return "\n".join(rows)
