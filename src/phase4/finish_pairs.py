"""Step 4.9c: paired differences on the hard test band for the arms of 4.9c (and the self4 arm of 4.9b).
Seeds pooled per puzzle, 95% bootstrap over puzzles. Each arm against its own start model; arms against self4.

    PYTHONPATH=src python -m phase4.finish_pairs        # results/phase4/4.9c/pairs.csv, val_jumps.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .sampler_eval import paired_ci

R = Path("results/phase4")


def main():
    t, b = pd.read_csv(R / "4.9c" / "test_solved.csv"), pd.read_csv(R / "4.9b" / "test_solved.csv")
    hl = pd.read_csv(R / "4.8d" / "hindsight_long_solved_test.csv")
    assert (t.puzzle == b.puzzle).all() and (t.puzzle == hl.puzzle).all() and (t["start|4"] == b["start|4"]).all()
    h = (t.band == "hard").to_numpy()
    pool = lambda d, pat, S: np.mean([d[pat.format(s=s) + f"|{S}"].to_numpy()[h] for s in range(3)], 0)  # noqa: E731
    rows = []
    for S in (4, 16):
        arms = {"self4 (4.9b, lr 1e-5)": pool(b, "self_diffusion_seed{s}", S), "lr1e-4": pool(t, "self_diffusion_seed{s}_lr1e-4", S),
                "lr3e-4": pool(t, "self_diffusion_seed{s}_lr3e-4", S), "hind": pool(t, "self_diffusion_seed{s}_hind", S)}
        start = {"givens_long": t[f"start|{S}"].to_numpy()[h], "hindsight_long": hl[f"valid_{S}"].to_numpy()[h]}
        pairs = [(a, "givens_long") for a in list(arms)[:3]] + [("hind", "hindsight_long"), ("lr1e-4", "self4 (4.9b, lr 1e-5)"),
                 ("lr3e-4", "self4 (4.9b, lr 1e-5)"), ("lr3e-4", "lr1e-4"), ("hind", "self4 (4.9b, lr 1e-5)")]
        for x, y in pairs:
            vx, vy = arms[x], arms[y] if y in arms else start[y]
            lo, hi = paired_ci(vx - vy)
            rows.append({"a": x, "b": y, "sampler_steps": S, "a_test": vx.mean(), "b_test": vy.mean(), "a_minus_b": (vx - vy).mean(),
                         "lo": lo, "hi": hi})
    df = pd.DataFrame(rows).round(4)
    df.to_csv(R / "4.9c" / "pairs.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))
    # val curve: evaluations (every 25 steps) where the 4-step val rate moved by more than 0.03
    c = pd.concat([pd.read_csv(R / e / "curves.csv").assign(exp=e) for e in ("4.9b", "4.9c")])
    c = c[c.val_solve.notna()].sort_values(["exp", "run", "step"])
    c["jump"] = c.groupby(["exp", "run"]).val_solve.diff().abs() > 0.03
    j = c.groupby(["exp", "run"]).agg(jumps=("jump", "sum"), evals=("jump", "size"), val_min=("val_solve", "min"),
                                      val_max=("val_solve", "max"), val_end=("val_solve", "last")).round(3).reset_index()
    j.to_csv(R / "4.9c" / "val_jumps.csv", index=False)
    print(j.to_string(index=False))


if __name__ == "__main__":
    main()
