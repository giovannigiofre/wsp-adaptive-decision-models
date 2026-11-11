"""Step 4.8b: paired differences between arms on the val puzzles, from `<arm>_solved_val.csv`.
95% bootstrap over puzzles of the paired difference (one seed per arm: the interval covers puzzle sampling only).

    PYTHONPATH=src python -m phase4.selfloop_pairs --exp 4.8b --pairs harvest:givens hindsight:givens hindsight:harvest
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from phase2.evaluate import bootstrap_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="4.8b")
    ap.add_argument("--pairs", nargs="+", default=["harvest:givens", "hindsight:givens", "hindsight:harvest"])
    a = ap.parse_args()
    out = Path("results/phase4") / a.exp
    rows = []
    for pair in a.pairs:
        x, y = pair.split(":")
        dx = pd.read_csv(out / f"{x}_solved_val.csv").set_index("puzzle")
        dy = pd.read_csv(out / f"{y}_solved_val.csv").set_index("puzzle").loc[dx.index]
        for col in [c for c in dx.columns if c.startswith("valid_")]:
            for band in ("easy", "medium", "hard"):
                m = (dx["band"] == band).to_numpy()
                d = (dx[col].to_numpy() - dy[col].to_numpy())[m].astype(float)
                lo, hi = bootstrap_ci(d)
                rows.append({"a": x, "b": y, "sampler_steps": int(col.split("_")[1]), "band": band, "n": int(m.sum()),
                             "a_valid": dx[col].to_numpy()[m].mean(), "b_valid": dy[col].to_numpy()[m].mean(),
                             "a_minus_b": d.mean(), "lo": lo, "hi": hi})
    df = pd.DataFrame(rows).round(4)
    df.to_csv(out / "arm_pairs.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
