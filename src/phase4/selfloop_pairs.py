"""Step 4.8b: paired differences between arms on the val puzzles, from `<arm>_solved_val.csv`.
95% bootstrap over puzzles of the paired difference (one seed per arm: the interval covers puzzle sampling only).

    PYTHONPATH=src python -m phase4.selfloop_pairs --exp 4.8b --pairs harvest:givens hindsight:givens hindsight:harvest
    PYTHONPATH=src python -m phase4.selfloop_pairs --pooled        # 4.8c: seeds 0 (4.8b), 1, 2 (4.8c) pooled per puzzle
    PYTHONPATH=src python -m phase4.selfloop_pairs --long          # 4.8e: hindsight_long (4.8d) minus givens_long (4.8e)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from phase2.evaluate import bootstrap_ci


def pooled(arms=("hindsight", "harvest"), control="givens"):
    """Seeds pooled per val puzzle (mean of valid over seeds), paired bootstrap over puzzles; per-seed differences beside."""
    R = Path("results/phase4")
    files = {a: [R / "4.8b" / f"{a}_solved_val.csv", R / "4.8c" / f"{a}_s1_solved_val.csv", R / "4.8c" / f"{a}_s2_solved_val.csv"]
             for a in arms}
    runs = {a: [pd.read_csv(f).set_index("puzzle") for f in fs] for a, fs in files.items()}
    ctrl = pd.read_csv(R / "4.8b" / f"{control}_solved_val.csv").set_index("puzzle")
    band = ctrl["band"]
    rows = []
    for col in ("valid_4", "valid_16", "valid_81"):
        mean = {a: sum(d[col] for d in ds) / len(ds) for a, ds in runs.items()}
        for b in ("easy", "medium", "hard"):
            m = (band == b).to_numpy()
            for x, y, per in ((arms[0], arms[1], True), (arms[0], control, False), (arms[1], control, False)):
                my = mean[y] if y in mean else ctrl[col]
                d = (mean[x] - my).to_numpy()[m].astype(float)
                lo, hi = bootstrap_ci(d)
                row = {"a": x, "b": y, "sampler_steps": int(col.split("_")[1]), "band": b,
                       "a_valid": mean[x].to_numpy()[m].mean(), "b_valid": my.to_numpy()[m].mean(),
                       "a_minus_b": d.mean(), "lo": lo, "hi": hi}
                for s in range(3):
                    row[f"a_seed{s}"] = runs[x][s][col].to_numpy()[m].mean()
                    row[f"b_seed{s}"] = runs[y][s][col].to_numpy()[m].mean() if per else float("nan")
                rows.append(row)
    df = pd.DataFrame(rows).round(4)
    df.to_csv(R / "4.8c" / "pooled_pairs.csv", index=False)
    print(df.to_string(index=False))


def long_pairs(split="val"):
    """4.8e: paired differences between the long models per puzzle, 95% bootstrap over puzzles.
    val: hindsight_long - givens_long. test: also each against the supervised model of 4.1."""
    R = Path("results/phase4")
    runs = {"hindsight_long": R / "4.8d", "givens_long": R / "4.8e", "sft": R / "sft"}
    pairs = [("hindsight_long", "givens_long")] + ([("sft", "givens_long"), ("sft", "hindsight_long")] if split == "test" else [])
    d = {n: pd.read_csv(runs[n] / f"{n}_solved_{split}.csv").set_index("puzzle") for n in {x for p in pairs for x in p}}
    rows = []
    for x, y in pairs:
        dx, dy = d[x], d[y].loc[d[x].index]
        for col in ("valid_4", "valid_16", "valid_81"):
            for band in ("easy", "medium", "hard"):
                m = (dx["band"] == band).to_numpy()
                diff = (dx[col].to_numpy() - dy[col].to_numpy())[m].astype(float)
                lo, hi = bootstrap_ci(diff)
                rows.append({"a": x, "b": y, "sampler_steps": int(col.split("_")[1]), "band": band, "n": int(m.sum()),
                             "a_valid": dx[col].to_numpy()[m].mean(), "b_valid": dy[col].to_numpy()[m].mean(),
                             "a_minus_b": diff.mean(), "lo": lo, "hi": hi})
    df = pd.DataFrame(rows).round(4)
    df.to_csv(R / "4.8e" / ("long_pairs.csv" if split == "val" else f"long_pairs_{split}.csv"), index=False)
    print(df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="4.8b")
    ap.add_argument("--pairs", nargs="+", default=["harvest:givens", "hindsight:givens", "hindsight:harvest"])
    ap.add_argument("--pooled", action="store_true")
    ap.add_argument("--long", action="store_true")
    ap.add_argument("--long-test", action="store_true")
    a = ap.parse_args()
    if a.pooled:
        return pooled()
    if a.long:
        return long_pairs()
    if a.long_test:
        return long_pairs("test")
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
