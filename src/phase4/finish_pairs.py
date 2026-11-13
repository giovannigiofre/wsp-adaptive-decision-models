"""Step 4.9c: paired differences on the hard test band for the arms of 4.9c (and the self4 arm of 4.9b).
Seeds pooled per puzzle, 95% bootstrap over puzzles. Each arm against its own start model; arms against self4.

    PYTHONPATH=src python -m phase4.finish_pairs        # results/phase4/4.9c/pairs.csv, val_jumps.csv
    PYTHONPATH=src python -m phase4.finish_pairs --lowlr  # 4.9d: results/phase4/4.9d/pairs.csv, val_rounds.csv
    PYTHONPATH=src python -m phase4.finish_pairs --lowlr-sft  # 4.9e: results/phase4/4.9e/pairs.csv
    PYTHONPATH=src python -m phase4.finish_pairs --after-drop  # 4.9f: results/phase4/4.9f/pairs.csv
    PYTHONPATH=src python -m phase4.finish_pairs --closing  # 4.9g, 4.9h: results/phase4/4.9g/pairs.csv, 4.9h/pairs.csv
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


def lowlr():
    """4.9d: the two long runs continued at lr 1e-5, hard test band, against their starts and the 500-step arms."""
    t, b = pd.read_csv(R / "4.9c" / "test_solved.csv"), pd.read_csv(R / "4.9b" / "test_solved.csv")
    rd = lambda e, n: pd.read_csv(R / e / f"{n}_solved_test.csv")  # noqa: E731
    d = {"hindsight_long": rd("4.8d", "hindsight_long"), "givens_long": rd("4.8e", "givens_long"),
         "hindsight_lr1e-5": rd("4.9d", "hindsight_lr1e-5"), "givens_lr1e-5": rd("4.9d", "givens_lr1e-5")}
    assert all((v.puzzle == t.puzzle).all() for v in d.values()) and (t.puzzle == b.puzzle).all()
    h = (t.band == "hard").to_numpy()
    pool = lambda f, pat, S: np.mean([f[pat.format(s=s) + f"|{S}"].to_numpy()[h] for s in range(3)], 0)  # noqa: E731
    rows = []
    for S in (4, 16, 81):
        v = {k: x[f"valid_{S}"].to_numpy()[h].astype(float) for k, x in d.items()}
        if S != 81:
            v["self4 (4.9b)"], v["hind (4.9c)"] = pool(b, "self_diffusion_seed{s}", S), pool(t, "self_diffusion_seed{s}_hind", S)
        pairs = [("hindsight_lr1e-5", "hindsight_long"), ("givens_lr1e-5", "givens_long"), ("hindsight_lr1e-5", "givens_lr1e-5")]
        pairs += [("hindsight_lr1e-5", "hind (4.9c)"), ("givens_lr1e-5", "self4 (4.9b)")] if S != 81 else []
        for x, y in pairs:
            lo, hi = paired_ci(v[x] - v[y])
            rows.append({"a": x, "b": y, "sampler_steps": S, "a_test": v[x].mean(), "b_test": v[y].mean(),
                         "a_minus_b": (v[x] - v[y]).mean(), "lo": lo, "hi": hi})
    df = pd.DataFrame(rows).round(4)
    df.to_csv(R / "4.9d" / "pairs.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))
    # hard val band per round: the last seven rounds at lr 3e-4 and the four at 1e-5
    out = []
    for n in ("hindsight_lr1e-5", "givens_lr1e-5"):
        c = pd.read_csv(R / "4.9d" / f"{n}.csv")
        c = c[c["round"] >= 28][["round", "step", "train_loss", "val_solve_hard", "val4_solve_hard", "val_solve_medium", "val4_solve_medium"]]
        out.append(c.assign(run=n, lr=np.where(c["round"] > 34, 1e-5, 3e-4)))
    r = pd.concat(out).round(4)
    r.to_csv(R / "4.9d" / "val_rounds.csv", index=False)
    print(r.to_string(index=False))


def lowlr_sft():
    """4.9e: the supervised model continued at lr 1e-5, hard test band, against its start and the 4.9d models."""
    rd = lambda e, n: pd.read_csv(R / e / f"{n}_solved_test.csv")  # noqa: E731
    d = {"sft": rd("sft", "sft"), "sft_lr1e-5": rd("4.9e", "sft_lr1e-5"), "givens_lr1e-5": rd("4.9d", "givens_lr1e-5"),
         "hindsight_lr1e-5": rd("4.9d", "hindsight_lr1e-5")}
    assert all((v.puzzle == d["sft"].puzzle).all() for v in d.values())
    rows = []
    for band in ("hard", "medium"):
        h = (d["sft"].band == band).to_numpy()
        for S in (4, 16, 81):
            v = {k: x[f"valid_{S}"].to_numpy()[h].astype(float) for k, x in d.items()}
            for x, y in (("sft_lr1e-5", "sft"), ("givens_lr1e-5", "sft_lr1e-5"), ("hindsight_lr1e-5", "sft_lr1e-5")):
                lo, hi = paired_ci(v[x] - v[y])
                rows.append({"band": band, "a": x, "b": y, "sampler_steps": S, "a_test": v[x].mean(), "b_test": v[y].mean(),
                             "a_minus_b": (v[x] - v[y]).mean(), "lo": lo, "hi": hi})
    df = pd.DataFrame(rows).round(4)
    df.to_csv(R / "4.9e" / "pairs.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))
    c = pd.read_csv(R / "4.9e" / "sft_lr1e-5.csv")
    print(c[c.step >= 40000][["step", "train_loss", "val_loss", "val_solve_hard", "val_solve_medium", "lr"]].round(4).to_string(index=False))


def after_drop():
    """4.9f: self-training and the supervised control from sft_lr1e-5, hard test band, seeds pooled per puzzle."""
    t, r44 = pd.read_csv(R / "4.9f" / "test_solved.csv"), pd.read_csv(R / "4.4" / "test_solved.csv")
    assert (t.puzzle == r44.puzzle).all()
    h = (t.band == "hard").to_numpy()
    pool = lambda f, pat, S: np.mean([f[pat.format(s=s) + f"|{S}"].to_numpy()[h] for s in range(3)], 0)  # noqa: E731
    rows = []
    for S in (4, 16):
        v = {"self": pool(t, "self_diffusion_seed{s}", S), "sup": pool(t, "supervised_seed{s}", S),
             "sft_lr1e-5": t[f"start|{S}"].to_numpy()[h].astype(float), "self of 4.4": pool(r44, "self_diffusion_seed{s}", S),
             "sft": r44[f"start|{S}"].to_numpy()[h].astype(float)}
        for x, y in (("self", "sft_lr1e-5"), ("sup", "sft_lr1e-5"), ("self", "sup"), ("self", "self of 4.4"), ("self of 4.4", "sft"),
                     ("sft_lr1e-5", "self of 4.4")):
            lo, hi = paired_ci(v[x] - v[y])
            row = {"a": x, "b": y, "sampler_steps": S, "a_test": v[x].mean(), "b_test": v[y].mean(), "a_minus_b": (v[x] - v[y]).mean(),
                   "lo": lo, "hi": hi}
            for s in range(3):
                row[f"a_seed{s}"] = (t[f"{'self_diffusion' if x == 'self' else 'supervised'}_seed{s}|{S}"].to_numpy()[h].mean()
                                     if x in ("self", "sup") else float("nan"))
            rows.append(row)
    df = pd.DataFrame(rows).round(4)
    df.to_csv(R / "4.9f" / "pairs.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))


def closing():
    """4.9g: wd1 and positive-only from sft_lr1e-5. 4.9h: own valid grids from givens_lr1e-5. Hard test band, seeds pooled."""
    pool = lambda f, pat, S, h: np.mean([f[pat.format(s=s) + f"|{S}"].to_numpy()[h] for s in range(3)], 0)  # noqa: E731
    seeds = lambda f, pat, S, h: [f[pat.format(s=s) + f"|{S}"].to_numpy()[h].mean() for s in range(3)]  # noqa: E731
    pd.set_option("display.width", 250)

    def table(v, per, pairs, out):
        rows = []
        for S in (4, 16):
            for x, y in pairs:
                lo, hi = paired_ci(v[S][x] - v[S][y])
                row = {"a": x, "b": y, "sampler_steps": S, "a_test": v[S][x].mean(), "b_test": v[S][y].mean(),
                       "a_minus_b": (v[S][x] - v[S][y]).mean(), "lo": lo, "hi": hi}
                for s in range(3):
                    row[f"a_seed{s}"] = per[S][x][s] if x in per[S] else float("nan")
                rows.append(row)
        df = pd.DataFrame(rows).round(4)
        df.to_csv(out, index=False)
        print(df.to_string(index=False))

    g, f = pd.read_csv(R / "4.9g" / "test_solved.csv"), pd.read_csv(R / "4.9f" / "test_solved.csv")
    assert (g.puzzle == f.puzzle).all() and (g["start|4"] == f["start|4"]).all()
    h = (g.band == "hard").to_numpy()
    pat = {"wd1": (g, "wd1_beta1_tau1_seed{s}"), "pos": (g, "wd1_beta0_tau1_seed{s}"),
           "self (4.9f)": (f, "self_diffusion_seed{s}"), "sup (4.9f)": (f, "supervised_seed{s}")}
    v = {S: {k: pool(d, p, S, h) for k, (d, p) in pat.items()} | {"sft_lr1e-5": g[f"start|{S}"].to_numpy()[h].astype(float)} for S in (4, 16)}
    per = {S: {k: seeds(d, p, S, h) for k, (d, p) in pat.items()} for S in (4, 16)}
    table(v, per, (("wd1", "sft_lr1e-5"), ("pos", "sft_lr1e-5"), ("pos", "wd1"), ("wd1", "self (4.9f)"), ("pos", "self (4.9f)"),
                   ("wd1", "sup (4.9f)")), R / "4.9g" / "pairs.csv")

    t, b = pd.read_csv(R / "4.9h" / "test_solved.csv"), pd.read_csv(R / "4.9b" / "test_solved.csv")
    assert (t.puzzle == b.puzzle).all() and (t.puzzle == g.puzzle).all()
    pat = {"self": (t, "self_diffusion_seed{s}"), "self4 (4.9b)": (b, "self_diffusion_seed{s}")}
    v = {S: {k: pool(d, p, S, h) for k, (d, p) in pat.items()} | {"givens_lr1e-5": t[f"start|{S}"].to_numpy()[h].astype(float),
             "givens_long": b[f"start|{S}"].to_numpy()[h].astype(float), "sft_lr1e-5": g[f"start|{S}"].to_numpy()[h].astype(float)} for S in (4, 16)}
    per = {S: {k: seeds(d, p, S, h) for k, (d, p) in pat.items()} for S in (4, 16)}
    table(v, per, (("self", "givens_lr1e-5"), ("self", "self4 (4.9b)"), ("self", "sft_lr1e-5"), ("givens_lr1e-5", "givens_long"),
                   ("self", "givens_long")), R / "4.9h" / "pairs.csv")


if __name__ == "__main__":
    import sys
    closing() if "--closing" in sys.argv else after_drop() if "--after-drop" in sys.argv else lowlr() if "--lowlr" in sys.argv else lowlr_sft() if "--lowlr-sft" in sys.argv else main()
