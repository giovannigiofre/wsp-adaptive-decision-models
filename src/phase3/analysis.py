"""Tables and figures for phase 3 (results/phase3/), all read from committed CSV/JSON.

PYTHONPATH=src python -m phase3.analysis repro
PYTHONPATH=src python -m phase3.analysis compact 3.1a   # local per-step logs -> results/phase3/3.1a/
PYTHONPATH=src python -m phase3.analysis 3.1a           # tables and figures for one experiment
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from phase2.plots import C  # noqa: E402  (same palette and rcParams as the phase 2 figures)

from .rl import CACHE  # noqa: E402

P2, OUT = Path("results/phase2/rl"), Path("results/phase3")
# (label, colour, phase 2 run prefix, phase 3 run prefix); colours as in rl_wd1_vs_positive_only.png
REPRO = [("wd1 (τ=1)", C[0], "wd1_tau1", "wd1_beta1_tau1"),
         ("positive-only (w⁻=0)", C[1], "positive_only_tau1", "wd1_beta0_tau1")]


def _runs(folder: Path, prefix: str) -> dict[int, tuple[pd.DataFrame, dict]]:
    out = {}
    for p in sorted(folder.glob(f"{prefix}_seed*.json")):
        j = json.load(open(p))
        out[j["seed"]] = (pd.read_csv(p.with_suffix(".csv")), j)
    return out


def repro():
    """3.0: phase 3 trainer vs phase 2 script, same settings and seeds."""
    rows, agree, diag = [], [], []
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, (label, colour, p2, p3) in zip(axs, REPRO):
        old, new = _runs(P2, p2), _runs(OUT / "rl", p3)
        for source, runs in (("phase2", old), ("phase3", new)):
            for seed, (df, j) in runs.items():
                v = df.dropna(subset=["val_solve"])
                rows.append({"variant": label, "source": source, "seed": seed, "final_val": j["final_val"],
                             "final_test": j["final_test"], "min_val": v.val_solve.min(),
                             "min_val_step": int(v.step[v.val_solve.idxmin()])})
        for seed, (df, _) in new.items():
            v = df.dropna(subset=["val_solve"])
            ax.plot(v.step, v.val_solve, color=colour, lw=2.6, alpha=0.75,
                    label="phase 3 trainer" if seed == min(new) else None)
            early = df[(df.step >= 1) & (df.step <= 50)]
            diag.append({"variant": label, "seed": seed, **early[["w_pos_on_invalid", "w_ess"]].mean().to_dict(),
                         "w_ess_min_all_steps": df.w_ess.min()})
            if seed in old:
                o = old[seed][0].dropna(subset=["val_solve"])
                d = (v.set_index("step").val_solve - o.set_index("step").val_solve).abs()
                agree.append({"variant": label, "seed": seed, "mean_abs_val_diff": d.mean(),
                              "max_abs_val_diff": d.max(), "first_step_differing": int(d[d > 0].index.min())
                              if (d > 0).any() else -1})
        for seed, (df, _) in old.items():
            v = df.dropna(subset=["val_solve"])
            ax.plot(v.step, v.val_solve, color="0.2", lw=0.9, ls="--",
                    label="phase 2 script" if seed == min(old) else None)
        ax.set(xlabel="RL step", title=f"{label}: 3 seeds each", ylim=(-0.02, 1.02))
        ax.legend(loc="upper left" if "wd1 (" in label else "upper center", fontsize=8)
    axs[0].set_ylabel("val solve rate (greedy, S=8)")
    fig.tight_layout()
    fig.savefig(OUT / "repro_3.0.png", dpi=110)
    plt.close(fig)
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "repro_3.0_runs.csv", index=False)
    summ = runs.groupby(["variant", "source"]).agg(
        n=("seed", "count"), final_val_mean=("final_val", "mean"), final_val_std=("final_val", "std"),
        final_test_mean=("final_test", "mean"), final_test_std=("final_test", "std"),
        min_val_mean=("min_val", "mean"), min_val_step_median=("min_val_step", "median")).reset_index()
    summ.to_csv(OUT / "repro_3.0_summary.csv", index=False)
    pd.set_option("display.width", 200)
    for name, df in (("runs", runs), ("summary", summ), ("curve agreement, same seed", pd.DataFrame(agree)),
                     ("weight diagnostics, steps 1-50", pd.DataFrame(diag))):
        print(f"--- {name}\n{df.round(3).to_string(index=False)}")
    print("wrote", OUT / "repro_3.0.png")


WINDOW_COLS = ["mean_reward", "mean_valid", "frac_skipped", "loss", "grad_norm", "w_entropy", "w_ess",
               "w_pos_on_invalid", "logp_valid", "logp_invalid", "pred_entropy", "clip_frac"]


def compact(exp: str):
    """Per-step logs in the local cache -> committed summary: one row per evaluation (window means of
    the per-step diagnostics since the previous evaluation) in curves.csv, one row per run in runs.csv."""
    src, dst = CACHE / exp / "logs", OUT / exp
    dst.mkdir(parents=True, exist_ok=True)
    curves, runs = [], []
    for p in sorted(src.glob("*.json")):
        j = json.load(open(p))
        df = pd.read_csv(p.with_suffix(".csv"))
        ev = df.dropna(subset=["val_solve"])[["step", "val_solve", "val_solve_s1"]]
        cols = [c for c in WINDOW_COLS if c in df.columns]
        win = pd.cut(df.step, [-1] + ev.step.tolist(), labels=ev.step.tolist()).astype(float)
        means = df[df.step > 0].groupby(win[df.step > 0], observed=True)[cols].mean().rename_axis("step").reset_index()
        cfg = {"run": j["name"], "objective": j["objective"], **j["objective_params"], "reward": j["reward"],
               "ref_mix": j.get("ref_mix", 0.0), "tag": j.get("tag", ""), "seed": j["seed"]}
        curves.append(ev.merge(means, on="step", how="left").assign(**cfg))
        reach = ev.step[ev.val_solve >= 0.7]
        runs.append({**cfg, "steps": j["steps"], "lr": j["lr"], "init": Path(j["init"]).name,
                     "final_val": j["final_val"], "final_test": j["final_test"],
                     "final_val_s1": j["final_val_s1"], "final_test_s1": j["final_test_s1"],
                     "best_val": j["best_val"], "best_step": j["best_step"], "best_test": j["best_test"],
                     "min_val": ev.val_solve.min(), "min_val_step": int(ev.step[ev.val_solve.idxmin()]),
                     "first_step_val_0.7": int(reach.iloc[0]) if len(reach) else -1,
                     "wall_clock_s": round(j["wall_clock_s_this_session"], 1), "torch": j["env"]["torch"],
                     "device": j["env"]["device"], "code_commit": j["code_commit"]})
    lead = ["run", "objective", "reward", "tag", "seed", "step"]
    c = pd.concat(curves)
    c[lead + [k for k in c.columns if k not in lead]].round(4).to_csv(dst / "curves.csv", index=False)
    pd.DataFrame(runs).round(4).to_csv(dst / "runs.csv", index=False)
    print(f"compacted {len(runs)} runs -> {dst}/curves.csv, runs.csv")


def load(exp: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(OUT / exp / "curves.csv"), pd.read_csv(OUT / exp / "runs.csv")


def band(ax, d: pd.DataFrame, col: str, color, label: str, ls="-"):
    """Mean line with a +/- 1 std band across seeds."""
    piv = d.pivot_table(index="step", columns="seed", values=col)
    m, s = piv.mean(axis=1), piv.std(axis=1)
    ax.plot(piv.index, m, color=color, ls=ls, label=f"{label} (n={piv.shape[1]})")
    ax.fill_between(piv.index, m - s, m + s, color=color, alpha=0.18, lw=0)


def seed_table(runs: pd.DataFrame, by: list[str], cols=("final_val", "final_test", "final_test_s1", "min_val")):
    g = runs.groupby(by)
    out = g.agg(n=("seed", "count"), **{f"{c}_{f}": (c, f) for c in cols for f in ("mean", "std")})
    out["min_val_step_median"] = g.min_val_step.median()
    return out.reset_index()


def save(fig, exp: str, name: str):
    fig.tight_layout()
    fig.savefig(OUT / exp / name, dpi=110)
    plt.close(fig)
    print("wrote", OUT / exp / name)


def at_steps(curves: pd.DataFrame, by: list[str], steps: list[int], col: str = "val_solve") -> pd.DataFrame:
    """Seed mean and std of `col` at the given steps, one row per group."""
    d = curves[curves.step.isin(steps)]
    g = d.groupby(by + ["step"])[col].agg(["mean", "std"]).unstack("step")
    g.columns = [f"{col}@{s}_{f}" for f, s in g.columns]
    return g[sorted(g.columns, key=lambda k: (int(k.split("@")[1].split("_")[0]), k))].reset_index()


WD1_VS_POS = [(1.0, "wd1 (τ=1)"), (0.0, "positive-only (w⁻=0)")]   # (beta, label); colours C[0], C[1]


def exp_3_1a():
    """Horizon: wd1 and positive-only for 2000 steps."""
    exp = "3.1a"
    c, r = load(exp)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for i, (b, lab) in enumerate(WD1_VS_POS):
        d = c[c.beta == b]
        band(ax[0], d, "val_solve", C[i], lab)
        band(ax[1], d, "val_solve_s1", C[i], lab)
        band(ax[2], d[d.step > 0], "frac_skipped", C[i], lab)
    for a in ax[:2]:
        a.axvline(500, color="0.4", ls=":", lw=1)
        a.set_ylim(-0.02, 1.02)
    ax[0].annotate("phase 2 stopped here", (500, 0.02), xytext=(6, 0), textcoords="offset points", fontsize=8,
                   color="0.3")
    ax[0].set(xlabel="RL step", ylabel="val solve rate (greedy, S=8)", title="Validation, 8 unmasking steps")
    ax[1].set(xlabel="RL step", ylabel="val solve rate (greedy, S=1)", title="Validation, one shot")
    ax[2].set(xlabel="RL step", ylabel="share of groups skipped (25-step mean)",
              title="Groups with equal rewards (no signal)", ylim=(-0.02, 1.02))
    ax[0].legend(loc="lower right", fontsize=8)
    save(fig, exp, "horizon.png")
    r["variant"] = r.beta.map(dict(WD1_VS_POS))
    c["variant"] = c.beta.map(dict(WD1_VS_POS))
    table = at_steps(c, ["variant"], [500, 1000, 1500, 2000])
    table.round(4).to_csv(OUT / exp / "val_by_step.csv", index=False)
    summ = seed_table(r, ["variant"])
    summ.round(4).to_csv(OUT / exp / "summary.csv", index=False)
    m = c.groupby(["variant", "step"]).val_solve.mean().unstack("variant")
    close = ((m - m.loc[2000]).abs() <= 0.02).all(axis=1)
    first = int(close[close[::-1].cumprod()[::-1].astype(bool)].index.min())   # stays within 0.02 from here on
    first_any = int(close[close].index.min())
    horizon = {"first_step_within_0.02_of_step_2000": first_any, "first_step_staying_within_0.02": first,
               "rule_horizon": int(-(-first_any // 500) * 500)}
    json.dump(horizon, open(OUT / exp / "horizon.json", "w"), indent=2)
    late = c[c.step > 1500].groupby("variant")[["frac_skipped", "mean_valid", "logp_valid", "logp_invalid",
                                                "pred_entropy", "grad_norm"]].mean()
    for name, df in (("val by step", table), ("final (step 2000)", summ), ("steps 1501-2000", late.reset_index())):
        print(f"--- {name}\n{df.round(3).to_string(index=False)}")
    print(r[["variant", "seed", "final_val", "final_test", "final_test_s1", "min_val", "min_val_step",
             "wall_clock_s"]].sort_values(["variant", "seed"]).round(3).to_string(index=False))
    print(horizon)


EXPERIMENTS: dict = {"3.1a": exp_3_1a}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", help="repro | compact | an experiment id")
    ap.add_argument("exp", nargs="?")
    a = ap.parse_args()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    if a.what == "repro":
        repro()
    elif a.what == "compact":
        compact(a.exp)
    else:
        EXPERIMENTS[a.what]()


if __name__ == "__main__":
    main()
