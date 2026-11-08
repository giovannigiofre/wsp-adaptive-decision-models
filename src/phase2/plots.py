"""Figures for phase 2 (matplotlib PNGs in results/phase2/), all read from committed CSV/JSON.

PYTHONPATH=src python -m phase2.plots sft|strategies|rl|likelihood|calibration|all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path("results/phase2")
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # fixed categorical order
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9, "lines.linewidth": 1.8})


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=110)
    plt.close(fig)
    print("wrote", OUT / name)


def sft():
    df = pd.read_csv(OUT / "sft_training.csv")
    sel = json.load(open(OUT / "sft_selection.json"))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(df.step, df.train_loss, color=C[0], label="train (running mean)")
    ax[0].plot(df.step, df.val_loss, color=C[1], label="val")
    ax[0].set(xlabel="training step", ylabel="masked-diffusion loss (1/t weighted)", title="Loss")
    ax[0].legend()
    ax[1].plot(df.step, df.val_solve, color=C[1], label="val (unseen solution grids)")
    ax[1].plot(df.step, df.train_solve, color=C[0], label="train probe (305 puzzles)")
    lo, hi = 0.3, 0.6
    ax[1].axhspan(lo, hi, color="0.85", alpha=0.5, lw=0, label="target window for weak SFT")
    ax[1].axvline(sel["weak_step"], color="0.3", ls="--", lw=1)
    ax[1].annotate(f"weak SFT: step {sel['weak_step']}\nval {sel['weak_val_solve']:.2f}",
                   (sel["weak_step"], sel["weak_val_solve"]), xytext=(8, -28), textcoords="offset points")
    ax[1].set(xlabel="training step", ylabel="solve rate (confidence, S=8, greedy)", title="Solve rate",
              ylim=(-0.02, 1.02))
    ax[1].legend(loc="lower right")
    _save(fig, "sft_training_curve.png")


def strategies():
    df = pd.read_csv(OUT / "strategies.csv")
    g = df.groupby(["ckpt", "strategy", "steps"]).agg(sr=("solve_rate", "mean"), lo=("ci_lo", "mean"),
                                                      hi=("ci_hi", "mean")).reset_index()
    ck = [c for c in ["sft_weak", "sft_full"] if c in set(g.ckpt)]
    fig, axs = plt.subplots(1, len(ck), figsize=(4.6 * len(ck), 3.4), sharey=True, squeeze=False)
    for ax, c in zip(axs[0], ck):
        for i, s in enumerate(["random", "confidence", "margin", "entropy"]):
            d = g[(g.ckpt == c) & (g.strategy == s)]
            ax.plot(d.steps, d.sr, marker="o", ms=4, color=C[i], label=s)
            ax.fill_between(d.steps, d.lo, d.hi, color=C[i], alpha=0.15, lw=0)
        ax.set(xscale="log", xticks=[1, 2, 4, 8, 16], xticklabels=[1, 2, 4, 8, 16], xlabel="step budget S",
               title=c, ylim=(-0.02, 1.02))
        ax.set_ylabel("test solve rate (95% bootstrap CI)")
        ax.legend()
    _save(fig, "strategies_solve_rate.png")


def _rl_runs():
    rows = []
    for p in sorted((OUT / "rl").glob("*.csv")):
        d = pd.read_csv(p)
        name = p.stem
        variant, tau, seed = name.split("_tau")[0], float(name.split("_tau")[1].split("_seed")[0]), \
            int(name.split("_seed")[1])
        d["variant"], d["tau"], d["seed"] = variant, tau, seed
        rows.append(d)
    return pd.concat(rows)


def _band(ax, d, col, color, label, smooth=1):
    piv = d.pivot_table(index="step", columns="seed", values=col).dropna()
    if smooth > 1:
        piv = piv.rolling(smooth, min_periods=1).mean()
    m, s = piv.mean(1), piv.std(1)
    ax.plot(piv.index, m, color=color, label=f"{label} (n={piv.shape[1]} seeds)")
    ax.fill_between(piv.index, m - s, m + s, color=color, alpha=0.18, lw=0)


def rl():
    df = _rl_runs()
    sel = json.load(open(OUT / "sft_selection.json"))
    groups = [("wd1", 1.0, "wd1 (τ=1)"), ("positive_only", 1.0, "positive-only (w⁻=0)")]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for i, (v, t, lab) in enumerate(groups):
        d = df[(df.variant == v) & (df.tau == t)]
        _band(ax[0], d[d.step > 0], "mean_reward", C[i], lab, smooth=10)
        _band(ax[1], d, "val_solve", C[i], lab)
        _band(ax[2], d[d.step > 0], "logp_correct", C[i], lab + " correct", smooth=10)
        _band(ax[2], d[d.step > 0], "logp_wrong", C[i + 2], lab + " wrong", smooth=10)
    ax[1].axhline(sel["full_val_solve"], color="0.4", ls=":", lw=1, label="fully trained SFT (reference)")
    ax[0].set(xlabel="RL step", ylabel="rollout reward (T=1), 10-step mean", title="Train reward")
    ax[1].set(xlabel="RL step", ylabel="val solve rate (greedy, S=8)", title="Validation")
    ax[2].set(xlabel="RL step", ylabel="one-step log p̂ of rollouts", title="Likelihood of rollouts")
    for a in ax:
        a.legend(fontsize=7)
    _save(fig, "rl_wd1_vs_positive_only.png")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for i, t in enumerate(sorted(df[df.variant == "wd1"].tau.unique())):
        d = df[(df.variant == "wd1") & (df.tau == t)]
        _band(ax[0], d[d.step > 0], "mean_reward", C[i], f"τ={t:g}", smooth=10)
        _band(ax[1], d, "val_solve", C[i], f"τ={t:g}")
    ax[0].set(xlabel="RL step", ylabel="rollout reward, 10-step mean", title="wd1 τ sweep: train reward")
    ax[1].set(xlabel="RL step", ylabel="val solve rate", title="wd1 τ sweep: validation")
    for a in ax:
        a.legend(fontsize=7)
    _save(fig, "rl_tau_sweep.png")
    # summary table across seeds + choice of best run for 2.4
    summ = pd.DataFrame([json.load(open(p)) for p in sorted((OUT / "rl").glob("*.json"))
                         if p.name != "best_run.json"])
    summ.drop(columns=["env"]).to_csv(OUT / "rl_runs.csv", index=False)
    agg = summ.groupby(["variant", "tau"]).agg(
        n=("seed", "count"), final_val_mean=("final_val", "mean"), final_val_std=("final_val", "std"),
        final_test_mean=("final_test", "mean"), final_test_std=("final_test", "std"),
        best_test_mean=("best_test", "mean"), best_test_std=("best_test", "std")).reset_index()
    agg.to_csv(OUT / "rl_summary.csv", index=False)
    print(agg.round(3).to_string())
    w = summ[(summ.variant == "wd1") & (summ.tau == 1.0)]
    best = w.sort_values("best_val", ascending=False).iloc[0]
    json.dump({"name": best["name"], "best_val": float(best.best_val), "best_step": int(best.best_step),
               "rule": "wd1 tau=1 run with the highest val solve rate; its best-val checkpoint"},
              open(OUT / "rl" / "best_run.json", "w"), indent=2)


def likelihood():
    d = pd.read_csv(OUT / "likelihood_estimators.csv")
    s = json.load(open(OUT / "likelihood_estimators.json"))
    cks = list(s)
    fig, axs = plt.subplots(1, len(cks), figsize=(4.4 * len(cks), 3.6), squeeze=False)
    for ax, c in zip(axs[0], cks):
        e = d[d.ckpt == c]
        for ok, col, lab in ((True, C[0], "valid"), (False, C[1], "invalid")):
            f = e[e.correct == ok]
            ax.scatter(f.seq, f.one_step_p0, s=9, alpha=0.6, color=col, label=f"{lab} rollout")
        lim = [min(e.seq.min(), e.one_step_p0.min()), 0.5]
        ax.plot(lim, lim, color="0.5", lw=1, ls=":")
        ax.set(xlabel="sequential log-lik (one cell per pass)", ylabel="one-step log p̂ (p_mask=0)",
               title=f"{c}: Pearson {s[c]['pearson_one_step_p0']:.2f}, Spearman {s[c]['spearman_one_step_p0']:.2f}")
        ax.legend(fontsize=7)
    _save(fig, "likelihood_estimators.png")


def calibration():
    b = pd.read_csv(OUT / "calibration_bins.csv")
    cal = pd.read_csv(OUT / "calibration.csv").set_index("ckpt")
    cks = ["sft_weak", "rl_best"]
    fig, axs = plt.subplots(2, 2, figsize=(8.5, 7.2))
    for j, c in enumerate(cks):
        for i, level in enumerate(["cell", "puzzle"]):
            ax = axs[i, j]
            d = b[(b.ckpt == c) & (b.level == level)]
            ok = d["count"] > 0
            ax.bar(d.bin[ok], d.acc[ok], width=0.09, color=C[0], alpha=0.85, label="accuracy in bin")
            ax.plot([0, 1], [0, 1], color="0.4", ls=":", lw=1, label="perfect calibration")
            ax2 = ax.twinx()
            ax2.step(d.bin, d["count"] / d["count"].sum(), where="mid", color=C[1], lw=1.2, label="share of samples")
            ax2.set_ylim(0, 1)
            ax2.set_ylabel("share of samples", color=C[1])
            ax2.grid(False)
            r = cal.loc[c]
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=f"{'commit confidence' if level == 'cell' else 'mean cell confidence'}",
                   ylabel="fraction correct" if level == "cell" else "fraction solved",
                   title=f"{c} / {level}: ECE {r[level + '_ece']:.3f}, AUROC {r[level + '_auroc']:.3f}")
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="upper left")
    _save(fig, "calibration_before_after.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["sft", "strategies", "rl", "likelihood", "calibration", "all"])
    a = ap.parse_args()
    fns = {"sft": sft, "strategies": strategies, "rl": rl, "likelihood": likelihood, "calibration": calibration}
    for k, f in fns.items():
        if a.what in (k, "all"):
            f()


if __name__ == "__main__":
    main()
