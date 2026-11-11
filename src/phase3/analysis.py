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
import numpy as np  # noqa: E402
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
               "w_pos_on_invalid", "logp_valid", "logp_invalid", "pred_entropy", "clip_frac", "frac_equal"]


def compact(exp: str, src: Path | None = None, dst: Path | None = None):
    """Per-step logs in the local cache -> committed summary: one row per evaluation (window means of
    the per-step diagnostics since the previous evaluation) in curves.csv, one row per run in runs.csv.
    `src`/`dst` and the optional `*_ref` columns are used by phase 4 (phase4.rl_analysis)."""
    src, dst = src or CACHE / exp / "logs", dst or OUT / exp
    dst.mkdir(parents=True, exist_ok=True)
    curves, runs = [], []
    for p in sorted(src.glob("*.json")):
        j = json.load(open(p))
        df = pd.read_csv(p.with_suffix(".csv"))
        ev = df.dropna(subset=["val_solve"])[[k for k in ("step", "val_solve", "val_solve_s1", "val_solve_ref")
                                              if k in df.columns]]
        cols = [c for c in WINDOW_COLS if c in df.columns]
        win = pd.cut(df.step, [-1] + ev.step.tolist(), labels=ev.step.tolist()).astype(float)
        means = df[df.step > 0].groupby(win[df.step > 0], observed=True)[cols].mean().rename_axis("step").reset_index()
        cfg = {"run": j["name"], "objective": j["objective"], **j["objective_params"], "reward": j["reward"],
               "ref_mix": j.get("ref_mix", 0.0), "ref_kl": j.get("ref_kl", 0.0),
               "likelihood": j.get("likelihood", "onestep"), "tag": j.get("tag", ""),
               "seed": j["seed"], **{k: j[k] for k in ("order_temp", "weights_over", "inner_steps", "grad_clip", "weight_decay",
                                                "rollout_steps", "supervised_on") if k in j}}
        curves.append(ev.merge(means, on="step", how="left").assign(**cfg))
        reach = ev.step[ev.val_solve >= 0.7]
        runs.append({**cfg, "steps": j["steps"], "lr": j["lr"], "init": Path(j["init"]).name,
                     "final_val": j["final_val"], "final_test": j["final_test"],
                     "final_val_s1": j["final_val_s1"], "final_test_s1": j["final_test_s1"],
                     "best_val": j["best_val"], "best_step": j["best_step"], "best_test": j["best_test"],
                     **{k: j[k] for k in ("final_val_ref", "final_test_ref", "band", "eval_steps", "eval_steps_ref")
                        if k in j},
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
    ax[0].legend(loc="center right", fontsize=8)
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


def load_many(*exps: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cs, rs = zip(*[tuple(d.assign(exp=e) for d in load(e)) for e in exps])
    c, r = pd.concat(cs, ignore_index=True), pd.concat(rs, ignore_index=True)
    for d in (c, r):
        d["tag"] = d["tag"].fillna("")
    return c, r


def ramp(cmap, n: int):
    """n steps of one hue, light to dark, for an ordered quantity."""
    return [cmap(x) for x in np.linspace(0.35, 0.95, n)]


def best_tau() -> float:
    return json.load(open(OUT / "3.1b" / "best_tau.json"))["best_tau"]


def show(**tables):
    for name, df in tables.items():
        print(f"--- {name}\n{df.round(3).to_string(index=False)}")


def exp_3_1b():
    """Sharp weights: tau sweep for wd1 and positive-only."""
    exp = "3.1b"
    c, r = load(exp)
    taus = sorted(c.tau.unique(), reverse=True)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for j, (b, lab, cmap) in enumerate([(1.0, "wd1", plt.cm.Blues), (0.0, "positive-only", plt.cm.Oranges)]):
        for t, col in zip(taus, ramp(cmap, len(taus))):
            band(ax[j], c[(c.beta == b) & (c.tau == t)], "val_solve", col, f"τ={t:g}")
        ax[j].set(title=f"{lab}: validation by τ", xlabel="RL step",
                  ylabel="val solve rate (greedy, S=8)", ylim=(-0.02, 1.02))
        ax[j].legend(fontsize=8, loc="lower right", title="darker = sharper weights", title_fontsize=8)
    early = c[(c.step > 0) & (c.step <= 50)].groupby(["beta", "tau"]).w_pos_on_invalid.mean().reset_index()
    for i, (b, lab) in enumerate([(1.0, "wd1"), (0.0, "positive-only")]):
        d = early[early.beta == b].sort_values("tau")
        ax[2].plot(d.tau, d.w_pos_on_invalid, marker="o", ms=5, color=C[i], label=lab)
    ax[2].set(xscale="log", xticks=taus, xticklabels=[f"{t:g}" for t in taus], xlabel="τ (log scale)",
              ylabel="share of w⁺ on invalid rollouts, steps 1–50", title="How selective the weights are",
              ylim=(-0.02, 0.6))
    ax[2].legend(fontsize=8)
    save(fig, exp, "tau_sweep.png")
    names = {1.0: "wd1", 0.0: "positive-only"}
    r["variant"] = r.beta.map(names)
    summ = seed_table(r, ["variant", "tau"], cols=("final_val", "final_test", "final_test_s1", "min_val"))
    summ = summ.merge(early.assign(variant=early.beta.map(names))[["variant", "tau", "w_pos_on_invalid"]],
                      on=["variant", "tau"]).rename(columns={"w_pos_on_invalid": "w_pos_on_invalid_steps_1_50"})
    summ.round(4).to_csv(OUT / exp / "summary.csv", index=False)
    w = summ[(summ.variant == "wd1") & (summ.tau < 1)]
    ties = w[w.final_val_mean >= w.final_val_mean.max() - 0.01]
    bt = {"best_tau": float(ties.tau.max()),
          "rule": "tau in {0.25, 0.1, 0.05} with the highest seed-mean final val solve rate for wd1; "
                  "within 0.01 of the best counts as a tie and the largest tau is taken",
          "wd1_final_val_by_tau": {f"{t:g}": round(v, 4) for t, v in zip(w.tau, w.final_val_mean)}}
    json.dump(bt, open(OUT / exp / "best_tau.json", "w"), indent=2)
    show(summary=summ)
    print(bt)


BETAS = [0.0, 0.25, 0.5, 1.0, 2.0]


def exp_3_1c():
    """Negative dose: beta sweep (beta 0 and 1 from 3.1b) and the floored negative term."""
    exp, bt = "3.1c", best_tau()
    c, r = load_many("3.1c", "3.1b")
    c, r = c[c.tau.isin([1.0, bt])], r[r.tau.isin([1.0, bt])]
    cm, rm = c[c.floor.isna()], r[r.floor.isna()]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for b, col in zip(BETAS, ramp(plt.cm.Purples, len(BETAS))):
        band(ax[0], cm[(cm.tau == 1.0) & (cm.beta == b)], "val_solve", col, f"β={b:g}")
    ax[0].set(title="Validation at τ=1 (β=0 positive-only, β=1 wd1)", xlabel="RL step",
              ylabel="val solve rate (greedy, S=8)", ylim=(-0.02, 1.02))
    ax[0].legend(fontsize=8, loc="lower right")
    g = rm.groupby(["tau", "beta"]).agg(test=("final_test", "mean"), test_sd=("final_test", "std"),
                                        low=("min_val", "mean"), low_sd=("min_val", "std")).reset_index()
    for i, t in enumerate(sorted(g.tau.unique(), reverse=True)):
        d = g[g.tau == t].set_index("beta").reindex(BETAS)
        for a, m, s in ((ax[1], "test", "test_sd"), (ax[2], "low", "low_sd")):
            a.errorbar(range(len(BETAS)), d[m], yerr=d[s], marker="o", ms=5, capsize=3, color=C[2 + i],
                       label=f"τ={t:g}")
    for a, title, yl in ((ax[1], "Final test solve rate", "test solve rate (mean ± std, 3 seeds)"),
                         (ax[2], "Lowest val solve rate during training", "lowest val solve rate")):
        a.set(xticks=range(len(BETAS)), xticklabels=[f"{b:g}" for b in BETAS], xlabel="β (weight of the negative term)",
              ylabel=yl, title=title, ylim=(-0.02, 1.02))
        a.legend(fontsize=8)
    save(fig, exp, "beta_sweep.png")
    summ = seed_table(rm, ["tau", "beta"])
    summ.round(4).to_csv(OUT / exp / "summary.csv", index=False)
    fl = r[(r.tau == 1.0) & (r.beta == 1.0)].assign(floor=lambda d: d.floor.fillna(0).map(lambda x: "none" if x == 0 else f"{x:g}"))
    late = c[(c.tau == 1.0) & (c.beta == 1.0) & (c.step > c.step.max() - 250)].assign(
        floor=lambda d: d.floor.fillna(0).map(lambda x: "none" if x == 0 else f"{x:g}"))
    floor = seed_table(fl, ["floor"]).merge(
        late.groupby("floor")[["logp_valid", "logp_invalid", "pred_entropy"]].mean().add_suffix("_last_250_steps")
        .reset_index(), on="floor")
    floor.round(4).to_csv(OUT / exp / "floor.csv", index=False)
    show(beta_sweep=summ, floor=floor)


def exp_3_1d_sft():
    """Pilot: SFT to convergence on N solutions; pick the N whose final val is closest to 0.45 (window 0.3-0.6)."""
    exp = "3.1d"
    js = [json.load(open(p)) for p in sorted((OUT / exp).glob("sft_subset_n*_seed0.json"))]
    df = pd.DataFrame([{k: j[k] for k in ("name", "n_solutions", "n_puzzles", "steps", "final_val_solve",
                                           "final_test_solve", "final_seen_solutions_solve",
                                           "final_unseen_train_solve", "wall_clock_s")} |
                       {"val_min_last_1000": j["val_solve_range_last_1000_steps"][0],
                        "val_max_last_1000": j["val_solve_range_last_1000_steps"][1]} for j in js])
    df = df.sort_values("n_solutions")
    inwin = df[(df.final_val_solve >= 0.3) & (df.final_val_solve <= 0.6)]
    pick = (inwin if len(inwin) else df).assign(d=lambda d: (d.final_val_solve - 0.45).abs()).sort_values("d").iloc[0]
    df["chosen"] = df.name == pick["name"]
    df.round(4).to_csv(OUT / exp / "sft_subset_summary.csv", index=False)
    (OUT / exp / "chosen_subset.txt").write_text(pick["name"] + "\n")
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for (_, row), col in zip(df.iterrows(), ramp(plt.cm.Greens, len(df))):
        d = pd.read_csv(OUT / exp / f"{row['name']}.csv")
        ax.plot(d.step, d.val_solve, color=col, label=f"N={row.n_solutions}")
    ax.axhspan(0.3, 0.6, color="0.85", alpha=0.5, lw=0)
    ax.set(xlabel="SFT step", ylabel="val solve rate (greedy, S=8)", ylim=(-0.02, 1.02),
           title="SFT on the puzzles of N train solutions")
    ax.legend(fontsize=8, title="grey band: target window")
    save(fig, exp, "sft_subset.png")
    show(sft_subset=df)
    print("chosen:", pick["name"], "in window" if len(inwin) else "(none in the 0.3-0.6 window; closest to 0.45)")


def exp_3_1d():
    """Starting model: two other SFT snapshots and a model converged on few solutions."""
    exp = "3.1d"
    c, r = load_many("3.1d", "3.1b")
    c, r = c[c.tau == 1.0], r[r.tau == 1.0]
    sub = (OUT / exp / "chosen_subset.txt").read_text().strip()
    n = sub.split("_n")[1].split("_")[0]
    starts = [("_sft100", "SFT step 100"), ("", "SFT step 120 (phase 2 start)"), ("_sft130", "SFT step 130"),
              ("_subset", f"converged on {n} solutions")]
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.3), sharey=True)
    rows = []
    for ax, (tag, title) in zip(axs, starts):
        for i, (b, lab) in enumerate(WD1_VS_POS):
            d = c[(c.tag == tag) & (c.beta == b)]
            band(ax, d, "val_solve", C[i], lab)
            rr = r[(r.tag == tag) & (r.beta == b)]
            rows.append({"start": title, "variant": lab, "n": len(rr), "start_val": d[d.step == 0].val_solve.mean(),
                         "final_val_mean": rr.final_val.mean(), "final_val_std": rr.final_val.std(),
                         "final_test_mean": rr.final_test.mean(), "final_test_std": rr.final_test.std(),
                         "min_val_mean": rr.min_val.mean(), "min_val_step_median": rr.min_val_step.median()})
        ax.set(title=title, xlabel="RL step", ylim=(-0.02, 1.02))
    axs[0].set_ylabel("val solve rate (greedy, S=8)")
    axs[0].legend(fontsize=8, loc="center right")
    save(fig, exp, "starting_model.png")
    summ = pd.DataFrame(rows)
    summ["gain_val"] = summ.final_val_mean - summ.start_val
    summ.round(4).to_csv(OUT / exp / "summary.csv", index=False)
    show(summary=summ)


def exp_3_1e():
    """Graded reward vs binary (binary cells from 3.1b)."""
    exp, bt = "3.1e", best_tau()
    c, r = load_many("3.1e", "3.1b")
    c, r = c[c.tau.isin([1.0, bt])], r[r.tau.isin([1.0, bt])]
    taus = sorted(c.tau.unique(), reverse=True)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for j, t in enumerate(taus):
        for i, (b, lab) in enumerate(WD1_VS_POS):
            for rew, ls in (("binary", "-"), ("graded", "--")):
                band(ax[j], c[(c.tau == t) & (c.beta == b) & (c.reward == rew)], "val_solve", C[i],
                     f"{lab.split(' (')[0]}, {rew}", ls=ls)
        ax[j].set(title=f"Validation at τ={t:g} (dashed = graded reward)", xlabel="RL step",
                  ylabel="val solve rate (greedy, S=8)", ylim=(-0.02, 1.02))
        ax[j].legend(fontsize=7, loc="lower right")
    for rew, ls in (("binary", "-"), ("graded", "--")):
        band(ax[2], c[(c.tau == 1.0) & (c.beta == 1.0) & (c.reward == rew) & (c.step > 0)], "frac_skipped", C[0],
             f"wd1, {rew}", ls=ls)
    ax[2].set(title="Groups with equal rewards, wd1 at τ=1", xlabel="RL step",
              ylabel="share of groups skipped (25-step mean)", ylim=(-0.02, 1.02))
    ax[2].legend(fontsize=8)
    save(fig, exp, "graded_reward.png")
    r["variant"] = r.beta.map(dict(WD1_VS_POS))
    summ = seed_table(r, ["tau", "variant", "reward"])
    early = c[(c.step > 0) & (c.step <= 300)].assign(variant=lambda d: d.beta.map(dict(WD1_VS_POS)))
    summ = summ.merge(early.groupby(["tau", "variant", "reward"]).frac_skipped.mean().rename("frac_skipped_steps_1_300")
                      .reset_index(), on=["tau", "variant", "reward"])
    v300 = c[c.step == 300].assign(variant=lambda d: d.beta.map(dict(WD1_VS_POS))).groupby(
        ["tau", "variant", "reward"]).val_solve.mean().rename("val_at_300").reset_index()
    summ = summ.merge(v300, on=["tau", "variant", "reward"])
    summ.round(4).to_csv(OUT / exp / "summary.csv", index=False)
    show(summary=summ)


def exp_3_1f():
    """Reference policy in the rollouts (no-reference cell from 3.1b)."""
    exp = "3.1f"
    c, r = load_many("3.1f", "3.1b")
    c, r = c[(c.tau == 1.0) & (c.beta == 1.0)], r[(r.tau == 1.0) & (r.beta == 1.0)]
    mixes = sorted(c.ref_mix.unique())
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for m, col in zip(mixes, ramp(plt.cm.Blues, len(mixes))):
        d = c[c.ref_mix == m]
        band(ax[0], d, "val_solve", col, f"mix={m:g}")
        band(ax[1], d[d.step > 0], "mean_valid", col, f"mix={m:g}")
        band(ax[2], d[d.step > 0], "frac_skipped", col, f"mix={m:g}")
    ax[0].set(title="wd1: validation (greedy, model alone)", xlabel="RL step", ylabel="val solve rate (S=8)",
              ylim=(-0.02, 1.02))
    ax[1].set(title="Valid rollouts (sampled from the mixture)", xlabel="RL step",
              ylabel="share of valid rollouts (25-step mean)", ylim=(-0.02, 1.02))
    ax[2].set(title="Groups with equal rewards", xlabel="RL step", ylabel="share of groups skipped",
              ylim=(-0.02, 1.02))
    ax[0].legend(fontsize=8, loc="lower right", title="darker = more reference")
    save(fig, exp, "reference_mix.png")
    summ = seed_table(r, ["ref_mix"])
    late = c[c.step > c.step.max() - 250].groupby("ref_mix")[["frac_skipped", "mean_valid"]].mean().add_suffix(
        "_last_250_steps").reset_index()
    summ = summ.merge(late, on="ref_mix")
    summ.round(4).to_csv(OUT / exp / "summary.csv", index=False)
    show(summary=summ)


def exp_3_1g():
    """Do the better variants survive other starts? (wd1 and positive-only from the same starts: 3.1d)"""
    exp = "3.1g"
    c, r = load_many("3.1g", "3.1d")
    for d in (c, r):
        d["ref_kl"] = d["ref_kl"].fillna(0.0)

    def arm(row) -> str:
        if row.objective == "filtered":
            return "filtered imitation"
        if row.objective == "self_diffusion":
            return "self-training, SFT loss"
        if row.ref_kl > 0:
            return "wd1 + KL 0.01"
        if row.floor == row.floor:      # not NaN
            return "wd1, floor −12"
        return dict(WD1_VS_POS)[row.beta]

    c["arm"], r["arm"] = c.apply(arm, axis=1), r.apply(arm, axis=1)
    n = (OUT / "3.1d" / "chosen_subset.txt").read_text().strip().split("_n")[1].split("_")[0]
    styles = [("wd1 (τ=1)", C[0], "-"), ("wd1 + KL 0.01", C[0], "--"), ("wd1, floor −12", C[0], ":"),
              ("positive-only (w⁻=0)", C[1], "-"), ("filtered imitation", C[5], "-"),
              ("self-training, SFT loss", C[4], "-")]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, (tag, title) in zip(axs, (("_sft100", "Start: SFT step 100 (val 0.11)"),
                                      ("_subset", f"Start: converged on {n} solutions (val 0.65)"))):
        for lab, col, ls in styles:
            band(ax, c[(c.tag == tag) & (c.arm == lab)], "val_solve", col, lab, ls=ls)
        ax.set(title=title, xlabel="RL step", ylim=(-0.02, 1.02))
    axs[0].set_ylabel("val solve rate (greedy, S=8)")
    axs[0].legend(fontsize=7, loc="center right")
    save(fig, exp, "other_starts.png")
    r["start"] = r.tag.map({"_sft100": "SFT step 100", "_subset": f"converged on {n} solutions"})
    g = r.groupby(["start", "arm"])
    summ = g.agg(n=("seed", "count"), final_test_mean=("final_test", "mean"), final_test_std=("final_test", "std"),
                 best_val_mean=("best_val", "mean"), best_test_mean=("best_test", "mean"),
                 final_val_mean=("final_val", "mean"), min_val_mean=("min_val", "mean")).reset_index()
    summ["max_drop_from_best_val"] = g.apply(lambda d: (d.best_val - d.final_val).max()).values
    summ = summ.sort_values(["start", "final_test_mean"], ascending=[True, False])
    summ.round(4).to_csv(OUT / exp / "summary.csv", index=False)
    show(summary=summ)


def exp_3_6a(exp: str = "3.6a"):
    """Chain-rule likelihoods in the loss (1000 steps) against the one-step runs of 3.1b/3.1c at matched steps.
    3.6a: fixed-order sequential. 3.6c adds the likelihood along the rollout's own path."""
    liks = (("onestep", "-"), ("sequential", "--")) + ((("trajectory", ":"),) if exp == "3.6c" else ())
    c, r = load_many(*(["3.6c"] if exp == "3.6c" else []), "3.6a", "3.1b", "3.1c")
    for d in (c, r):
        d["likelihood"] = d["likelihood"].fillna("onestep")
    c, r = c[c.tau == 1.0], r[r.tau == 1.0]

    def arm(row) -> str:
        if row.beta == 0.0:
            return "positive-only"
        if row.floor == row.floor:
            return "wd1, floored"
        return "wd1"

    c["arm"], r["arm"] = c.apply(arm, axis=1), r.apply(arm, axis=1)
    c = c[c.beta.isin([0.0, 1.0]) & ~((c.arm == "wd1, floored") & (c.floor == -15))]
    r = r[r.beta.isin([0.0, 1.0]) & ~((r.arm == "wd1, floored") & (r.floor == -15))]
    cols = {"wd1": C[0], "wd1, floored": C[2], "positive-only": C[1]}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for lab, col in cols.items():
        for lik, ls in liks:
            d = c[(c.arm == lab) & (c.likelihood == lik) & (c.step <= 1000)]
            band(ax[0], d, "val_solve", col, f"{lab}, {lik}", ls=ls)
            if lab != "positive-only":
                band(ax[1], d[d.step > 0], "logp_valid", col, f"{lab}, {lik}", ls=ls)
    ax[0].set(title="Validation (dashed: sequential, dotted: rollout path)" if exp == "3.6c"
              else "Validation (dashed = sequential likelihood in the loss)", xlabel="RL step",
              ylabel="val solve rate (greedy, S=8)", ylim=(-0.02, 1.02))
    ax[1].set(title="Log-likelihood of valid rollouts, each by its own estimate", xlabel="RL step",
              ylabel="log p̂ of valid rollouts (25-step mean)")
    ax[0].legend(fontsize=7, loc="lower right")
    ax[1].legend(fontsize=7, loc="lower left")
    save(fig, exp, "sequential_likelihood.png" if exp == "3.6a" else "rollout_path_likelihood.png")
    v1000 = c[c.step == 1000].groupby(["arm", "likelihood"]).val_solve.agg(["mean", "std"]).add_prefix("val_at_1000_")
    lo = c.groupby(["arm", "likelihood", "seed"]).val_solve.min().groupby(["arm", "likelihood"]).mean().rename("min_val_mean")
    first = c[(c.step > 0) & (c.step <= 50)].groupby(["arm", "likelihood"]).logp_valid.mean().rename("logp_valid_steps_1_50")
    last = c[(c.step > 750) & (c.step <= 1000)].groupby(["arm", "likelihood"]).logp_valid.mean().rename("logp_valid_steps_751_1000")
    summ = pd.concat([v1000, lo, first, last], axis=1).reset_index()
    summ["logp_valid_change"] = summ.logp_valid_steps_751_1000 - summ.logp_valid_steps_1_50
    fin = r[r.likelihood != "onestep"].groupby(["arm", "likelihood"]).agg(
        final_test_mean=("final_test", "mean"), final_test_std=("final_test", "std")).reset_index()
    summ = summ.merge(fin, on=["arm", "likelihood"], how="left")
    summ.round(4).to_csv(OUT / exp / "summary.csv", index=False)
    show(summary=summ)


def exp_3_6b():
    """Rarer reward (SFT step 90) and a weaker checker (lenient reward)."""
    exp = "3.6b"
    c, r = load_many("3.6b", "3.1b", "3.1c", "3.2")
    for d in (c, r):
        d["ref_kl"] = d["ref_kl"].fillna(0.0)

    def arm(row) -> str:
        if row.objective == "filtered":
            return "filtered imitation"
        if row.objective == "self_diffusion":
            return "self-training, SFT loss"
        if row.objective == "wd1" and row.beta == 1.0 and row.ref_kl == 0 and row.tau == 1.0:
            return "wd1, floor −12" if row.floor == -12 else ("wd1" if row.floor != row.floor else "")
        return ""

    c["arm"], r["arm"] = c.apply(arm, axis=1), r.apply(arm, axis=1)
    c, r = c[(c.arm != "") & (c.seed <= 2)], r[(r.arm != "") & (r.seed <= 2)]
    ref = (c.exp != "3.6b") & (c.reward == "binary") & (c.tag == "")        # strict reward, phase 2 start
    styles = [("wd1", C[0], "-"), ("wd1, floor −12", C[0], ":"), ("filtered imitation", C[5], "-"),
              ("self-training, SFT loss", C[4], "-")]
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.6))
    for lab, col, ls in styles:
        band(axs[0], c[(c.exp == "3.6b") & (c.tag == "_sft90") & (c.arm == lab)], "val_solve", col, lab, ls=ls)
        band(axs[1], c[(c.exp == "3.6b") & (c.reward == "lenient") & (c.arm == lab)], "val_solve", col, lab, ls=ls)
        band(axs[2], c[(c.exp == "3.6b") & (c.reward == "lenient") & (c.arm == lab) & (c.step > 0)], "mean_reward",
             col, lab, ls=ls)
    axs[0].set(title="Rarer reward: start at SFT step 90 (val 0.007)", xlabel="RL step",
               ylabel="val solve rate (greedy, S=8)", ylim=(-0.02, 1.02))
    axs[1].set(title="Weaker checker: true solve rate", xlabel="RL step", ylabel="val solve rate (true validity)",
               ylim=(-0.02, 1.02))
    axs[2].set(title="Weaker checker: the rewarded quantity", xlabel="RL step",
               ylabel="share of rollouts with valid rows and columns", ylim=(-0.02, 1.02))
    axs[0].legend(fontsize=7, loc="center right")
    save(fig, exp, "rarer_reward_weaker_checker.png")
    rows = []
    for lab, _, _ in styles:
        base = r[ref.reindex(r.index, fill_value=False) & (r.arm == lab)] if False else \
            r[(r.exp != "3.6b") & (r.reward == "binary") & (r.tag == "") & (r.arm == lab)]
        sparse = r[(r.exp == "3.6b") & (r.tag == "_sft90") & (r.arm == lab)]
        len_ = r[(r.exp == "3.6b") & (r.reward == "lenient") & (r.arm == lab)]
        late = c[(c.exp == "3.6b") & (c.reward == "lenient") & (c.arm == lab) & (c.step > 1250)].mean_reward.mean()
        rows.append({"arm": lab, "strict_reward_from_step_120_test": base.final_test.mean(),
                     "rarer_reward_test": sparse.final_test.mean(), "rarer_reward_test_std": sparse.final_test.std(),
                     "rarer_reward_min_val": sparse.min_val.mean(),
                     "weaker_checker_true_test": len_.final_test.mean(), "weaker_checker_true_test_std": len_.final_test.std(),
                     "weaker_checker_rewarded_share_last_250": late, "weaker_checker_min_val": len_.min_val.mean()})
    summ = pd.DataFrame(rows)
    summ["gap_true_vs_strict"] = summ.strict_reward_from_step_120_test - summ.weaker_checker_true_test
    summ.round(4).to_csv(OUT / exp / "summary.csv", index=False)
    show(summary=summ)


def method(row) -> str:
    if row.objective == "wd1":
        return dict(WD1_VS_POS)[row.beta]
    if row.objective == "reinforce":
        return "REINFORCE, normalised" if row.normalize is True or row.normalize == 1.0 else "REINFORCE"
    return {"filtered": "filtered imitation", "self_diffusion": "self-training, SFT loss", "grpo": "d1-style GRPO",
            "supervised": "supervised reference (labels)"}[row.objective]


CONTRAST = [("wd1 (τ=1)", C[0]), ("REINFORCE", C[2]), ("REINFORCE, normalised", C[3]), ("d1-style GRPO", C[4])]
PUSH_UP = [("positive-only (w⁻=0)", C[1]), ("filtered imitation", C[5]), ("self-training, SFT loss", C[4])]
REFERENCE = ("supervised reference (labels)", "0.35")


def exp_3_2():
    """wd1 next to simpler objectives."""
    exp = "3.2"
    c, r = load_many("3.2", "3.1b")       # wd1 and positive-only at tau = 1, seeds 0-2, come from 3.1b
    c, r = c[(c.exp == "3.2") | (c.tau == 1.0)], r[(r.exp == "3.2") | (r.tau == 1.0)]
    c["method"], r["method"] = c.apply(method, axis=1), r.apply(method, axis=1)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    for a, fam, title in ((ax[0], CONTRAST, "Raise valid, lower invalid rollouts"),
                          (ax[1], PUSH_UP, "Only raise (valid) rollouts")):
        for lab, col in fam:
            band(a, c[c.method == lab], "val_solve", col, lab)
        band(a, c[c.method == REFERENCE[0]], "val_solve", REFERENCE[1], "supervised reference", ls=":")
        a.set(title=title, xlabel="RL step", ylabel="val solve rate (greedy, S=8)", ylim=(-0.02, 1.02))
        a.legend(fontsize=7, loc="lower right")
    order = r.groupby("method").final_test.mean().sort_values().index.tolist()
    colour = dict(CONTRAST + PUSH_UP + [REFERENCE])
    for y, m in enumerate(order):
        d = r[r.method == m]
        ax[2].scatter(d.final_test, [y] * len(d), s=22, color=colour[m], alpha=0.75, zorder=3)
        ax[2].plot([d.final_test.mean()] * 2, [y - 0.3, y + 0.3], color="0.15", lw=1.5, zorder=4)
    ax[2].set(yticks=range(len(order)), yticklabels=order, xlabel="final test solve rate (dots: seeds, bar: mean)",
              title="Final test solve rate", xlim=(min(0.5, r.final_test.min() - 0.03), 1.01))
    ax[2].grid(axis="y", visible=False)
    save(fig, exp, "objectives.png")
    summ = seed_table(r, ["method"], cols=("final_val", "final_test", "final_test_s1", "min_val"))
    reach = r.assign(reached=r["first_step_val_0.7"].where(r["first_step_val_0.7"] >= 0)).groupby("method").agg(
        first_step_val_07_median=("reached", "median"), seeds_reaching_07=("reached", "count")).reset_index()
    summ = summ.merge(reach, on="method").sort_values("final_test_mean", ascending=False)
    summ.round(4).to_csv(OUT / exp / "summary.csv", index=False)
    r[["method", "seed", "final_val", "final_test", "final_test_s1", "min_val", "min_val_step"]].sort_values(
        ["method", "seed"]).round(4).to_csv(OUT / exp / "per_seed.csv", index=False)
    show(summary=summ)


def reliability(ax, conf: np.ndarray, solved: np.ndarray, color, label: str, bins=None, ls="-"):
    """Accuracy per confidence bin; marker area grows with the number of puzzles in the bin."""
    from phase2.evaluate import reliability_bins
    if bins is None:
        _, acc, mc, cnt = reliability_bins(conf, solved, 10)
    else:   # discrete stated values
        mc = np.array(bins)
        cnt = np.array([(np.isclose(conf, b)).sum() for b in bins])
        acc = np.array([solved[np.isclose(conf, b)].mean() if n else np.nan for b, n in zip(bins, cnt)])
    ok = cnt > 0
    ax.plot(mc[ok], acc[ok], color=color, ls=ls, lw=1.5, label=label, zorder=3)
    ax.scatter(mc[ok], acc[ok], s=12 + 150 * cnt[ok] / cnt.sum(), color=color, zorder=4, edgecolor="white", lw=0.8)


def exp_3_3(exp: str = "3.3"):
    """Confidence token with a Brier-shaped reward vs probe vs implicit confidence.
    3.3b is the same analysis for the frozen-denoiser runs; the no-token baseline is always 3.3's."""
    rows = []
    for p in sorted((OUT / exp).glob("conf_*.json")):
        j = json.load(open(p))
        rows.append({"arm": f"token {j['place']}, {j['mode']}", "seed": j["seed"], **j["test"],
                     "base_rate_brier": j["test"]["solve"] * (1 - j["test"]["solve"]),
                     "wall_clock_s": j["wall_clock_s"]})
    runs = pd.DataFrame(rows)
    runs.round(4).to_csv(OUT / exp / "runs.csv", index=False)
    cols = ["solve", "base_rate_brier", "q_expected_brier", "q_expected_ece", "q_expected_auroc", "q_stated_brier",
            "q_stated_ece", "implicit_brier", "implicit_ece", "implicit_auroc"]
    summ = runs.groupby("arm")[cols].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ.round(4).to_csv(OUT / exp / "summary.csv", index=False)
    base = pd.read_csv(OUT / "3.3" / "baseline_implicit.csv")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    wd1 = [p for p in sorted((OUT / "3.3").glob("baseline_wd1_seed*_test.csv"))]
    if wd1:
        d = pd.concat([pd.read_csv(p) for p in wd1])
        s = d.solved.to_numpy()
        reliability(ax[0], d.implicit.to_numpy(), s, C[0], "commit confidence, raw")
        reliability(ax[0], d.slope.to_numpy(), s, C[2], "recalibrated, 1 parameter")
        reliability(ax[0], d.platt.to_numpy(), s, C[3], "recalibrated, 2 parameters")
    ax[0].set_title("No token: implicit confidence (wd1 runs)")
    buckets = [0.125, 0.375, 0.625, 0.875]
    for a, place in ((ax[1], "last"), (ax[2], "first")):
        for mode, col in (("rl", C[1]), ("probe", C[4])):
            fs = sorted((OUT / exp).glob(f"conf_{mode}_{place}_*test.csv"))
            if fs:
                d = pd.concat([pd.read_csv(p) for p in fs])
                reliability(a, d.q_stated.to_numpy(), d.solved.to_numpy(), col,
                            "Brier-shaped reward (RL)" if mode == "rl" else "expected-Brier probe", bins=buckets)
        a.set_title(f"Token read from the {'completed grid' if place == 'last' else 'puzzle alone'}")
    for a in ax:
        a.plot([0, 1], [0, 1], color="0.5", ls=":", lw=1)
        a.set(xlabel="stated confidence", ylabel="share of puzzles solved (test, seeds pooled)", xlim=(-0.02, 1.02),
              ylim=(-0.02, 1.02))
        a.legend(fontsize=8, loc="lower right")
    save(fig, exp, "calibration.png")
    show(token_arms=summ, implicit_baseline=base.drop(columns=[c for c in base.columns if c.endswith("_params")]))


ORDER_VARIANTS = [("reinforce", "probs", "zero", "RL, 3 heuristic scores, random start", C[0], "-"),
                  ("reinforce", "probs", "confidence", "RL, 3 heuristic scores, confidence start", C[1], "-"),
                  ("reinforce", "hidden", "zero", "RL, + hidden state, random start", C[2], "-"),
                  ("reinforce", "hidden", "confidence", "RL, + hidden state, confidence start", C[3], "-"),
                  ("oracle", "probs", "zero", "teacher (uses solutions), 3 scores", C[4], "--"),
                  ("oracle", "hidden", "zero", "teacher (uses solutions), + hidden state", C[5], "--")]


def exp_3_4b():
    """Learned unmasking order on the frozen denoiser."""
    exp = "3.4b"
    js = [json.load(open(p)) for p in sorted((OUT / exp).glob("*.json"))]
    runs = pd.DataFrame([{k: j[k] for k in ("name", "method", "features", "init", "lr", "seed", "steps", "best_val",
                                             "best_step", "final_val", "final_test", "best_test",
                                             "final_test_agree_conf", "wall_clock_s")} |
                         {"device": j["env"]["device"]} for j in js])
    curves = pd.concat([pd.read_csv(OUT / exp / f"{n}.csv").assign(name=n) for n in runs.name]).merge(
        runs[["name", "method", "features", "init", "seed"]], on="name")
    ref = pd.read_csv(OUT / "3.4a" / "oracle.csv").set_index(["split", "order"]).solve_rate
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.8), gridspec_kw={"width_ratios": [1.15, 1]})
    labels = {}
    for m, f, i, lab, col, ls in ORDER_VARIANTS:
        d = curves[(curves.method == m) & (curves.features == f) & (curves.init == i)]
        labels[(m, f, i)] = (lab, col)
        if len(d):
            band(ax[0], d, "val_solve", col, lab, ls=ls)
    for name, y in (("oracle order", ref["val", "oracle"]), ("confidence rule", ref["val", "confidence"]),
                    ("random order", ref["val", "random"])):
        ax[0].axhline(y, color="0.4", ls=":", lw=1)
        ax[0].annotate(name, (curves.step.max(), y), xytext=(-4, 3), textcoords="offset points", ha="right",
                       fontsize=7, color="0.3")
    ax[0].set(xlabel="training step", ylabel="val solve rate (greedy order, one cell per step)", ylim=(-0.02, 1.04),
              title="Learning the order on the frozen weak denoiser")
    ax[0].legend(fontsize=7, loc="lower right", bbox_to_anchor=(1.0, 0.16))
    runs["variant"] = [labels[(m, f, i)][0] for m, f, i in zip(runs.method, runs.features, runs.init)]
    order = runs.groupby("variant").final_test.mean().sort_values().index.tolist()
    colour = {lab: col for lab, col in labels.values()}
    for y, v in enumerate(order):
        d = runs[runs.variant == v]
        ax[1].scatter(d.final_test, [y] * len(d), s=22, color=colour[v], alpha=0.75, zorder=3)
        ax[1].plot([d.final_test.mean()] * 2, [y - 0.3, y + 0.3], color="0.15", lw=1.5, zorder=4)
    for name, x in (("confidence", ref["test", "confidence"]), ("oracle", ref["test", "oracle"])):
        ax[1].axvline(x, color="0.4", ls=":", lw=1)
        ax[1].annotate(name, (x, len(order) - 0.45), xytext=(-3, 0), textcoords="offset points", ha="right",
                       fontsize=7, color="0.3")
    ax[1].set(yticks=range(len(order)), yticklabels=order, xlabel="final test solve rate (dots: seeds, bar: mean)",
              title="Test", ylim=(-0.6, len(order) - 0.2))
    ax[1].grid(axis="y", visible=False)
    save(fig, exp, "order_policy.png")
    runs.round(4).to_csv(OUT / exp / "runs.csv", index=False)
    summ = runs.groupby("variant").agg(n=("seed", "count"), final_test_mean=("final_test", "mean"),
                                       final_test_std=("final_test", "std"), final_val_mean=("final_val", "mean"),
                                       agree_with_confidence=("final_test_agree_conf", "mean"),
                                       lr=("lr", "first")).reset_index().sort_values("final_test_mean", ascending=False)
    summ.round(4).to_csv(OUT / exp / "summary.csv", index=False)
    show(summary=summ)
    print(ref.unstack("split").round(3).to_string())


def exp_3_2b():
    """KL to the starting model: GRPO without it, REINFORCE and wd1 with it."""
    exp = "3.2b"
    c, r = load_many("3.2b", "3.2", "3.1b")
    keep = lambda d: d[(d.exp != "3.1b") | (d.tau == 1.0)]   # noqa: E731
    c, r = keep(c), keep(r)
    c, r = c[c.seed <= 2], r[r.seed <= 2]
    for d in (c, r):
        d["ref_kl"] = d["ref_kl"].fillna(0.0)

    def arm(row) -> str:
        if row.objective == "grpo":
            return "GRPO (KL 0.04)" if row.kl > 0 else "GRPO, KL removed"
        if row.objective == "reinforce" and row.normalize is not True and row.normalize != 1.0:
            return "REINFORCE + KL 0.04" if row.ref_kl > 0 else "REINFORCE"
        if row.objective == "wd1" and row.beta == 1.0:
            return f"wd1 + KL {row.ref_kl:g}" if row.ref_kl > 0 else "wd1"
        return ""

    c["arm"], r["arm"] = c.apply(arm, axis=1), r.apply(arm, axis=1)
    c, r = c[c.arm != ""], r[r.arm != ""]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
    for a, arms in ((ax[0], [("REINFORCE", C[2], "-"), ("REINFORCE + KL 0.04", C[2], "--"),
                             ("GRPO (KL 0.04)", C[4], "--"), ("GRPO, KL removed", C[4], "-")]),
                    (ax[1], [("wd1", C[0], "-"), ("wd1 + KL 0.01", C[0], "-."), ("wd1 + KL 0.04", C[0], "--")])):
        for lab, col, ls in arms:
            band(a, c[c.arm == lab], "val_solve", col, lab, ls=ls)
        a.set(xlabel="RL step", ylabel="val solve rate (greedy, S=8)", ylim=(-0.02, 1.02))
        a.legend(fontsize=8, loc="lower right", title="dashed = with KL to the start", title_fontsize=8)
    ax[0].set_title("REINFORCE and GRPO")
    ax[1].set_title("wd1")
    for lab, col, ls in [("wd1", C[0], "-"), ("wd1 + KL 0.04", C[0], "--"), ("REINFORCE", C[2], "-"),
                         ("REINFORCE + KL 0.04", C[2], "--")]:
        band(ax[2], c[(c.arm == lab) & (c.step > 0)], "logp_valid", col, lab, ls=ls)
    ax[2].set(title="Estimated likelihood of valid rollouts", xlabel="RL step", ylabel="one-step log p̂ (25-step mean)")
    ax[2].legend(fontsize=8, loc="lower left")
    save(fig, exp, "kl_to_start.png")
    summ = seed_table(r, ["arm"])
    late = c[c.step > c.step.max() - 250].groupby("arm")[["logp_valid", "logp_invalid", "frac_skipped"]].mean().add_suffix(
        "_last_250_steps").reset_index()
    summ = summ.merge(late, on="arm").sort_values("final_test_mean", ascending=False)
    summ.round(4).to_csv(OUT / exp / "summary.csv", index=False)
    show(summary=summ)


EXPERIMENTS: dict = {"3.6a": exp_3_6a, "3.6c": lambda: exp_3_6a("3.6c"), "3.6b": exp_3_6b, "3.3b": lambda: exp_3_3("3.3b"), "3.1g": exp_3_1g, "3.2b": exp_3_2b, "3.3": exp_3_3, "3.4b": exp_3_4b, "3.1a": exp_3_1a, "3.1b": exp_3_1b, "3.1c": exp_3_1c, "3.1d-sft": exp_3_1d_sft,
                     "3.1d": exp_3_1d, "3.1e": exp_3_1e, "3.1f": exp_3_1f, "3.2": exp_3_2}


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
