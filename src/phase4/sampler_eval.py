"""Step 4.2: the plateau model of 4.1 under different unmasking strategies and step budgets (inference only).

    PYTHONPATH=src python -m phase4.sampler_eval --size medium

Writes to results/phase4/4.2/: strategies.csv (+ table, plot), paired differences to `confidence`, oracle-order
headroom on the hard band (phase3.order_policy.oracle_order: one cell per step, commit a cell the model currently
gets right; it uses the solution, so it is an analysis tool, not a method), and calibration of the commit
confidence per band.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from phase2.evaluate import auroc, bootstrap_ci, calibration_data, ece, reliability_bins, run_sampler
from phase2.model import build_model
from phase2.sampler import STRATEGIES
from phase2.utils import env_record, get_device, seed_all
from phase3.order_policy import oracle_order

from .data import load_sets
from .generate import cache_dir

BUDGETS = (1, 4, 8, 16, 32, 81)
COLORS = {"random": "#7f7f7f", "confidence": "#2a78b5", "margin": "#d9822b", "entropy": "#8e44ad", "oracle": "#2e8b57"}


def paired_ci(d: np.ndarray, n: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase4.yaml")
    ap.add_argument("--size", default="medium")
    ap.add_argument("--ckpt", default=None, help="default: final.pt of the 4.1 run `sft`")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="results/phase4/4.2")
    ap.add_argument("--calib-steps", type=int, default=16)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    cfg["model"].update(cfg["sizes"][a.size])
    dev = get_device(cfg["runtime"]["device"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ckpt = Path(a.ckpt).expanduser() if a.ckpt else cache_dir(cfg) / "sft" / "sft" / "final.pt"
    sets, bands, names = load_sets(cfg, dev, splits=(a.split,))
    ps, band = sets[a.split], bands[a.split].numpy()
    model = build_model(cfg).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.eval()
    masks = [("all", np.ones(len(ps), bool))] + [(n, band == b) for b, n in enumerate(names)]
    t_start = time.time()

    # --- strategies x budgets -------------------------------------------------------------------------------
    rows, per_puzzle = [], {}
    for strat in STRATEGIES:
        for S in BUDGETS:
            runs = []
            for sd in ([0, 1, 2] if strat == "random" else [0]):
                seed_all(sd)
                t0 = time.time()
                solved = run_sampler(model, ps, S, strat, 0.0, batch=1024)["solved"].numpy().astype(float)
                runs.append(solved)
                for label, m in masks:
                    lo, hi = bootstrap_ci(solved[m])
                    rows.append({"strategy": strat, "steps": S, "order_seed": sd, "band": label, "n": int(m.sum()),
                                 "solve_rate": solved[m].mean(), "ci_lo": lo, "ci_hi": hi, "wall_s": time.time() - t0})
            per_puzzle[strat, S] = np.mean(runs, 0)
        print(strat, "done", f"{time.time() - t_start:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "strategies.csv", index=False)
    table = df.groupby(["band", "strategy", "steps"]).solve_rate.mean().unstack("steps").round(3)
    table.to_csv(out / "strategies_table.csv")
    print(table.to_string())
    diffs = []
    for strat in ("random", "margin", "entropy"):
        for S in BUDGETS:
            d = per_puzzle[strat, S] - per_puzzle["confidence", S]
            for label, m in masks:
                lo, hi = paired_ci(d[m])
                diffs.append({"strategy": strat, "minus": "confidence", "steps": S, "band": label, "diff": d[m].mean(),
                              "ci_lo": lo, "ci_hi": hi})
    dd = pd.DataFrame(diffs)
    dd.to_csv(out / "strategies_paired_diff.csv", index=False)
    print(dd[dd.band.isin(["all", "hard"])].round(3).to_string(index=False))

    # --- oracle-order headroom on the hard band (one cell per step) -------------------------------------------
    hard = torch.from_numpy(band == names.index("hard")).to(dev)
    hs = ps.subset(hard)
    orc = oracle_order(model, hs)
    o = orc["solved"].numpy().astype(float)
    c81 = per_puzzle["confidence", 81][band == names.index("hard")]
    lo, hi = bootstrap_ci(o); dlo, dhi = paired_ci(o - c81)
    stuck = orc["stuck_at"].numpy()
    oracle = {"band": "hard", "n": len(o), "oracle_solve": o.mean(), "oracle_ci": [lo, hi],
              "confidence_81_solve": c81.mean(), "oracle_minus_confidence_81": (o - c81).mean(), "diff_ci": [dlo, dhi],
              "share_never_stuck": float((stuck < 0).mean()),
              "median_stuck_step_when_stuck": float(np.median(stuck[stuck >= 0])) if (stuck >= 0).any() else None,
              "mean_hidden_cells": float(hs.k.float().mean())}
    json.dump(oracle, open(out / "oracle_hard.json", "w"), indent=1)
    print(json.dumps(oracle, indent=1))

    # --- calibration of the commit confidence, per band -------------------------------------------------------
    res = run_sampler(model, ps, a.calib_steps, "confidence", 0.0, batch=1024)
    cal, bins = [], []
    for label, m in masks:
        idx = torch.from_numpy(m)
        sub = {k: v[idx] for k, v in res.items()}
        cc, co, pc, po = calibration_data(sub, ps.subset(idx.to(dev)).to("cpu"))
        cal.append({"band": label, "steps": a.calib_steps, "solve_rate": po.mean(), "cell_acc": co.mean(),
                    "mean_cell_conf": cc.mean(), "cell_ece": ece(cc, co), "cell_auroc": auroc(cc, co),
                    "conf_correct_cells": cc[co].mean(), "conf_wrong_cells": cc[~co].mean() if (~co).any() else np.nan,
                    "mean_puzzle_conf": pc.mean(), "puzzle_ece": ece(pc, po), "puzzle_auroc": auroc(pc, po),
                    "n_cells": len(cc), "n_wrong_cells": int((~co).sum()), "n_puzzles": len(pc)})
        ctr, acc, mc, cnt = reliability_bins(cc, co.astype(float))
        bins += [{"band": label, "bin": x, "acc": a_, "mean_conf": m_, "count": int(n_)} for x, a_, m_, n_ in zip(ctr, acc, mc, cnt)]
    cal = pd.DataFrame(cal); cal.to_csv(out / "calibration.csv", index=False)
    bins = pd.DataFrame(bins); bins.to_csv(out / "calibration_bins.csv", index=False)
    print(cal.round(4).T.to_string())

    # --- plots ------------------------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = df.groupby(["band", "strategy", "steps"]).agg(y=("solve_rate", "mean"), lo=("ci_lo", "mean"), hi=("ci_hi", "mean")).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
    for ax, b in zip(axes, names):
        for strat in STRATEGIES:
            d = g[(g.band == b) & (g.strategy == strat)]
            ax.errorbar(d.steps, d.y, yerr=[d.y - d.lo, d.hi - d.y], color=COLORS[strat], marker="o", ms=4, lw=1.8, capsize=2,
                        label=strat + (" (mean of 3 orders)" if strat == "random" else ""))
        if b == "hard":
            ax.axhline(oracle["oracle_solve"], color=COLORS["oracle"], ls="--", lw=1.5, label="oracle order (uses solutions)")
        ax.set_xscale("log"); ax.set_xticks(list(BUDGETS)); ax.set_xticklabels([str(x) for x in BUDGETS]); ax.minorticks_off()
        ax.set_title(f"{b} band (n = {int((band == names.index(b)).sum())})", fontsize=10); ax.set_xlabel("sampler steps")
        ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False); ax.set_ylim(0, 1.02)
    axes[0].set_ylabel(f"{a.split} solve rate (95% bootstrap)"); axes[2].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "strategies.png", dpi=110); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
    for ax, b in zip(axes, names):
        d = bins[(bins.band == b) & (bins["count"] > 0)]
        ax.plot([0, 1], [0, 1], color="#bbbbbb", lw=1, ls="--")
        ax.plot(d.mean_conf, d.acc, color=COLORS["confidence"], marker="o", ms=5, lw=1.8)
        for x, y, n in zip(d.mean_conf, d.acc, d["count"]):
            ax.annotate(f"{n:,}", (x, y), textcoords="offset points", xytext=(-4, 6), fontsize=6.5, color="#555555", ha="right")
        r = cal[cal.band == b].iloc[0]
        au = "no wrong cell" if np.isnan(r.cell_auroc) else f"AUROC {r.cell_auroc:.3f}"
        ax.set_title(f"{b}: ECE {r.cell_ece:.3f}, {au}", fontsize=10)
        ax.set_xlabel("confidence at commit (cells per bin shown)"); ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("share of committed cells that are correct")
    fig.tight_layout(); fig.savefig(out / "calibration.png", dpi=110); plt.close(fig)

    json.dump({"ckpt": str(ckpt), "split": a.split, "n": len(ps), "wall_clock_s": time.time() - t_start, "env": env_record(dev)},
              open(out / "run.json", "w"), indent=1)
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
