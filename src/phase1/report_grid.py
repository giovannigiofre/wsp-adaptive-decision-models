"""Step 1.2 aggregation: results/phase1/raw/<task>.csv -> results/phase1/strategies.csv and
one accuracy-vs-steps plot per task with 95% bootstrap intervals.

Usage: python -m src.phase1.report_grid
"""
from __future__ import annotations

import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .calibration import bootstrap_ci
from .load import REPO, load_config

COLORS = {"random": "#8172B2", "confidence": "#4C72B0", "margin": "#DD8452", "entropy": "#55A868"}


def main():
    cfg = load_config()
    nboot = cfg["calibration"]["bootstrap_samples"]
    out_rows = []
    for task in ["sudoku", "countdown", "gsm8k"]:
        p = REPO / f"results/phase1/raw/{task}.csv"
        if not p.exists():
            continue
        groups = defaultdict(list)
        for r in csv.DictReader(open(p)):
            groups[(r["strategy"], int(r["steps"]), int(r["gen_length"]))].append(r)
        # only keep examples completed for every config, so all cells use the same examples
        ids_per = [set(r["id"] for r in g) for g in groups.values()]
        common = set.intersection(*ids_per)
        fig, ax = plt.subplots(figsize=(5, 3.6), constrained_layout=True)
        by_strat = defaultdict(list)
        for (s, st, gl), g in sorted(groups.items()):
            g = [r for r in g if r["id"] in common]
            sc = np.array([float(r["score"]) for r in g])
            cor = np.array([float(r["correct"]) for r in g])
            lo, hi = bootstrap_ci(sc, nboot)
            clo, chi = bootstrap_ci(cor, nboot)
            row = {"task": task, "strategy": s, "steps": st, "gen_length": gl, "n": len(g),
                   "accuracy": sc.mean(), "acc_lo": lo, "acc_hi": hi,
                   "solved": cor.mean(), "solved_lo": clo, "solved_hi": chi,
                   "mean_wall_s": np.mean([float(r["wall_s"]) for r in g]),
                   "forward_passes": int(np.mean([int(r["nfe"]) for r in g]))}
            out_rows.append(row)
            by_strat[s].append(row)
        for s, rows in by_strat.items():
            x = np.array([r["steps"] for r in rows]) * (1 + 0.012 * list(COLORS).index(s))
            y = np.array([r["accuracy"] for r in rows])
            err = np.array([[r["accuracy"] - r["acc_lo"], r["acc_hi"] - r["accuracy"]] for r in rows]).T
            ax.errorbar(x, y, yerr=err, marker="o", capsize=3, label=s, color=COLORS.get(s))
        steps = sorted({r["steps"] for r in out_rows if r["task"] == task})
        ax.set_xscale("log", base=2)
        ax.set_xticks(steps, [str(s) for s in steps])
        ax.minorticks_off()
        n = len(common)
        metric = "cell accuracy (d1 scorer)" if task == "sudoku" else "accuracy (d1 scorer)"
        ax.set_xlabel(f"denoising steps (gen_length={rows[0]['gen_length']}, block 32)")
        ax.set_ylabel(metric)
        ax.set_ylim(bottom=0)
        ax.set_title(f"{task}: n={n} examples, 95% bootstrap CI", fontsize=10)
        ax.legend(fontsize=8)
        fig.savefig(REPO / f"results/phase1/strategies_{task}.png", dpi=90)
        plt.close(fig)
    with open(REPO / "results/phase1/strategies.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        w.writeheader()
        for r in out_rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    for r in out_rows:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    paired(nboot)


def paired(nboot, ref="confidence"):
    """Paired comparison: per example, mean score over the step budgets for each strategy;
    then the mean of (ref - other) with a 95% bootstrap CI over examples.
    Writes results/phase1/strategies_paired.csv."""
    rows = []
    for task in ["sudoku", "countdown", "gsm8k"]:
        p = REPO / f"results/phase1/raw/{task}.csv"
        if not p.exists():
            continue
        per = defaultdict(lambda: defaultdict(list))
        for r in csv.DictReader(open(p)):
            per[r["strategy"]][r["id"]].append(float(r["score"]))
        if ref not in per:
            continue
        for s in sorted(per):
            if s == ref:
                continue
            ids = sorted(set(per[ref]) & set(per[s]))
            d = np.array([np.mean(per[ref][i]) - np.mean(per[s][i]) for i in ids])
            lo, hi = bootstrap_ci(d, nboot)
            rows.append({"task": task, "comparison": f"{ref} - {s}", "n_examples": len(ids),
                         "mean_diff": round(float(d.mean()), 4), "lo": round(lo, 4), "hi": round(hi, 4)})
    with open(REPO / "results/phase1/strategies_paired.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
