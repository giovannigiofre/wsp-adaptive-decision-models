"""Tables and figures for phase 3 (results/phase3/), all read from committed CSV/JSON.

PYTHONPATH=src python -m phase3.analysis repro
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["repro"])
    {"repro": repro}[ap.parse_args().what]()


if __name__ == "__main__":
    main()
