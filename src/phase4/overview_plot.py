"""Phase 4 overview figure: the three long models on the hard test band, before and after the learning-rate drop.

Reads the committed per-model test tables (valid share with a 95% bootstrap over puzzles); trains and samples nothing.

    PYTHONPATH=src python -m phase4.overview_plot        # results/phase4/overview_hard_test.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

R = Path("results/phase4")
# (label, colour, table at constant lr 3e-4, table after 4,000 more steps at lr 1e-5)
MODELS = (
    ("stored solutions (supervised)", "#0072B2", "sft/sft_valid_test.csv", "4.9e/sft_lr1e-5_valid_test.csv"),
    ("printed digits only (givens)", "#D55E00", "4.8e/givens_long_valid_test.csv", "4.9d/givens_lr1e-5_valid_test.csv"),
    ("own checker-valid grids (hindsight)", "#009E73", "4.8d/hindsight_long_valid_test.csv", "4.9d/hindsight_lr1e-5_valid_test.csv"),
)
STEPS = (4, 16, 81)


def hard(path: str) -> pd.DataFrame:
    d = pd.read_csv(R / path)
    return d[d.band == "hard"].set_index("sampler_steps").loc[list(STEPS)]


def main():
    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=110)
    for i, (label, colour, before, after) in enumerate(MODELS):
        for j, (path, filled) in enumerate(((before, False), (after, True))):
            d = hard(path)
            x = [k + (i - 1) * 0.24 + (j - 0.5) * 0.09 for k in range(len(STEPS))]
            ax.errorbar(x, d.valid, yerr=[d.valid - d.lo, d.hi - d.valid], fmt="o", ms=6.5, color=colour, lw=1.2, capsize=0,
                        mfc=colour if filled else "white", mew=1.6,
                        label=f"{label}, {'then lr 1e-5' if filled else 'lr 3e-4'}")
    ax.set_xticks(range(len(STEPS)), [f"{s} sampler steps" for s in STEPS])
    ax.set_ylabel("hard test puzzles solved (share of 667)")
    ax.set_ylim(0.35, 0.95)
    ax.grid(axis="y", lw=0.5, alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Phase 4: what the training signal buys, before and after a learning-rate drop", fontsize=10, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.tight_layout()
    out = R / "overview_hard_test.png"
    fig.savefig(out)
    print(out, out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
