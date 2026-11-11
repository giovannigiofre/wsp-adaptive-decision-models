"""Step 4.0 data report: checks the cached 9x9 datasets and writes results/phase4/data_report.md.

    PYTHONPATH=src python -m phase4.analysis data --config configs/phase4.yaml

Training (step 4.1) may start only if every line of the gate table says PASS.

Step 4.1 report (training curve, val solve rate per band x sampler steps for the plateau model):

    PYTHONPATH=src python -m phase4.analysis sft --size medium
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

from .generate import SPLITS, cache_dir, load
from .sudoku9 import BOXES, COLS, GROUPS, ROWS, canonical_digits, logic_solve, one_pass_fill, solve, to_str

G = np.asarray(GROUPS)
DIFF_NAMES = ["naked singles", "naked + hidden singles", "needs more"]


def np_valid(sol: np.ndarray) -> np.ndarray:
    """Vectorised validity, independent of the solver: every unit holds 1..9."""
    return (np.sort(sol[:, G], axis=2) == np.arange(1, 10)).all(axis=(1, 2))


def solver_check(d: dict, idx: np.ndarray) -> tuple[int, int]:
    """(puzzles with exactly one solution, of those equal to the stored one)."""
    unique = same = 0
    for i in idx:
        sols = solve(d["puzzle"][i].tolist(), 2)
        if len(sols) == 1:
            unique += 1
            same += sols[0] == tuple(d["solution"][i].tolist())
    return unique, same


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def data_report(cfg: dict, root: Path, out: Path, train_sample: int = 5000) -> bool:
    t0 = time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {s: load(cfg, s, root) for s in SPLITS}
    stats = json.loads((root / "generation_stats.json").read_text())
    bands = stats["bands"]
    band_rng = cfg["data"]["bands"]
    gate: list[tuple[str, bool, str]] = []
    L: list[str] = []
    rng = np.random.default_rng(0)

    # 1. sizes
    L += ["## 1. Sizes and generation", "", "| split | " + " | ".join(bands) + " | total |", "|---|" + "---|" * (len(bands) + 1)]
    for s in SPLITS:
        c = [int((data[s]["band"] == b).sum()) for b in range(len(bands))]
        L.append(f"| {s} | " + " | ".join(f"{v:,}" for v in c) + f" | {sum(c):,} |")
    att, rea = stats["attempts"], stats["reached"]
    L += ["", f"Generation: {stats['seconds']} s with {stats['workers']} processes, {stats['grids']:,} solved grids drawn. "
          f"Grids dropped as repeats: {stats['dropped_duplicates']}.", "",
          "A removal order is kept only if it hides exactly k cells with the solution still unique. Share of orders kept, by k:", "",
          "| k | " + " | ".join(str(k) for k in sorted(att, key=int)) + " |", "|---|" + "---|" * len(att),
          "| kept | " + " | ".join(f"{rea[k] / max(1, att[k]):.2f}" for k in sorted(att, key=int)) + " |", ""]

    # 2. correctness
    L += ["## 2. Correctness", ""]
    ok_all = True
    for s in SPLITS:
        d = data[s]
        n = len(d["k"])
        valid = int(np_valid(d["solution"]).sum())
        consistent = int(((d["puzzle"] == 0) | (d["puzzle"] == d["solution"])).all(axis=1).sum())
        hidden_ok = int(((d["puzzle"] == 0).sum(axis=1) == d["k"]).sum())
        in_band = int(sum(band_rng[bands[b]][0] <= k <= band_rng[bands[b]][1] for k, b in zip(d["k"], d["band"])))
        idx = np.arange(n) if s != "train" else rng.choice(n, size=min(train_sample, n), replace=False)
        unique, same = solver_check(d, idx)
        L.append(f"- **{s}** ({n:,} puzzles): stored solution valid {valid:,}/{n:,}; givens agree with it {consistent:,}/{n:,}; "
                 f"hidden cells equal k {hidden_ok:,}/{n:,}; k inside its band {in_band:,}/{n:,}; "
                 f"solver finds exactly one solution on {unique:,}/{len(idx):,} checked, equal to the stored one on {same:,}/{len(idx):,}"
                 + (" (random sample)" if s == "train" else " (all)") + ".")
        ok_all &= valid == consistent == hidden_ok == in_band == n and unique == same == len(idx)
    gate.append(("Every stored solution is valid, agrees with the givens, and is the only solution "
                 f"(all val and test, {train_sample:,} train)", ok_all, ""))
    L += ["", "The validity check here is a separate numpy implementation, not the solver. The solver itself is tested on published "
          "puzzles (`tests/test_sudoku9.py`): Norvig's grid1 with its known solution, a 17-given puzzle and Inkala's 2012 puzzle "
          "(one solution each), and a grid with a removed rectangle (exactly two solutions).", ""]

    # 3. leakage
    L += ["## 3. Leakage and duplicates", ""]
    sol_sets = {s: {r.tobytes() for r in data[s]["solution"]} for s in SPLITS}
    can_sets = {s: {bytes(canonical_digits(r.tolist())) for r in data[s]["solution"]} for s in SPLITS}
    puz_sets = {s: {r.tobytes() for r in data[s]["puzzle"]} for s in SPLITS}
    leak_ok = True
    for s in SPLITS:
        n = len(data[s]["k"])
        L.append(f"- {s}: {len(sol_sets[s]):,} distinct solutions, {len(can_sets[s]):,} distinct after digit relabelling, "
                 f"{len(puz_sets[s]):,} distinct puzzles, of {n:,}.")
        leak_ok &= len(sol_sets[s]) == len(can_sets[s]) == len(puz_sets[s]) == n
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        i1, i2 = len(sol_sets[a] & sol_sets[b]), len(can_sets[a] & can_sets[b])
        L.append(f"- {a} ∩ {b}: {i1} shared solutions, {i2} after digit relabelling.")
        leak_ok &= i1 == i2 == 0
    gate.append(("No solution shared across splits, also after digit relabelling; no duplicates inside a split", leak_ok, ""))
    L += ["", "Not checked: equivalence under the other Sudoku symmetries (row/column permutations inside a band, band and stack "
          "permutations, transposition). Two such grids look different cell by cell but share structure.", ""]

    # 4. digit balance
    sol = data["train"]["solution"]
    n = len(sol)
    freq = np.stack([(sol == d).mean(axis=0) for d in range(1, 10)])  # (9, 81)
    z = (freq - 1 / 9) / np.sqrt((1 / 9) * (8 / 9) / n)
    sim = np.abs(np.random.default_rng(1).standard_normal((2000, 729))).max(axis=1)
    zmax = float(np.abs(z).max())
    balance_ok = zmax < float(np.quantile(sim, 0.999))
    L += ["## 4. Digit balance (train solutions)", "",
          f"Frequency of each digit in each of the 81 positions: min {freq.min():.4f}, max {freq.max():.4f} (uniform: {1 / 9:.4f}). "
          f"Largest deviation is {zmax:.2f} standard errors; for 729 independent uniform frequencies the largest would be "
          f"{np.median(sim):.2f} at the median and {np.quantile(sim, 0.999):.2f} at the 99.9th percentile "
          "(rough reference: the 729 frequencies are not independent).", ""]
    gate.append(("Digit frequency per position consistent with 1/9", balance_ok, f"largest deviation {zmax:.2f} standard errors"))

    # 5. where the hidden cells are
    L += ["## 5. Hidden cells and givens (train)", "",
          "| band | mean hidden | hidden share per position, min–max | max \\|z\\| | givens per row, mean (min–max) | per column | per box | puzzles with an empty row, column or box |",
          "|---|---|---|---|---|---|---|---|"]
    pos_ok = True
    d = data["train"]
    for b, name in enumerate(bands):
        m = d["band"] == b
        hid = d["puzzle"][m] == 0
        share = hid.mean(axis=0)
        p = hid.mean()
        zz = float(np.abs((share - p) / np.sqrt(p * (1 - p) / m.sum())).max())
        pos_ok &= zz < 4.5
        cells = []
        empty = np.zeros(int(m.sum()), dtype=bool)
        for units in (ROWS, COLS, BOXES):
            g = (~hid)[:, np.asarray(units)].sum(axis=2)
            cells.append(f"{g.mean():.2f} ({g.min()}–{g.max()})")
            empty |= (g == 0).any(axis=1)
        L.append(f"| {name} | {hid.sum(axis=1).mean():.1f} | {share.min():.3f}–{share.max():.3f} | {zz:.2f} | "
                 + " | ".join(cells) + f" | {pct(empty.mean())} |")
    gate.append(("Hidden cells spread evenly over the 81 positions in every band", pos_ok, "largest deviation below 4.5 standard errors in every band"))
    L += ["", "z compares each position's hidden share with the band average; with 81 positions a largest value near 3 is expected by chance. "
          "Positions are not exactly exchangeable, because the uniqueness filter depends on the grid.", ""]

    # 6. difficulty and baselines (test split: all puzzles; train: stored difficulty)
    L += ["## 6. Difficulty and rule baselines", "",
          "Difficulty class by logic: solved by naked singles alone (a cell with one candidate, repeated until stuck); solved once "
          "hidden singles are added (a digit with one possible cell in a unit); solved by neither.", "",
          "| split | band | puzzles | naked singles | naked + hidden singles | needs more |", "|---|---|---|---|---|---|"]
    share = {}
    for s in ("train", "test"):
        for b, name in enumerate(bands):
            m = data[s]["band"] == b
            sh = [float((data[s]["difficulty"][m] == c).mean()) for c in range(3)]
            share[(s, name)] = sh
            L.append(f"| {s} | {name} | {int(m.sum()):,} | " + " | ".join(pct(v) for v in sh) + " |")
    d = data["test"]
    recheck = all(
        (0 if 0 not in logic_solve(p.tolist(), False)[0] else 1 if 0 not in logic_solve(p.tolist(), True)[0] else 2) == c
        for p, c in zip(d["puzzle"], d["difficulty"]))
    L += ["", f"Stored classes recomputed on the test split: {'all equal' if recheck else 'MISMATCH'}.", "",
          "Baselines on the test split, with no learning:", "",
          "| band | one pass: puzzles solved | one pass: hidden cells filled | of the filled, correct | repeat until stuck (naked singles): puzzles solved | with hidden singles too |",
          "|---|---|---|---|---|---|"]
    base = {}
    for b, name in enumerate(bands):
        m = np.where(d["band"] == b)[0]
        solved = filled = correct = hidden = 0
        for i in m:
            p, s_ = d["puzzle"][i].tolist(), d["solution"][i].tolist()
            g = one_pass_fill(p)
            solved += g == s_
            for a, x, y in zip(p, g, s_):
                if a == 0:
                    hidden += 1
                    filled += x != 0
                    correct += x == y
        sh = share[("test", name)]
        base[name] = (solved / len(m), sh[0], sh[0] + sh[1])
        L.append(f"| {name} | {pct(solved / len(m))} | {pct(filled / hidden)} | {pct(correct / max(1, filled))} | {pct(sh[0])} | {pct(sh[0] + sh[1])} |")
    differ = all(base[bands[i]][1] > base[bands[i + 1]][1] + 0.1 for i in range(len(bands) - 1))
    hard_more = share[("test", bands[-1])][2]
    gate.append(("Bands differ in difficulty (share solved by naked singles drops by more than 10 points per band)", differ, ""))
    gate.append(("The hard band contains puzzles that singles do not solve", hard_more > 0.1 and recheck, f"{pct(hard_more)} of test"))
    gate.append(("\"Repeat until stuck\" does not solve everything in every band", any(v[1] < 0.9 for v in base.values()),
                 "; ".join(f"{k} {pct(v[1])}" for k, v in base.items())))

    L += ["", "By number of hidden cells (train):", "", "| k | puzzles | naked singles | naked + hidden singles | needs more |", "|---|---|---|---|---|"]
    d = data["train"]
    ks = sorted(set(d["k"].tolist()))
    by_k = np.array([[float((d["difficulty"][d["k"] == k] == c).mean()) for c in range(3)] for k in ks])
    for k, row in zip(ks, by_k):
        L.append(f"| {k} | {int((d['k'] == k).sum()):,} | " + " | ".join(pct(v) for v in row) + " |")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=110)
        for c, (col, mk) in enumerate((("#2a78d6", "o"), ("#e07b1a", "s"), ("#7a3fb8", "^"))):
            ax.plot(ks, 100 * by_k[:, c], color=col, marker=mk, ms=4, lw=1.8, label=DIFF_NAMES[c])
        for name, (lo, hi) in band_rng.items():
            ax.axvline(hi + 0.5, color="#bbbbbb", lw=0.8, ls=":")
            ax.text((lo + hi) / 2, 103, name, ha="center", va="bottom", fontsize=9, color="#555555")
        ax.set_xlabel("hidden cells (k)"); ax.set_ylabel("share of train puzzles (%)"); ax.set_ylim(0, 112)
        ax.set_yticks(range(0, 101, 25)); ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#eeeeee", lw=0.8)
        ax.legend(title="solved by", frameon=False, fontsize=8, title_fontsize=8, loc="center left")
        fig.tight_layout(); fig.savefig(out.parent / "data_difficulty_by_k.png"); plt.close(fig)
        L += ["", "![difficulty by k](data_difficulty_by_k.png)"]
    except Exception as e:  # the report does not depend on the plot
        L += ["", f"(plot not written: {e})"]

    # 7. examples
    L += ["", "## 7. Ten test puzzles per band", "", "One line per puzzle (row-major, `.` = hidden), with k and difficulty class. The first of each band is also drawn.", ""]
    d = data["test"]
    for b, name in enumerate(bands):
        m = np.where(d["band"] == b)[0][:10]
        L += [f"**{name}**", "", "```", to_str(d["puzzle"][m[0]].tolist()), ""]
        for i in m:
            L.append("".join("." if v == 0 else str(v) for v in d["puzzle"][i]) + f"  k={d['k'][i]}  {DIFF_NAMES[d['difficulty'][i]]}")
        L += ["```", ""]

    passed = all(ok for _, ok, _ in gate)
    head = ["# Phase 4, step 4.0: data report", "",
            f"Built by `python -m phase4.analysis data` from the cache in `{cfg['data']['cache_dir']}` "
            f"(generator seed {cfg['data']['seed']}, scale {stats.get('scale', 1.0)}). Report time: {time.time() - t0:.0f} s.", "",
            "## Gate", "", "| check | result | note |", "|---|---|---|"]
    head += [f"| {name} | {'PASS' if ok else '**FAIL**'} | {note} |" for name, ok, note in gate]
    head += ["", f"**Gate: {'PASS' if passed else 'FAIL'}.** " + ("Training may start." if passed else "Do not train on this data."), "",
             "Limit of this dataset: puzzles come from hiding random cells of random grids. They are easier than curated hard "
             "collections, and the difficulty classes use only two solving techniques, so \"needs more\" is a mixed bag.", ""]
    out.write_text("\n".join(head + L) + "\n")
    return passed


BAND_COLORS = {"all": "#333333", "easy": "#2a78b5", "medium": "#d9822b", "hard": "#8e44ad"}


def sft_report(cfg: dict, name: str, out: Path, budgets=(1, 4, 8, 16, 32, 81), split: str = "val") -> None:
    """Training curve from <name>.csv and solve rate per band x sampler steps (confidence, greedy) for final.pt."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import torch

    from phase2.evaluate import bootstrap_ci, run_sampler
    from phase2.model import build_model
    from phase2.utils import get_device

    from .data import load_sets

    log = pd.read_csv(out / f"{name}.csv")
    done = json.loads((out / f"{name}.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(log.step, log.train_loss, color="#999999", lw=1, label="train (last batch)")
    ax[0].plot(log.step, log.val_loss, color="#333333", lw=2, label="val")
    ax[0].set_yscale("log"); ax[0].set_xlabel("training step"); ax[0].set_ylabel("masked diffusion loss (log scale)")
    ax[0].set_title(f"{done['n_params'] / 1e6:.2f}M model, stop: {done['stop_reason']} at step {done['final_step']:,}", fontsize=10)
    for b in ("easy", "medium", "hard"):
        ax[1].plot(log.step, log[f"val_solve_{b}"], color=BAND_COLORS[b], lw=1.5, label=b)
    ax[1].plot(log.step, log.val_solve, color=BAND_COLORS["all"], lw=2, ls="--", label="all")
    ax[1].set_ylim(0, 1.02); ax[1].set_xlabel("training step"); ax[1].set_ylabel(f"{split} solve rate (confidence, 16 steps)")
    for a in ax:
        a.grid(alpha=0.25); a.legend(frameon=False, fontsize=8)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / f"{name}_training_curve.png", dpi=110); plt.close(fig)

    dev = get_device(cfg["runtime"]["device"])
    sets, bands, names = load_sets(cfg, dev, splits=(split,))
    ps, band = sets[split], bands[split].cpu()
    model = build_model(cfg).to(dev)
    model.load_state_dict(torch.load(Path(done["ckpt_dir"]) / "final.pt", map_location=dev))
    rows = []
    for steps in budgets:
        t0 = time.time()
        solved = run_sampler(model, ps, steps, "confidence", batch=1024)["solved"].numpy().astype(float)
        dt = time.time() - t0
        for label, m in [("all", np.ones(len(solved), bool))] + [(n, (band == b).numpy()) for b, n in enumerate(names)]:
            lo, hi = bootstrap_ci(solved[m])
            rows.append({"split": split, "strategy": "confidence", "steps": steps, "band": label, "n": int(m.sum()),
                         "solve": solved[m].mean(), "lo": lo, "hi": hi, "seconds": dt})
    df = pd.DataFrame(rows)
    df.to_csv(out / f"{name}_solve_by_steps.csv", index=False)
    fig, a = plt.subplots(figsize=(5.6, 3.6))
    for label in ["easy", "medium", "hard", "all"]:
        d = df[df.band == label]
        a.errorbar(d.steps, d.solve, yerr=[d.solve - d.lo, d.hi - d.solve], color=BAND_COLORS[label], marker="o", ms=5,
                   lw=2 if label != "all" else 1.5, ls="--" if label == "all" else "-", capsize=2, label=label)
    a.set_xscale("log"); a.set_xticks(list(budgets)); a.set_xticklabels([str(b) for b in budgets]); a.minorticks_off()
    a.set_ylim(0, 1.02); a.set_xlabel("sampler steps (confidence, greedy)"); a.set_ylabel(f"{split} solve rate (95% bootstrap)")
    a.grid(alpha=0.25); a.legend(frameon=False, fontsize=8, title="band"); a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / f"{name}_solve_by_steps.png", dpi=110); plt.close(fig)
    print(df.pivot(index="steps", columns="band", values="solve").round(3).to_string())
    print(df[df.band == "hard"][["steps", "solve", "lo", "hi", "seconds"]].round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["data", "sft"])
    ap.add_argument("--config", default="configs/phase4.yaml")
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--name", default="sft")
    ap.add_argument("--size", default=None, help="key of `sizes` in the config (sft report)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.what == "sft":
        if a.size:
            cfg["model"].update(cfg["sizes"][a.size])
        sft_report(cfg, a.name, Path(a.out or "results/phase4/sft"))
        return
    root = Path(a.root).expanduser() if a.root else cache_dir(cfg)
    ok = data_report(cfg, root, Path(a.out or "results/phase4/data_report.md"))
    print("gate:", "PASS" if ok else "FAIL", "->", a.out or "results/phase4/data_report.md")

if __name__ == "__main__":
    main()
