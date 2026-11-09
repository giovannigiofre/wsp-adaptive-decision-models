"""Step 1.3: ECE, reliability diagrams, AUROC; answer-level and Sudoku token-level calibration;
early- vs late-commit accuracy. Also `bootstrap_ci` used by 1.2 reporting.

Usage: python -m src.phase1.calibration
Reads results/phase1/tokens/<task>.jsonl (from 1.2), strategy=confidence, full step budget.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .load import REPO, load_config


def bootstrap_ci(x, n_boot=1000, seed=0, stat=np.mean, alpha=0.05):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)]
    return float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def reliability(conf, correct, n_bins=10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        out.append((edges[b], edges[b + 1], int(m.sum()), conf[m].mean() if m.any() else np.nan,
                    correct[m].mean() if m.any() else np.nan))
    return out


def ece(conf, correct, n_bins=10):
    n = len(conf)
    return float(sum(c / n * abs(a - p) for _, _, c, p, a in reliability(conf, correct, n_bins) if c > 0))


def auroc(conf, correct):
    from sklearn.metrics import roc_auc_score
    correct = np.asarray(correct)
    if correct.min() == correct.max():
        return float("nan")  # undefined when only one class is present
    return float(roc_auc_score(correct, conf))


def boot_metric(f, conf, correct, n_boot=1000, seed=0):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(conf), len(conf))
        v = f(conf[i], correct[i])
        if not np.isnan(v):
            vals.append(v)
    if len(vals) < n_boot / 2:
        return float("nan"), float("nan")
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def plot_reliability(ax, conf, correct, n_bins, title):
    rel = reliability(conf, correct, n_bins)
    centers = [(lo + hi) / 2 for lo, hi, *_ in rel]
    accs = [a if c > 0 else 0 for *_, c, _, a in rel]
    counts = [c for _, _, c, _, _ in rel]
    ax.bar(centers, accs, width=1 / n_bins * 0.9, color="#4C72B0", alpha=0.8, label="accuracy in bin")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for x, a, c in zip(centers, accs, counts):
        if c:
            ax.text(x, a + 0.02, str(c), ha="center", fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("confidence")
    ax.set_ylabel("fraction correct")
    ax.set_title(title, fontsize=9)


def main():
    cfg = load_config()
    nb, nboot = cfg["calibration"]["n_bins"], cfg["calibration"]["bootstrap_samples"]
    out = REPO / "results/phase1/calibration"
    out.mkdir(parents=True, exist_ok=True)
    metrics = []
    recs = {}
    for task in ["sudoku", "countdown", "gsm8k"]:
        p = REPO / f"results/phase1/tokens/{task}.jsonl"
        if not p.exists():
            continue
        rs = [json.loads(l) for l in open(p)]
        full = max(r["steps"] for r in rs if r["gen_length"] == rs[0]["gen_length"])
        sel = [r for r in rs if r["strategy"] == "confidence" and r["steps"] == full]
        if sel:  # a partially finished grid may have no full-budget confidence rows yet
            recs[task] = sel

    # ---------- answer-level
    fig, axes = plt.subplots(2, max(1, len(recs)), figsize=(4 * max(1, len(recs)), 7), constrained_layout=True,
                             squeeze=False)
    for j, (task, rs) in enumerate(recs.items()):
        correct = np.array([r["correct"] for r in rs])
        for i, agg in enumerate(["mean", "min"]):
            conf = np.array([(np.mean if agg == "mean" else np.min)(r["commit_conf"][:max(1, r["n_content"])])
                             for r in rs])
            e, a = ece(conf, correct, nb), auroc(conf, correct)
            elo, ehi = boot_metric(lambda c, y: ece(c, y, nb), conf, correct, nboot)
            alo, ahi = boot_metric(auroc, conf, correct, nboot)
            metrics.append({"task": task, "level": "answer", "confidence": f"{agg} commit conf (content tokens)",
                            "n": len(rs), "accuracy": correct.mean(), "mean_confidence": conf.mean(),
                            "ece": e, "ece_lo": elo, "ece_hi": ehi, "auroc": a, "auroc_lo": alo, "auroc_hi": ahi})
            plot_reliability(axes[i, j], conf, correct, nb,
                             f"{task}: {agg} conf, n={len(rs)}, acc={correct.mean():.2f}\nECE={e:.2f} AUROC={a:.2f}")
    fig.savefig(out / "reliability_answer_level.png", dpi=80)
    plt.close(fig)

    # ---------- token-level (Sudoku cells)
    early_late = []
    if "sudoku" in recs:
        cells = [c for r in recs["sudoku"] for c in (r["cells"] or [])]
        if cells:
            conf = np.array([c[2] for c in cells])
            corr = np.array([c[4] for c in cells])
            step = np.array([c[3] for c in cells])
            e, a = ece(conf, corr, nb), auroc(conf, corr)
            elo, ehi = boot_metric(lambda c, y: ece(c, y, nb), conf, corr, nboot)
            alo, ahi = boot_metric(auroc, conf, corr, nboot)
            metrics.append({"task": "sudoku", "level": "token (empty cell)", "confidence": "commit conf of the cell's token",
                            "n": len(cells), "accuracy": corr.mean(), "mean_confidence": conf.mean(),
                            "ece": e, "ece_lo": elo, "ece_hi": ehi, "auroc": a, "auroc_lo": alo, "auroc_hi": ahi})
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
            plot_reliability(axes[0], conf, corr, nb, f"Sudoku cells: n={len(cells)} cells, "
                             f"acc={corr.mean():.2f}\nECE={e:.2f} AUROC={a:.2f}")
            # early vs late: rank of the commit step among the answer's cells
            ranks = []
            for r in recs["sudoku"]:
                cs = r["cells"] or []
                if len(cs) < 2:
                    continue
                st = np.array([c[3] for c in cs], float)
                rk = (st.argsort().argsort()) / (len(cs) - 1)
                ranks += [(q, c[4]) for q, c in zip(rk, cs)]
            if ranks:
                rk = np.array(ranks)
                bins = [(0, 1 / 3, "first third"), (1 / 3, 2 / 3, "middle third"), (2 / 3, 1.01, "last third")]
                accs, cis, labels = [], [], []
                for lo, hi, lab in bins:
                    m = (rk[:, 0] >= lo) & (rk[:, 0] < hi)
                    accs.append(rk[m, 1].mean())
                    cis.append(bootstrap_ci(rk[m, 1], nboot))
                    labels.append(f"{lab}\n(n={m.sum()})")
                    early_late.append({"task": "sudoku", "unit": "empty cell", "commit_order": lab, "n": int(m.sum()),
                                       "accuracy": accs[-1], "lo": cis[-1][0], "hi": cis[-1][1]})
                yerr = np.array([[a - c[0], c[1] - a] for a, c in zip(accs, cis)]).T
                axes[1].bar(range(3), accs, yerr=yerr, color="#55A868", capsize=4)
                axes[1].set_xticks(range(3), labels)
                axes[1].set_ylabel("cell accuracy")
                axes[1].set_ylim(0, 1)
                axes[1].set_title("Cells by commit order within the answer\n(95% bootstrap CI)", fontsize=9)
            fig.savefig(out / "sudoku_token_level.png", dpi=80)
            plt.close(fig)
            sudoku_cells_by_digit(recs["sudoku"], out)

    # ---------- early vs late, answer-level for all tasks: commit time of the answer span
    for task, rs in recs.items():
        pts = []
        for r in rs:
            sp = r.get("ans_tok_span")
            if not sp:
                continue
            total = max(r["commit_step"]) + 1
            t = np.mean(r["commit_step"][sp[0]:sp[1]]) / total
            pts.append((t, r["correct"]))
        if len(pts) >= 4:
            pts = np.array(pts)
            med = np.median(pts[:, 0])
            for lab, m in [("answer span committed early (<= median)", pts[:, 0] <= med),
                           ("answer span committed late (> median)", pts[:, 0] > med)]:
                lo, hi = bootstrap_ci(pts[m, 1], nboot)
                early_late.append({"task": task, "unit": "answer", "commit_order": lab, "n": int(m.sum()),
                                   "accuracy": pts[m, 1].mean() if m.any() else float("nan"), "lo": lo, "hi": hi})

    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics[0]))
        w.writeheader()
        for m in metrics:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
    if early_late:
        with open(out / "early_vs_late.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(early_late[0]))
            w.writeheader()
            for m in early_late:
                w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})
    for m in metrics:
        print(m)
    for m in early_late:
        print(m)


def sudoku_cells_by_digit(recs, out):
    """Which digit the model wrote into each empty cell (from the extracted answer in the raw CSV),
    with mean commit confidence and accuracy. Explains low/inverted AUROC when the model copies
    the puzzle's '0's with high confidence."""
    raw = {}
    for r in csv.DictReader(open(REPO / "results/phase1/raw/sudoku.csv")):
        raw[(r["id"], r["strategy"], int(r["steps"]))] = r["extracted"] or ""
    by = defaultdict(list)
    for r in recs:
        ext = raw.get((r["id"], r["strategy"], r["steps"]), "")
        for c in r["cells"] or []:
            by[ext[c[0]] if c[0] < len(ext) else "?"].append((c[2], c[4]))
    with open(out / "sudoku_cells_by_digit.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["digit_written", "n_cells", "mean_commit_conf", "cell_accuracy"])
        for d, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
            v = np.array(v, float)
            w.writerow([d, len(v), round(v[:, 0].mean(), 4), round(v[:, 1].mean(), 4)])


if __name__ == "__main__":
    main()
