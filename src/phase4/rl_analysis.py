"""Step 4.4: tables, figures and scored predictions for the RL runs on 9x9 (results/phase4/<exp>/).

    PYTHONPATH=src python -m phase4.rl_analysis compact 4.4    # local per-step logs -> curves.csv, runs.csv
    PYTHONPATH=src python -m phase4.rl_analysis evaluate 4.4   # final.pt of every run on the test set, per puzzle
    PYTHONPATH=src python -m phase4.rl_analysis report 4.4     # tables, figures, predictions (from committed files)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from phase2.evaluate import run_sampler
from phase2.model import build_model
from phase2.utils import get_device, seed_all
from phase3 import analysis as A3

from .data import load_sets
from .generate import cache_dir
from .sampler_eval import paired_ci

OUT = Path("results/phase4")
C = A3.C
WD1, POS, FLOOR, KL = "wd1 (τ=1)", "positive-only (w⁻=0)", "wd1, floored", "wd1 + KL to start"
FILT, SELF, GRPO = "filtered imitation", "self-training, SFT loss", "d1-style GRPO"
CONTRAST = [(WD1, C[0]), (FLOOR, C[2]), (KL, C[3]), (GRPO, C[4])]      # raise valid, lower invalid rollouts
PUSH_UP = [(POS, C[1]), (FILT, C[5]), (SELF, "#7b4bb3")]   # not C[4]: GRPO has it and both are in one panel later                    # only raise (valid) rollouts
ORDER = [WD1, FLOOR, KL, GRPO, POS, FILT, SELF]


def method(row) -> str:
    if row.objective == "wd1":
        if row.beta == 0:
            return POS
        if row.ref_kl > 0:
            return KL
        return FLOOR if pd.notna(row.get("floor")) else WD1
    return {"filtered": FILT, "self_diffusion": SELF, "grpo": GRPO}[row.objective]


def _cfg(config: str):
    rl = yaml.safe_load(open(config))
    base = yaml.safe_load(open(rl["base_config"]))
    base["model"].update(base["sizes"][rl["size"]])
    return rl, base


def compact(exp: str, config: str):
    _, base = _cfg(config)
    A3.compact(exp, cache_dir(base) / exp / "logs", OUT / exp)


def evaluate(exp: str, config: str, budgets=(4, 16)):
    """Greedy `confidence` decoding of the start model and of every run's final.pt on the whole test set
    (all three bands), one 0/1 column per (run, sampler steps)."""
    rl, base = _cfg(config)
    dev = get_device(base["runtime"]["device"])
    root = cache_dir(base)
    sets, bands, names = load_sets(base, dev, splits=("test",))
    ps = sets["test"]
    model = build_model(base).to(dev)
    cols = {"puzzle": np.arange(len(ps)), "band": np.array(names)[bands["test"].numpy()]}
    ckpts = [("start", root / rl["init"]["checkpoint"])] + [
        (p.stem, root / exp / "ckpt" / p.stem / "final.pt") for p in sorted((root / exp / "logs").glob("*.json"))]
    for name, ck in ckpts:
        model.load_state_dict(torch.load(ck, map_location=dev))
        model.eval()
        for S in budgets:
            seed_all(0)
            cols[f"{name}|{S}"] = run_sampler(model, ps, S, "confidence", 0.0, batch=1024)["solved"].numpy().astype(int)
        print(name, {S: round(cols[f"{name}|{S}"][cols["band"] == rl["band"]].mean(), 3) for S in budgets}, flush=True)
    (OUT / exp).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cols).to_csv(OUT / exp / "test_solved.csv", index=False)
    print("wrote", OUT / exp / "test_solved.csv")


def _fmt_ci(d: np.ndarray) -> tuple[float, float, float]:
    lo, hi = paired_ci(d)
    return float(d.mean()), lo, hi


def load(exp: str):
    c, r = pd.read_csv(OUT / exp / "curves.csv"), pd.read_csv(OUT / exp / "runs.csv")
    for d in (c, r):
        d["tag"] = d["tag"].astype("string").fillna("")
        d["method"] = d.apply(method, axis=1)
    return c, r


def report(exp: str, config: str, steps: int = 500):
    import matplotlib.pyplot as plt
    rl, _ = _cfg(config)
    band, S, S_ref = rl["band"], rl["rl"]["rollout_steps"], rl["rl"]["eval_steps_ref"]
    out = OUT / exp
    c, r = load(exp)
    horizon_c, c, r = c[c.tag != ""], c[c.tag == ""], r[r.tag == ""]
    t = pd.read_csv(out / "test_solved.csv")
    start = c[c.step == 0].groupby("step")[["val_solve", "val_solve_ref"]].mean().iloc[0]

    # --- per-method table: seeds, paired differences on the test puzzles of the trained band ------------------
    hard = t[t.band == band]
    per_puzzle = {}     # method -> {S: per-puzzle solve share over seeds}
    rows = []
    for m in ORDER:
        d = r[r.method == m]
        if d.empty:
            continue
        per_puzzle[m] = {s: hard[[f"{n}|{s}" for n in d.run]].mean(axis=1).to_numpy() for s in (S, S_ref)}
        row = {"method": m, "n_seeds": len(d),
               "val_mean": d.final_val.mean(), "val_std": d.final_val.std(),
               "test_mean": d.final_test.mean(), "test_std": d.final_test.std(),
               "test_ref_mean": d.final_test_ref.mean(), "test_ref_std": d.final_test_ref.std(),
               "min_val": d.min_val.min(), "seeds_above_start": int((d.final_val > start.val_solve).sum())}
        for s, tag in ((S, "test"), (S_ref, "test_ref")):
            row[f"{tag}_minus_start"], row[f"{tag}_minus_start_lo"], row[f"{tag}_minus_start_hi"] = _fmt_ci(
                per_puzzle[m][s] - hard[f"start|{s}"].to_numpy())
        for b in sorted(set(t.band) - {band}):                   # bands not trained on: was anything broken?
            tb = t[t.band == b]
            row[f"{b}_test_ref"] = tb[[f"{n}|{S_ref}" for n in d.run]].mean().mean()
            row[f"{b}_test_ref_start"] = tb[f"start|{S_ref}"].mean()
        early, late = c[(c.method == m) & (c.step > 0) & (c.step <= 50)], c[(c.method == m) & (c.step > steps - 50)]
        row.update(w_pos_on_invalid_first50=early.w_pos_on_invalid.mean(), frac_skipped=c[c.method == m].frac_skipped.mean(),
                   valid_rollouts_first50=early.mean_valid.mean(), valid_rollouts_last50=late.mean_valid.mean(),
                   logp_valid_first50=early.logp_valid.mean(), logp_valid_last50=late.logp_valid.mean(),
                   logp_invalid_first50=early.logp_invalid.mean(), logp_invalid_last50=late.logp_invalid.mean(),
                   wall_clock_min=d.wall_clock_s.mean() / 60)
        rows.append(row)
    summ = pd.DataFrame(rows)
    summ.round(4).to_csv(out / "summary.csv", index=False)
    r[["method", "seed", "final_val", "final_test", "final_val_ref", "final_test_ref", "final_test_s1", "min_val",
       "min_val_step", "best_val", "best_step", "wall_clock_s"]].sort_values(["method", "seed"]).round(4).to_csv(
        out / "per_seed.csv", index=False)

    # --- paired differences between methods (test puzzles of the trained band, seeds pooled) ------------------
    pairs = [(FLOOR, WD1), (KL, WD1), (SELF, WD1), (FILT, WD1), (GRPO, WD1), (POS, WD1), (SELF, FILT)]
    prow = []
    for a, b in pairs:
        if a in per_puzzle and b in per_puzzle:
            for s in (S, S_ref):
                mean, lo, hi = _fmt_ci(per_puzzle[a][s] - per_puzzle[b][s])
                prow.append({"a": a, "b": b, "sampler_steps": s, "a_minus_b": mean, "lo": lo, "hi": hi})
    paired = pd.DataFrame(prow)
    paired.round(4).to_csv(out / "paired.csv", index=False)

    # --- predictions written in experiments/4.4/README.md before the runs --------------------------------------
    g = summ.set_index("method")
    pd_ = paired.set_index(["a", "b", "sampler_steps"])
    best = g.test_mean.idxmax()
    pos = c[c.method == POS]
    pos_drop = pos[pos.step <= 150].groupby("seed").val_solve.min() - start.val_solve
    excl = lambda a: bool(pd_.loc[(a, WD1, S), "lo"] > 0)      # noqa: E731
    pred = {
        "start": {"val_solve": float(start.val_solve), "val_solve_ref": float(start.val_solve_ref),
                  "test_solve": float(hard[f"start|{S}"].mean()), "test_solve_ref": float(hard[f"start|{S_ref}"].mean())},
        "1_wd1_slow_and_logp_valid_falls": {
            "seeds_above_start": int(g.loc[WD1, "seeds_above_start"]), "below_best_arm": bool(best != WD1),
            "logp_valid_first50": float(g.loc[WD1, "logp_valid_first50"]),
            "logp_valid_last50": float(g.loc[WD1, "logp_valid_last50"]),
            "held": bool(g.loc[WD1, "seeds_above_start"] == g.loc[WD1, "n_seeds"] and best != WD1
                         and g.loc[WD1, "logp_valid_last50"] < g.loc[WD1, "logp_valid_first50"])},
        "2_bounding_makes_wd1_faster": {
            "floored_minus_wd1": [float(x) for x in pd_.loc[(FLOOR, WD1, S)]],
            "kl_minus_wd1": [float(x) for x in pd_.loc[(KL, WD1, S)]], "held": excl(FLOOR) and excl(KL)},
        "3_positive_only_weight_on_invalid_and_collapse": {
            "w_pos_on_invalid_first50": float(g.loc[POS, "w_pos_on_invalid_first50"]),
            "largest_val_drop_first150_per_seed": [float(x) for x in pos_drop],
            "held": bool(0.3 <= g.loc[POS, "w_pos_on_invalid_first50"] <= 0.6 and (pos_drop < -0.10).sum() >= 2)},
        "4_own_valid_rollouts_best": {"best_by_test_mean": best, "held": best in (SELF, FILT)},
    }
    json.dump(pred, open(out / "predictions.json", "w"), indent=2, ensure_ascii=False)

    # --- figures -------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.7), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    for a, fam, title in ((ax[0], CONTRAST, "Raise valid, lower invalid rollouts"),
                          (ax[1], PUSH_UP, "Only raise (valid) rollouts")):
        for lab, col in fam:
            A3.band(a, c[c.method == lab], "val_solve", col, lab)
        a.axhline(start.val_solve, color="0.35", ls=":", lw=1.2, label="start (plateau model)")
        a.set(title=title, xlabel="RL step", ylabel=f"val solve rate, {band} band (greedy, {S} steps)")
        a.legend(fontsize=7, loc="best")
    lo = min(c.val_solve.min(), start.val_solve) - 0.03
    hi = max(c.val_solve.max(), start.val_solve) + 0.03
    ax[0].set_ylim(lo, hi); ax[1].set_ylim(lo, hi)
    colour = dict(CONTRAST + PUSH_UP)
    order = r.groupby("method").final_test.mean().sort_values().index.tolist()
    for y, m in enumerate(order):
        d = r[r.method == m]
        ax[2].scatter(d.final_test, [y] * len(d), s=22, color=colour[m], alpha=0.75, zorder=3)
        ax[2].plot([d.final_test.mean()] * 2, [y - 0.3, y + 0.3], color="0.15", lw=1.5, zorder=4)
    ax[2].axvline(pred["start"]["test_solve"], color="0.35", ls=":", lw=1.2)
    ax[2].annotate("start", (pred["start"]["test_solve"], len(order) - 0.55), xytext=(3, 0),
                   textcoords="offset points", fontsize=8, color="0.3")
    ax[2].set(yticks=range(len(order)), yticklabels=order, ylim=(-0.6, len(order) - 0.3),
              xlabel=f"test solve rate, {band} band, {S} steps", title="Final test solve rate (dots: seeds, bar: mean)")
    ax[2].grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(out / "objectives.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(1, 4, figsize=(16, 3.5))
    panels = [("val_solve_ref", f"val solve rate with {S_ref} sampler steps", start.val_solve_ref),
              ("mean_valid", "share of valid sampled rollouts (25-step mean)", None),
              ("logp_valid", "one-step log p̂ of valid rollouts", None),
              ("logp_invalid", "one-step log p̂ of invalid rollouts", None)]
    for a, (col, title, ref) in zip(ax, panels):
        for lab in ORDER:
            d = c[(c.method == lab)] if col == "val_solve_ref" else c[(c.method == lab) & (c.step > 0)]
            if d[col].notna().any():
                A3.band(a, d, col, colour[lab], lab)
        if ref is not None:
            a.axhline(ref, color="0.35", ls=":", lw=1.2)
        a.set(title=title, xlabel="RL step")
    ax[0].legend(fontsize=6.5, loc="best")
    fig.tight_layout(); fig.savefig(out / "diagnostics.png", dpi=110); plt.close(fig)

    if len(horizon_c):
        h = horizon_c[["step", "val_solve", "val_solve_ref", "mean_valid", "frac_skipped", "logp_valid"]]
        h.round(4).to_csv(out / "horizon_wd1.csv", index=False)
        print("--- horizon (wd1, 1,000 steps)\n", h[h.step.isin([0, 250, 500, 750, 1000])].round(3).to_string(index=False))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
    print(summ.round(3).T.to_string()); print(paired.round(3).to_string(index=False))
    print(json.dumps(pred, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["compact", "evaluate", "report"])
    ap.add_argument("exp")
    ap.add_argument("--config", default="configs/phase4_rl.yaml")
    a = ap.parse_args()
    {"compact": compact, "evaluate": evaluate, "report": report}[a.what](a.exp, a.config)


if __name__ == "__main__":
    main()
