"""Phase 2 evaluations that read checkpoints and write CSV/JSON to results/phase2/.

PYTHONPATH=src python -m phase2.analysis strategies    # 2.2: strategy x step budget on test
PYTHONPATH=src python -m phase2.analysis likelihood    # 2.3 sanity: one-step vs sequential log-lik
PYTHONPATH=src python -m phase2.analysis calibration   # 2.4: SFT vs RL calibration on test
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import batch_validity, build_datasets
from .evaluate import auroc, bootstrap_ci, calibration_data, ece, run_sampler
from .model import build_model
from .rl_wd1 import one_step_loglik, sequential_loglik
from .sampler import sample
from .utils import CACHE, env_record, get_device, load_cfg, seed_all

OUT = Path("results/phase2")


def load(cfg, path, dev):
    m = build_model(cfg).to(dev)
    m.load_state_dict(torch.load(path, map_location=dev))
    return m.eval()


def strategies(cfg, ds, dev, ckpts: dict[str, str]):
    rows = []
    test = ds["test"]
    for name, path in ckpts.items():
        model = load(cfg, path, dev)
        for strat in cfg["sampler"]["strategies"]:
            for S in cfg["sampler"]["step_budgets"]:
                seeds = [0, 1, 2] if strat == "random" else [0]
                for sd in seeds:
                    seed_all(sd)
                    t0 = time.time()
                    out = run_sampler(model, test, S, strat, cfg["sampler"]["eval_temperature"])
                    solved = out["solved"].numpy().astype(float)
                    lo, hi = bootstrap_ci(solved)
                    rows.append({"ckpt": name, "strategy": strat, "steps": S, "order_seed": sd,
                                 "solve_rate": solved.mean(), "ci_lo": lo, "ci_hi": hi, "n": len(solved),
                                 "n_forward": int(min(S, int(test.k.max()))), "wall_s": time.time() - t0})
                    for k in sorted(set(test.k.tolist())):
                        sel = (test.k == k).cpu().numpy()
                        rows[-1][f"solve_k{k}"] = solved[sel].mean()
        print(name, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "strategies.csv", index=False)
    piv = (df.groupby(["ckpt", "strategy", "steps"]).solve_rate.mean().unstack("steps").round(3))
    piv.to_csv(OUT / "strategies_table.csv")
    print(piv.to_string())


def likelihood(cfg, ds, dev, ckpts: dict[str, str], n_puzzles: int = 128, per: int = 2):
    test = ds["test"]
    rows, summ = [], {}
    for name, path in ckpts.items():
        model = load(cfg, path, dev)
        seed_all(0)
        idx = torch.randperm(len(test))[:n_puzzles].to(dev)
        P = test.puzzles[idx].repeat_interleave(per, 0)
        with torch.no_grad():
            O = sample(model, P, cfg["rl"]["rollout_steps"], "confidence", 1.0).grid
            r = batch_validity(O, P)
            t0 = time.time()
            seq = sequential_loglik(model, P, O)
            t_seq = time.time() - t0
            t0 = time.time()
            one0 = one_step_loglik(model, P, O, 0.0)
            t_one = time.time() - t0
            one15 = one_step_loglik(model, P, O, cfg["rl"]["prompt_mask_prob"])
            one15avg = torch.stack([one_step_loglik(model, P, O, cfg["rl"]["prompt_mask_prob"])
                                    for _ in range(16)]).mean(0)
        d = pd.DataFrame({"ckpt": name, "correct": r.cpu().numpy(), "k": (P == 0).sum(1).cpu().numpy(),
                          "seq": seq.cpu().numpy(), "one_step_p0": one0.cpu().numpy(),
                          "one_step_p015": one15.cpu().numpy(), "one_step_p015_avg16": one15avg.cpu().numpy()})
        rows.append(d)
        s = {"n": len(d), "frac_correct": float(d.correct.mean()), "sec_sequential": t_seq, "sec_one_step": t_one}
        for col in ["one_step_p0", "one_step_p015", "one_step_p015_avg16"]:
            s[f"pearson_{col}"] = float(d[["seq", col]].corr().iloc[0, 1])
            s[f"spearman_{col}"] = float(d[["seq", col]].corr(method="spearman").iloc[0, 1])
            # per-cell normalised (removes the trivial correlation driven by the number of hidden cells)
            s[f"pearson_percell_{col}"] = float(np.corrcoef(d.seq / d.k, d[col] / d.k)[0, 1])
        s["mean_seq_minus_one_step_p0"] = float((d.seq - d.one_step_p0).mean())
        summ[name] = s
    pd.concat(rows).to_csv(OUT / "likelihood_estimators.csv", index=False)
    json.dump(summ, open(OUT / "likelihood_estimators.json", "w"), indent=2)
    print(json.dumps(summ, indent=2))


def calibration(cfg, ds, dev, ckpts: dict[str, str], n_bins: int = 10):
    test = ds["test"]
    S = cfg["rl"]["rollout_steps"]
    rows, bins = [], []
    for name, path in ckpts.items():
        model = load(cfg, path, dev)
        out = run_sampler(model, test, S, "confidence", 0.0)
        cc, co, pc, po = calibration_data(out, test.to("cpu"))
        rows.append({"ckpt": name, "solve_rate": float(po.mean()), "cell_acc": float(co.mean()),
                     "mean_cell_conf": float(cc.mean()), "cell_ece": ece(cc, co, n_bins), "cell_auroc": auroc(cc, co),
                     "mean_puzzle_conf": float(pc.mean()), "puzzle_ece": ece(pc, po, n_bins),
                     "puzzle_auroc": auroc(pc, po), "conf_correct_cells": float(cc[co].mean()),
                     "conf_wrong_cells": float(cc[~co].mean()) if (~co).any() else float("nan"),
                     "n_cells": len(cc), "n_puzzles": len(pc)})
        from .evaluate import reliability_bins
        for level, c, o in (("cell", cc, co), ("puzzle", pc, po)):
            ctr, acc, mc, cnt = reliability_bins(c, o.astype(float), n_bins)
            bins += [{"ckpt": name, "level": level, "bin": float(x), "acc": a_, "mean_conf": m_, "count": int(n_)}
                     for x, a_, m_, n_ in zip(ctr, acc, mc, cnt)]
    pd.DataFrame(rows).to_csv(OUT / "calibration.csv", index=False)
    pd.DataFrame(bins).to_csv(OUT / "calibration_bins.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string())


def rl_ckpts(pattern: str) -> dict[str, str]:
    return {p.parent.name: str(p) for p in sorted((CACHE / "rl").glob(pattern))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["strategies", "likelihood", "calibration"])
    ap.add_argument("--config", default="configs/phase2.yaml")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    dev = get_device(a.device or cfg["runtime"]["device"])
    ds = {k: v.to(dev) for k, v in build_datasets(cfg, cfg["data"].get("split_seed", 0)).items()}
    OUT.mkdir(parents=True, exist_ok=True)
    sft = {"sft_weak": str(CACHE / "sft" / "sft_weak.pt"), "sft_full": str(CACHE / "sft" / "sft_full.pt")}
    t0 = time.time()
    if a.what == "strategies":
        strategies(cfg, ds, dev, sft)
    else:
        best = json.load(open(OUT / "rl" / "best_run.json"))  # written by phase2.plots rl
        rl = {"rl_best": str(CACHE / "rl" / best["name"] / "best.pt")}
        if a.what == "likelihood":
            likelihood(cfg, ds, dev, {"sft_weak": sft["sft_weak"], **rl})
        else:
            all_rl = {f"rl_{k}": v for k, v in rl_ckpts("*/best.pt").items()}
            calibration(cfg, ds, dev, {"sft_weak": sft["sft_weak"], **rl, "sft_full": sft["sft_full"], **all_rl})
    print(json.dumps({"wall_clock_s": time.time() - t0, **env_record(dev)}))


if __name__ == "__main__":
    main()
