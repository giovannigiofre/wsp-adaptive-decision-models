"""A stated-confidence token for the toy Sudoku dLLM, trained with a Brier-shaped reward (phase 3, track C).

Idea from RLCR (arXiv:2507.16806): the model states a confidence q and is rewarded with
r' = r - (q - r)^2, where r is correctness. The Brier penalty is a proper scoring rule, so the expected
reward is maximised by q = P(correct). This is a toy adaptation, not a reproduction of that paper.

Toy version: a 17th position with a learned query vector reads the 16 grid positions through the frozen
architecture of the denoiser (shared weights) and outputs a distribution over K confidence buckets.
Grid positions cannot attend to it (attention mask), so the grid predictions are exactly those of the
underlying denoiser and any change in solve rate comes from training, not from the extra position.

  place = last   q is read from the completed grid:   "is the grid I just wrote valid?"
  place = first  q is read from the puzzle alone:      "will I solve this puzzle?"

  mode = rl      q is sampled, its log-prob is added to log p_hat, and the RL objective uses r'.
  mode = probe   the grid is trained by RL on the plain reward r; the confidence head minimises the
                 expected Brier loss sum_k p_k (q_k - r)^2 directly. Its gradient is the exact expectation
                 of the policy gradient of the Brier reward, so this is the no-sampling-noise reference.

PYTHONPATH=src python -m phase3.conf_token train --mode rl --place last --seed 0
PYTHONPATH=src python -m phase3.conf_token baseline --ckpt weak=~/dllm-cache/phase2/sft/sft_weak.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from phase2.data import PuzzleSet, batch_validity, build_datasets, digits_to_tokens
from phase2.evaluate import auroc, calibration_data, ece, run_sampler
from phase2.model import build_model
from phase2.sampler import sample
from phase2.utils import CACHE as PHASE2_CACHE
from phase2.utils import apply_overrides, env_record, get_device, load_cfg, seed_all

from .objectives import WEIGHTED, weighted_loss
from .rl import CACHE, git_commit, loglik, objective_params, one_step_logprobs, sha256

OUT = Path("results/phase3")


class ConfHead(nn.Module):
    """Denoiser + a confidence query at position 17. forward(grid digits (B,16)) -> bucket logits (B,K)."""

    def __init__(self, base: nn.Module, n_buckets: int = 4):
        super().__init__()
        self.base = base
        d = base.head.in_features
        self.query = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.query, std=0.02)
        self.head = nn.Linear(d, n_buckets)
        nn.init.zeros_(self.head.weight)     # starts uniform over the buckets
        nn.init.zeros_(self.head.bias)
        mask = torch.zeros(17, 17, dtype=torch.bool)
        mask[:16, 16] = True                 # True = may not attend: grid positions never see the query
        self.register_buffer("attn_mask", mask, persistent=False)

    def hidden(self, grid: torch.Tensor) -> torch.Tensor:
        b = self.base
        h = b.tok(digits_to_tokens(grid)) + b.pos
        if b.structured:
            h = h + b.row(b.rcb[0]) + b.col(b.rcb[1]) + b.box(b.rcb[2])
        h = torch.cat([h, self.query.expand(len(grid), -1, -1)], 1)
        return b.norm(b.enc(h, mask=self.attn_mask))

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        return self.head(self.hidden(grid)[:, 16]).float()

    def new_parameters(self):
        return [self.query, *self.head.parameters()]


def brier_reward(r: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return r - (q - r) ** 2


def expected_brier(logits: torch.Tensor, r: torch.Tensor, buckets: torch.Tensor) -> torch.Tensor:
    """mean over samples of sum_k p_k (q_k - r)^2."""
    return (logits.softmax(-1) * (buckets[None] - r[:, None]) ** 2).sum(1).mean()


def calib_metrics(q: np.ndarray, solved: np.ndarray, prefix: str) -> dict[str, float]:
    return {f"{prefix}_brier": float(((q - solved) ** 2).mean()), f"{prefix}_ece": ece(q, solved),
            f"{prefix}_auroc": auroc(q, solved), f"{prefix}_mean": float(q.mean())}


def implicit_confidence(out: dict, ps: PuzzleSet) -> np.ndarray:
    """Puzzle-level mean commit confidence, as in phase 2.4."""
    return calibration_data(out, ps)[2]


@torch.no_grad()
def evaluate(model, conf: ConfHead | None, ps: PuzzleSet, S: int, place: str, buckets: torch.Tensor):
    """Greedy grid (confidence strategy, S steps), then the stated confidence. Returns metrics and arrays."""
    out = run_sampler(model, ps, S, "confidence")
    solved = out["solved"].float().numpy()
    arrays = {"solved": solved, "implicit": implicit_confidence(out, ps)}
    m = {"solve": float(solved.mean()), **calib_metrics(arrays["implicit"], solved, "implicit")}
    if conf is not None:
        ctx = out["grid"].to(ps.puzzles.device) if place == "last" else ps.puzzles
        p = torch.cat([conf(ctx[i:i + 2048]).softmax(-1) for i in range(0, len(ps), 2048)])
        arrays["q_expected"] = (p * buckets).sum(1).cpu().numpy()
        arrays["q_stated"] = buckets[p.argmax(1)].cpu().numpy()
        m.update(calib_metrics(arrays["q_expected"], solved, "q_expected"))
        m.update(calib_metrics(arrays["q_stated"], solved, "q_stated"))
    return m, arrays


def train_cmd(a):
    cfg = apply_overrides(load_cfg(a.config), a.set)
    base = load_cfg(cfg["base_config"])
    rc, cc = cfg["rl"], cfg["conf_token"]
    params = objective_params(rc)
    steps, B, G, S = rc["max_rl_steps"], rc["batch_puzzles"], rc["group_size"], rc["rollout_steps"]
    dev = get_device(a.device or base["runtime"]["device"])
    name = f"conf_{a.mode}_{a.place}_seed{a.seed}{a.tag}"
    out, ck = OUT / a.exp, CACHE / a.exp / "ckpt"
    out.mkdir(parents=True, exist_ok=True)
    ck.mkdir(parents=True, exist_ok=True)
    if (out / f"{name}.json").exists() and not a.force:
        print(f"skip {name}: already finished", flush=True)
        return
    ds = {k: v.to(dev) for k, v in build_datasets(base, base["data"].get("split_seed", 0)).items()}
    train, val, test = ds["train"], ds["val"], ds["test"]
    init = Path(a.init) if a.init else PHASE2_CACHE / cfg["init"]["checkpoint"]
    model = build_model(base).to(dev)
    model.load_state_dict(torch.load(init, map_location=dev))
    model.eval()
    seed_all(a.seed)
    conf = ConfHead(model, len(cc["buckets"])).to(dev)
    buckets = torch.tensor(cc["buckets"], device=dev)
    opt = torch.optim.AdamW([{"params": model.parameters(), "lr": rc["lr"]},
                             {"params": conf.new_parameters(), "lr": cc["head_lr"]}], weight_decay=0.0)
    fields = ["step", "mean_valid", "mean_q", "mean_reward", "frac_skipped", "loss", "val_solve",
              "val_q_expected_brier", "val_q_expected_ece", "val_q_expected_auroc", "val_implicit_brier",
              "elapsed_s"]
    rows = []

    def log_eval(step, acc, t0):
        m, _ = evaluate(model, conf, val, S, a.place, buckets)
        row = {"step": step, "val_solve": m["solve"], "elapsed_s": time.time() - t0,
               **{f"val_{k}": m[k] for k in ("q_expected_brier", "q_expected_ece", "q_expected_auroc",
                                             "implicit_brier")},
               **{k: float(np.mean(v)) for k, v in acc.items() if v}}
        rows.append(row)
        print(f"{name} step={step} val={m['solve']:.3f} brier_q={m['q_expected_brier']:.3f} "
              f"brier_implicit={m['implicit_brier']:.3f} auroc_q={m['q_expected_auroc']:.3f}", flush=True)

    t0 = time.time()
    log_eval(0, {}, t0)
    acc = {k: [] for k in ("mean_valid", "mean_q", "mean_reward", "frac_skipped", "loss")}
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(train), (B,), device=dev)
        P = train.puzzles[idx].repeat_interleave(G, 0)
        with torch.no_grad():
            O = sample(model, P, S, rc["rollout_strategy"], rc["rollout_temperature"]).grid
            ctx = O if a.place == "last" else P
            k = torch.multinomial(conf(ctx).softmax(-1), 1).squeeze(1)
        r = batch_validity(O, P).float()
        q = buckets[k]
        rew = brier_reward(r, q) if a.mode == "rl" else r
        gw = WEIGHTED[rc["objective"]](rew.view(B, G), **params)
        keep = gw.keep
        gw.pos, gw.neg = gw.pos[keep], gw.neg[keep]
        sel = keep.repeat_interleave(G)
        losses = []
        for _ in range(rc["inner_steps"]):
            loss = torch.zeros((), device=dev)
            if keep.any():
                logp, given = one_step_logprobs(model, P[sel], O[sel], rc["prompt_mask_prob"])
                lp = loglik(logp, O[sel], given)
                if a.mode == "rl":
                    lp = lp + conf(ctx[sel]).log_softmax(-1).gather(1, k[sel, None]).squeeze(1)
                loss = loss + weighted_loss(lp.view(-1, G), gw)
            if a.mode == "probe":
                loss = loss + cc["probe_weight"] * expected_brier(conf(ctx), r, buckets)
            if not loss.requires_grad:
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(conf.parameters(), rc["grad_clip"])
            opt.step()
            losses.append(loss.item())
        for key, v in (("mean_valid", r.mean().item()), ("mean_q", q.mean().item()),
                       ("mean_reward", rew.mean().item()), ("frac_skipped", 1 - keep.float().mean().item()),
                       ("loss", float(np.mean(losses)) if losses else float("nan"))):
            acc[key].append(v)
        if step % rc["eval_every"] == 0 or step == steps:
            acc["loss"] = [x for x in acc["loss"] if x == x]
            log_eval(step, acc, t0)
            acc = {k: [] for k in acc}
    pd.DataFrame(rows)[fields].round(4).to_csv(out / f"{name}.csv", index=False)
    torch.save({"model": model.state_dict(), "query": conf.query.data, "head": conf.head.state_dict()},
               ck / f"{name}_final.pt")
    mv, _ = evaluate(model, conf, val, S, a.place, buckets)
    mt, arr = evaluate(model, conf, test, S, a.place, buckets)
    pd.DataFrame(arr).round(4).to_csv(out / f"{name}_test.csv", index=False)
    summ = {"name": name, "mode": a.mode, "place": a.place, "seed": a.seed, "steps": steps,
            "objective": rc["objective"], "objective_params": params, "lr": rc["lr"], "head_lr": cc["head_lr"],
            "buckets": cc["buckets"], "init": str(init), "init_sha256": sha256(init), "overrides": a.set,
            "code_commit": git_commit(), "val": mv, "test": mt, "wall_clock_s": time.time() - t0,
            "env": env_record(dev)}
    json.dump(summ, open(out / f"{name}.json", "w"), indent=2)
    print(json.dumps({"name": name, "test": mt}), flush=True)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fit_platt(conf: np.ndarray, solved: np.ndarray, slope_only: bool = False):
    """Recalibration of a scalar confidence on its logit: sigmoid(a * logit(c) + b), fitted by maximum
    likelihood. slope_only fixes b = 0 (the one-parameter, temperature-like version)."""
    from scipy.optimize import minimize
    x = _logit(conf)

    def nll(theta):
        z = theta[0] * x + (0.0 if slope_only else theta[1])
        return float(np.mean(np.logaddexp(0, z) - solved * z))

    res = minimize(nll, [1.0] if slope_only else [1.0, 0.0], method="Nelder-Mead")
    a, b = res.x[0], (0.0 if slope_only else res.x[1])
    return lambda c: 1 / (1 + np.exp(-(a * _logit(c) + b))), (float(a), float(b))


def baseline_cmd(a):
    """Implicit (commit) confidence of checkpoints without a confidence token, raw and recalibrated
    on val. Recalibration is done on the puzzle-level score because scaling the logits themselves
    would change the unmasking order and therefore the answers."""
    cfg = load_cfg(a.config)
    base = load_cfg(cfg["base_config"])
    S = cfg["rl"]["rollout_steps"]
    dev = get_device(a.device or base["runtime"]["device"])
    ds = {k: v.to(dev) for k, v in build_datasets(base, base["data"].get("split_seed", 0)).items()}
    out = OUT / a.exp
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in a.ckpt:
        label, path = spec.split("=", 1)
        model = build_model(base).to(dev)
        model.load_state_dict(torch.load(Path(path).expanduser(), map_location=dev))
        model.eval()
        _, av = evaluate(model, None, ds["val"], S, "last", None)
        mt, at = evaluate(model, None, ds["test"], S, "last", None)
        row = {"checkpoint": label, "test_solve": mt["solve"],
               **{k.replace("implicit", "raw"): v for k, v in mt.items() if k.startswith("implicit")}}
        for tag, slope_only in (("slope", True), ("platt", False)):
            f, theta = fit_platt(av["implicit"], av["solved"], slope_only)
            at[tag] = f(at["implicit"])
            row.update(calib_metrics(at[tag], at["solved"], tag))
            row[f"{tag}_params"] = str([round(x, 3) for x in theta])
        rows.append(row)
        pd.DataFrame(at).round(4).to_csv(out / f"baseline_{label}_test.csv", index=False)
    df = pd.DataFrame(rows).round(4)
    df.to_csv(out / "baseline_implicit.csv", index=False)
    print(df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["train", "baseline"])
    ap.add_argument("--config", default="configs/phase3.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--device", default=None)
    ap.add_argument("--exp", default="3.3")
    ap.add_argument("--mode", choices=["rl", "probe"], default="rl")
    ap.add_argument("--place", choices=["last", "first"], default="last")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None)
    ap.add_argument("--ckpt", nargs="*", default=[], help="baseline: label=path pairs")
    ap.add_argument("--tag", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    {"train": train_cmd, "baseline": baseline_cmd}[a.what](a)


if __name__ == "__main__":
    main()
