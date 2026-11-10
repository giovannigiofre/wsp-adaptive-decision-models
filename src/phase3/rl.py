"""Generic RL loop for the toy Sudoku dLLM with a pluggable objective and reward (phase 3).

Per RL step: B puzzles x G completions (same sampler settings as phase 2), reward from `rewards.py`,
objective from `objectives.py`, one-step mean-field log p_hat, mu inner gradient steps.
For the weighted-likelihood objectives the order of random draws matches phase2.rl_wd1 (puzzle indices,
rollouts, prompt masks), so the default config and `--set rl.objective_params.beta=0` reproduce phase 2's
wd1 and positive-only runs. Data, model and sampler come from the phase 2 config named in `base_config`.

`rl.ref_mix` > 0 samples the rollouts from mixed logits, (1 - ref_mix) * current + ref_mix * frozen start,
which is how the reference policy enters the upstream wd1 code (github.com/xiaohangt/wd1, commit 1a2e254,
wd1/trainers/rev_grpo_ref_pol_trainer.py, `generate`). The loss is unchanged, as upstream.

`rl.likelihood: sequential` replaces the one-step estimate in the weighted objectives by the fixed-order
sequential likelihood (16 forward passes per inner step), which phase 2 used as the reference.
`rl.likelihood: trajectory` uses the chain rule along the rollout's own unmasking path instead (one pass per
sampler step): the log-probability of the values the sampler actually chose, in the states it chose them in.

`rl.ref_kl` > 0 adds, to a weighted-likelihood objective, a penalty towards the frozen start: ref_kl times
the per-cell KL estimate exp(d) - d - 1 with d = log p_ref - log p (the estimator of the d1 GRPO trainer),
summed over the non-given cells of every rollout of a kept group. This is not part of upstream wd1.

Resumable (state.pt every eval), one CSV row per step. With `--exp ID` the per-step logs stay in the
local cache and `python -m phase3.analysis compact ID` writes one row per evaluation to results/phase3/ID/.

PYTHONPATH=src python -m phase3.rl --seed 0
PYTHONPATH=src python -m phase3.rl --seed 0 --set rl.objective_params.beta=0
PYTHONPATH=src python -m phase3.rl --seed 0 --exp 3.2 --set rl.objective=reinforce

Phase 4: a config with `task: sudoku9` (configs/phase4_rl.yaml) switches the data to one difficulty band of the
cached 9x9 sets, the start to the 4.1 plateau model and the cache to ~/dllm-cache/phase4; `rl.eval_steps_ref`
adds a val solve rate at a second sampler budget (column val_solve_ref). Nothing else changes.

PYTHONPATH=src python -m phase3.rl --config configs/phase4_rl.yaml --exp 4.4 --seed 0
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from phase2.data import PuzzleSet, batch_validity, build_datasets, digits_to_tokens
from phase2.evaluate import solve_rate
from phase2.model import build_model
from phase2.sampler import sample
from phase2.train_diffusion import diffusion_loss
from phase2.utils import CACHE as PHASE2_CACHE
from phase2.utils import apply_overrides, env_record, get_device, load_cfg, seed_all

from .objectives import KIND, WEIGHTED, advantages, grpo_loss, wd1_batch, weight_diagnostics, weighted_loss
from .rewards import REWARDS

CACHE = Path(os.environ.get("PHASE3_CACHE", Path.home() / "dllm-cache" / "phase3"))
FIELDS = ["step", "mean_reward", "mean_valid", "frac_skipped", "loss", "grad_norm", "w_entropy", "w_ess",
          "w_pos_on_invalid", "logp_valid", "logp_invalid", "pred_entropy", "clip_frac", "val_solve",
          "val_solve_s1", "elapsed_s"]


def prompt_mask(puzzles: torch.Tensor, p_mask: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Given cells that are additionally masked in the likelihood estimate (d1/wd1 `p_mask_prompt`)."""
    return (puzzles != 0) & (torch.rand(puzzles.shape, device=puzzles.device, generator=generator) < p_mask)


def one_step_logprobs(model, puzzles: torch.Tensor, completions: torch.Tensor, p_mask: float = 0.0,
                      generator: torch.Generator | None = None, extra: torch.Tensor | None = None):
    """Same input and random draw as phase2.rl_wd1.one_step_loglik, but returns the per-cell
    log-probs (N,16,4) and the given mask so diagnostics can reuse the forward pass.
    `extra` is a prompt mask drawn earlier (then no random draw happens here)."""
    given = puzzles != 0
    if extra is None:
        extra = prompt_mask(puzzles, p_mask, generator)
    inp = torch.where(given & ~extra, completions, torch.zeros_like(completions))
    return F.log_softmax(model(digits_to_tokens(inp)).float(), -1), given


def cell_logp(logp: torch.Tensor, completions: torch.Tensor) -> torch.Tensor:
    """(N,16) log p of each completion cell."""
    return logp.gather(-1, (completions - 1).unsqueeze(-1)).squeeze(-1)


def loglik(logp: torch.Tensor, completions: torch.Tensor, given: torch.Tensor) -> torch.Tensor:
    """(N,) one-step log p_hat: sum over non-given cells of log p(completion cell)."""
    return (cell_logp(logp, completions) * (~given)).sum(1)


def sequential_cell_logp(model, puzzles: torch.Tensor, completions: torch.Tensor,
                         extra: torch.Tensor | None = None) -> torch.Tensor:
    """(N,16) per-cell log p under a fixed row-major order: cell i is predicted with the givens (minus
    the prompt mask `extra`) and all earlier non-given cells revealed. 16 forward passes, with gradient.
    Same construction as phase2.rl_wd1.sequential_loglik, which phase 2 used as the 'true' likelihood."""
    given = puzzles != 0
    shown = given if extra is None else given & ~extra
    out = torch.zeros(completions.shape, device=completions.device, dtype=torch.float32)
    for i in range(completions.shape[1]):
        if not (~given[:, i]).any():
            continue
        visible = shown.clone()
        visible[:, :i] |= ~given[:, :i]
        inp = torch.where(visible, completions, torch.zeros_like(completions))
        lp_i = F.log_softmax(model(digits_to_tokens(inp)).float()[:, i], -1)
        out[:, i] = lp_i.gather(-1, (completions[:, i] - 1).unsqueeze(-1)).squeeze(-1)
    return out


def trajectory_cell_logp(model, puzzles: torch.Tensor, completions: torch.Tensor, commit_step: torch.Tensor,
                         extra: torch.Tensor | None = None) -> torch.Tensor:
    """(N,16) per-cell log p along the rollout's own unmasking path: a cell committed at sampler step s is
    predicted with the givens (minus the prompt mask `extra`) and the cells committed before s revealed,
    which is the input the sampler saw when it chose that value. One forward pass per sampler step, with
    gradient. The choice of *which* cells to commit is treated as fixed (no gradient through the order)."""
    given = puzzles != 0
    shown = given if extra is None else given & ~extra
    out = torch.zeros(completions.shape, device=completions.device, dtype=torch.float32)
    for s in range(int(commit_step.max().item()) + 1):
        now = ~given & (commit_step == s)
        if not now.any():
            continue
        visible = shown | (~given & (commit_step >= 0) & (commit_step < s))
        inp = torch.where(visible, completions, torch.zeros_like(completions))
        lp = cell_logp(F.log_softmax(model(digits_to_tokens(inp)).float(), -1), completions)
        out = torch.where(now, lp, out)
    return out


def predictive_entropy(logp: torch.Tensor, given: torch.Tensor) -> torch.Tensor:
    """Mean entropy (nats, max ln 4) of the one-step predictive distribution over non-given cells."""
    ent = -(logp.exp() * logp).sum(-1)
    return (ent * (~given)).sum() / (~given).sum()


class MixedLogits(torch.nn.Module):
    """Rollout policy with a reference model: logits = (1 - mix) * model + mix * reference."""

    def __init__(self, model, ref, mix: float):
        super().__init__()
        self.model, self.ref, self.mix = model, ref, mix

    def forward(self, tokens):
        return (1 - self.mix) * self.model(tokens) + self.mix * self.ref(tokens)


def objective_params(rc: dict) -> dict:
    """Defaults of the chosen objective, updated with `rl.objective_params`."""
    return {**rc["objective_defaults"][rc["objective"]], **rc["objective_params"]}


def run_name(rc: dict, seed: int, tag: str = "") -> str:
    params = "_".join(f"{k}{float(v):g}" for k, v in sorted(objective_params(rc).items()) if v is not None)
    parts = [rc["objective"], params, "" if rc["reward"] == "binary" else rc["reward"],
             f"ref{rc['ref_mix']:g}" if rc.get("ref_mix", 0) else "",
             f"refkl{rc['ref_kl']:g}" if rc.get("ref_kl", 0) else "",
             {"sequential": "seqlik", "trajectory": "trajlik"}.get(rc.get("likelihood", "onestep"), ""),
             f"ord{rc['order_temp']:g}" if rc.get("order_temp", 0) else "",
             "batchw" if rc.get("weights_over", "group") == "batch" else "", f"seed{seed}{tag}"]
    return "_".join(p for p in parts if p)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase3.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. rl.objective_params.beta=0")
    ap.add_argument("--init", default=None, help="checkpoint path; default is init.checkpoint in the config")
    ap.add_argument("--device", default=None)
    ap.add_argument("--exp", default=None, help="experiment id, e.g. 3.1a: per-step logs and checkpoints go "
                    "to the local cache under that id; `phase3.analysis compact` writes the committed summary")
    ap.add_argument("--out", default="results/phase3/rl")
    ap.add_argument("--ckpt-dir", default=str(CACHE / "rl"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--force", action="store_true", help="rerun even if the run's summary JSON exists")
    a = ap.parse_args()
    cfg = apply_overrides(load_cfg(a.config), a.set)
    base = load_cfg(cfg["base_config"])
    p4 = cfg.get("task") == "sudoku9"                              # phase 4: 9x9 data, plateau model, own cache
    cache, fields = CACHE, FIELDS
    if p4:
        from phase4.generate import cache_dir
        base["model"].update(base["sizes"][cfg["size"]])
        cache = cache_dir(base)
    rc = cfg["rl"]
    kind, params, reward_fn = KIND[rc["objective"]], objective_params(rc), REWARDS[rc["reward"]]
    steps, B, G, S = rc["max_rl_steps"], rc["batch_puzzles"], rc["group_size"], rc["rollout_steps"]
    S_ref = rc.get("eval_steps_ref")                               # optional second sampler budget at evaluation
    if S_ref:
        fields = FIELDS + ["val_solve_ref"]
    mu, p_mask, ref_mix = rc["inner_steps"], rc["prompt_mask_prob"], rc.get("ref_mix", 0.0)
    ref_kl = rc.get("ref_kl", 0.0) if kind == "weighted" else 0.0
    lik = rc.get("likelihood", "onestep")                          # weighted objectives only
    if lik not in ("onestep", "sequential", "trajectory"):
        raise SystemExit(f"unknown rl.likelihood {lik!r}")
    sequential = lik != "onestep"                                  # chain-rule estimates: no one-step logp
    order_temp = rc.get("order_temp", 0.0)                         # phase 4.5a: sampled unmasking order in rollouts
    batch_w = rc.get("weights_over", "group") == "batch"           # phase 4.5a: wd1 weights over the whole batch
    if batch_w and rc["objective"] != "wd1":
        raise SystemExit("rl.weights_over=batch is defined for wd1 only")
    if batch_w or order_temp:
        fields = fields + ["frac_equal"]
    dev = get_device(a.device or base["runtime"]["device"])
    name = run_name(rc, a.seed, a.tag)
    if a.exp:
        a.out, a.ckpt_dir = str(cache / a.exp / "logs"), str(cache / a.exp / "ckpt")
    out, ck = Path(a.out), Path(a.ckpt_dir) / name
    if (out / f"{name}.json").exists() and not a.force:
        print(f"skip {name}: already finished", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    ck.mkdir(parents=True, exist_ok=True)

    init = Path(a.init) if a.init else (cache if p4 else PHASE2_CACHE) / cfg["init"]["checkpoint"]
    init_hash = sha256(init)
    if a.init is None and init_hash != cfg["init"]["sha256"]:
        raise SystemExit(f"{init} does not match the sha256 pinned in {a.config}")

    if p4:
        from phase4.data import load_sets
        sets, bands, names = load_sets(base, dev)
        ds = {k: sets[k].subset((bands[k] == names.index(cfg["band"])).to(dev)) for k in sets}
    else:
        ds = {k: v.to(dev) for k, v in build_datasets(base, base["data"].get("split_seed", 0)).items()}
    train, val, test = ds["train"], ds["val"], ds["test"]
    model = build_model(base).to(dev)
    model.load_state_dict(torch.load(init, map_location=dev))
    model.eval()  # no dropout during RL (rollouts and gradient passes), as in phase 2
    ref = None
    if ref_mix or ref_kl or (kind == "grpo" and params["kl"]):   # frozen copy of the starting model
        ref = copy.deepcopy(model)
        for p in ref.parameters():
            p.requires_grad_(False)
    roll_model = MixedLogits(model, ref, ref_mix) if ref_mix else model
    opt = torch.optim.AdamW(model.parameters(), lr=rc["lr"], weight_decay=0.0)
    log_path, state_path = out / f"{name}.csv", ck / "state.pt"
    t_min = base["diffusion_training"]["t_min"]

    def evaluate(ps) -> tuple[float, float]:
        return solve_rate(model, ps, S, "confidence"), solve_rate(model, ps, 1, "confidence")

    def update(loss) -> float:
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), rc["grad_clip"]).item()
        opt.step()
        return norm

    start, best = 1, (-1.0, 0)
    if state_path.exists():  # resume
        st = torch.load(state_path, map_location=dev)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        torch.set_rng_state(st["rng_cpu"].cpu()); start, best = st["step"] + 1, tuple(st["best"])
        if dev.type == "mps":
            torch.mps.set_rng_state(st["rng_dev"].cpu())
        import pandas as pd
        pd.read_csv(log_path).query("step < @start").to_csv(log_path, index=False)
        print(f"resumed {name} at step {start}", flush=True)
    else:
        seed_all(a.seed)
        v8, v1 = evaluate(val)
        best = (v8, 0)
        torch.save(model.state_dict(), ck / "best.pt")
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fields)
            w.writeheader()
            w.writerow({"step": 0, "val_solve": v8, "val_solve_s1": v1, "elapsed_s": 0.0,
                        **({"val_solve_ref": solve_rate(model, val, S_ref, "confidence")} if S_ref else {})})
    t0 = time.time()
    for step in range(start, steps + 1):
        idx = torch.randint(0, len(train), (B,), device=dev)
        P = train.puzzles[idx].repeat_interleave(G, 0)
        with torch.no_grad():
            ro = sample(roll_model, P, S, rc["rollout_strategy"], rc["rollout_temperature"], order_temp=order_temp)
            O, CS = ro.grid, ro.commit_step
        valid = batch_validity(O, P).view(B, G)
        r = reward_fn(O, P).view(B, G)
        row = {"step": step, "mean_reward": r.mean().item(), "mean_valid": valid.float().mean().item()}
        losses, norms = [], []
        if kind == "diffusion":   # SFT loss on own valid rollouts, or on the true solutions (uses labels)
            if rc["objective"] == "supervised":
                Pt, Ot = P, train.solutions[idx].repeat_interleave(G, 0)
                row["frac_skipped"] = 0.0
            else:
                sel = valid.view(-1)
                Pt, Ot = P[sel], O[sel]
                row["frac_skipped"] = 1 - valid.any(1).float().mean().item()
            if len(Pt):
                for _ in range(mu):
                    loss = diffusion_loss(model, PuzzleSet(Pt, Ot, torch.zeros(len(Pt))), t_min)
                    norms.append(update(loss)); losses.append(loss.item())
        else:
            A, keep = advantages(r)
            if batch_w or order_temp:
                row["frac_equal"] = 1 - keep.float().mean().item()   # groups with equal rewards, used or not
            if batch_w:                                    # every group is used, also those with equal rewards
                keep = torch.ones_like(keep)
            row["frac_skipped"] = 1 - keep.float().mean().item()
            if keep.any():
                Pk, Ok = P.view(B, G, -1)[keep].reshape(-1, P.shape[1]), O.view(B, G, -1)[keep].reshape(-1, P.shape[1])
                vk = valid[keep]
                if kind == "weighted":
                    gw = wd1_batch(r, **params) if batch_w else WEIGHTED[rc["objective"]](r, **params)
                    gw.pos, gw.neg = gw.pos[keep], gw.neg[keep]
                    for _ in range(mu):
                        extra = prompt_mask(Pk, p_mask)   # same single random draw as before
                        if sequential:
                            given = Pk != 0
                            cl = (trajectory_cell_logp(model, Pk, Ok, CS.view(B, G, -1)[keep].reshape(-1, P.shape[1]), extra)
                                  if lik == "trajectory" else sequential_cell_logp(model, Pk, Ok, extra))
                            lp = (cl * (~given)).sum(1).view(-1, G)
                            logp = None
                        else:
                            logp, given = one_step_logprobs(model, Pk, Ok, extra=extra)
                            lp = loglik(logp, Ok, given).view(-1, G)
                        loss = weighted_loss(lp, gw)
                        if ref_kl:
                            with torch.no_grad():
                                rlp = cell_logp(one_step_logprobs(ref, Pk, Ok, extra=extra)[0], Ok)
                            d = rlp - cell_logp(logp, Ok)
                            kl_cells = ((torch.exp(d) - d - 1) * (~given)).sum(1).view(-1, G).sum(1)
                            loss = loss + ref_kl * kl_cells.mean()
                        norms.append(update(loss)); losses.append(loss.item())
                    row.update(weight_diagnostics(gw.w_pos, vk))
                else:   # grpo: old (and reference) per-cell log-probs under the masks of each inner step
                    Ak = A[keep].reshape(-1)
                    masks = [prompt_mask(Pk, p_mask) for _ in range(mu)]
                    with torch.no_grad():
                        old = [cell_logp(one_step_logprobs(model, Pk, Ok, extra=m)[0], Ok) for m in masks]
                        refs = [cell_logp(one_step_logprobs(ref, Pk, Ok, extra=m)[0], Ok) if params["kl"]
                                else None for m in masks]
                    clips = []
                    for m, o, rf in zip(masks, old, refs):
                        logp, given = one_step_logprobs(model, Pk, Ok, extra=m)
                        loss, clip = grpo_loss(cell_logp(logp, Ok), o, Ak, ~given, params["epsilon"],
                                               params["kl"], rf)
                        norms.append(update(loss)); losses.append(loss.item()); clips.append(clip)
                    lp = loglik(logp, Ok, given).view(-1, G)
                    row["clip_frac"] = sum(clips) / len(clips)
                lpd = lp.detach()
                row.update(logp_valid=lpd[vk].mean().item() if vk.any() else float("nan"),
                           logp_invalid=lpd[~vk].mean().item() if (~vk).any() else float("nan"),
                           pred_entropy=predictive_entropy(logp.detach(), given).item() if logp is not None
                           else float("nan"))
        if losses:
            row.update(loss=sum(losses) / len(losses), grad_norm=sum(norms) / len(norms))
        if step % rc["eval_every"] == 0 or step == steps:
            row["val_solve"], row["val_solve_s1"] = evaluate(val)
            if S_ref:
                row["val_solve_ref"] = solve_rate(model, val, S_ref, "confidence")
            if row["val_solve"] > best[0]:
                best = (row["val_solve"], step)
                torch.save(model.state_dict(), ck / "best.pt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": list(best),
                        "rng_cpu": torch.get_rng_state(),
                        "rng_dev": torch.mps.get_rng_state() if dev.type == "mps" else None}, state_path)
            print(f"{name} step={step} reward={row['mean_reward']:.3f} skip={row['frac_skipped']:.2f} "
                  f"val={row['val_solve']:.3f} val_s1={row['val_solve_s1']:.3f}", flush=True)
        row["elapsed_s"] = time.time() - t0
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fields).writerow(row)
    torch.save(model.state_dict(), ck / "final.pt")
    final_val, final_val_s1 = evaluate(val)
    final_test, final_test_s1 = evaluate(test)
    summ = {"name": name, "objective": rc["objective"], "objective_params": params,
            "reward": rc["reward"], "ref_mix": ref_mix, "ref_kl": ref_kl,
            "likelihood": lik, "order_temp": order_temp, "weights_over": "batch" if batch_w else "group", "seed": a.seed, "tag": a.tag, "lr": rc["lr"],
            "steps": steps, "init": str(init),
            "init_sha256": init_hash, "overrides": a.set, "code_commit": git_commit(),
            "best_val": best[0], "best_step": best[1], "final_val": final_val, "final_test": final_test,
            "final_val_s1": final_val_s1, "final_test_s1": final_test_s1,
            "wall_clock_s_this_session": time.time() - t0, "env": env_record(dev)}
    if S_ref:
        summ.update(eval_steps=S, eval_steps_ref=S_ref, final_val_ref=solve_rate(model, val, S_ref, "confidence"),
                    final_test_ref=solve_rate(model, test, S_ref, "confidence"))
    if p4:
        summ.update(band=cfg["band"], n_train=len(train), n_val=len(val), n_test=len(test))
    model.load_state_dict(torch.load(ck / "best.pt", map_location=dev))
    summ["best_test"] = solve_rate(model, test, S, "confidence")
    json.dump(summ, open(out / f"{name}.json", "w"), indent=2)
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
