"""Generic RL loop for the toy Sudoku dLLM with a pluggable objective and reward (phase 3.0).

Per RL step: B puzzles x G completions (same sampler settings as phase 2), reward from `rewards.py`,
group weights from `objectives.py`, one-step mean-field log p_hat, mu inner gradient steps.
The order of random draws matches phase2.rl_wd1 (puzzle indices, rollouts, prompt masks), so
`--set rl.objective_params.beta=1` and `=0` reproduce phase 2's wd1 and positive-only runs up to MPS
non-determinism. Data, model and sampler come from the phase 2 config named in `base_config`.

Resumable (state.pt every eval), one CSV row per step.

PYTHONPATH=src python -m phase3.rl --seed 0
PYTHONPATH=src python -m phase3.rl --seed 0 --set rl.objective_params.beta=0
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from phase2.data import batch_validity, build_datasets, digits_to_tokens
from phase2.evaluate import solve_rate
from phase2.model import build_model
from phase2.sampler import sample
from phase2.utils import CACHE as PHASE2_CACHE
from phase2.utils import apply_overrides, env_record, get_device, load_cfg, seed_all

from .objectives import OBJECTIVES, weight_diagnostics, weighted_loglik_loss
from .rewards import REWARDS

CACHE = Path(os.environ.get("PHASE3_CACHE", Path.home() / "dllm-cache" / "phase3"))
FIELDS = ["step", "mean_reward", "mean_valid", "frac_skipped", "loss", "grad_norm", "w_entropy", "w_ess",
          "w_pos_on_invalid", "logp_valid", "logp_invalid", "pred_entropy", "val_solve", "val_solve_s1",
          "elapsed_s"]


def one_step_logprobs(model, puzzles: torch.Tensor, completions: torch.Tensor, p_mask: float = 0.0,
                      generator: torch.Generator | None = None):
    """Same input and random draw as phase2.rl_wd1.one_step_loglik, but returns the per-cell
    log-probs (N,16,4) and the given mask so diagnostics can reuse the forward pass."""
    given = puzzles != 0
    extra = given & (torch.rand(puzzles.shape, device=puzzles.device, generator=generator) < p_mask)
    inp = torch.where(given & ~extra, completions, torch.zeros_like(completions))
    return F.log_softmax(model(digits_to_tokens(inp)).float(), -1), given


def loglik(logp: torch.Tensor, completions: torch.Tensor, given: torch.Tensor) -> torch.Tensor:
    """(N,) one-step log p_hat: sum over non-given cells of log p(completion cell)."""
    lp = logp.gather(-1, (completions - 1).unsqueeze(-1)).squeeze(-1)
    return (lp * (~given)).sum(1)


def predictive_entropy(logp: torch.Tensor, given: torch.Tensor) -> torch.Tensor:
    """Mean entropy (nats, max ln 4) of the one-step predictive distribution over non-given cells."""
    ent = -(logp.exp() * logp).sum(-1)
    return (ent * (~given)).sum() / (~given).sum()


def run_name(rc: dict, seed: int, tag: str = "") -> str:
    params = "_".join(f"{k}{float(v):g}" for k, v in sorted(rc["objective_params"].items()))
    reward = "" if rc["reward"] == "binary" else f"_{rc['reward']}"
    return f"{rc['objective']}_{params}{reward}_seed{seed}{tag}"


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
    ap.add_argument("--out", default="results/phase3/rl")
    ap.add_argument("--ckpt-dir", default=str(CACHE / "rl"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cfg = apply_overrides(load_cfg(a.config), a.set)
    base = load_cfg(cfg["base_config"])
    rc = cfg["rl"]
    objective, reward_fn = OBJECTIVES[rc["objective"]], REWARDS[rc["reward"]]
    steps, B, G, S = rc["max_rl_steps"], rc["batch_puzzles"], rc["group_size"], rc["rollout_steps"]
    dev = get_device(a.device or base["runtime"]["device"])
    name = run_name(rc, a.seed, a.tag)
    out, ck = Path(a.out), Path(a.ckpt_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    ck.mkdir(parents=True, exist_ok=True)

    init = Path(a.init) if a.init else PHASE2_CACHE / cfg["init"]["checkpoint"]
    init_hash = sha256(init)
    if a.init is None and init_hash != cfg["init"]["sha256"]:
        raise SystemExit(f"{init} does not match the sha256 pinned in {a.config}")

    ds = {k: v.to(dev) for k, v in build_datasets(base, base["data"].get("split_seed", 0)).items()}
    train, val, test = ds["train"], ds["val"], ds["test"]
    model = build_model(base).to(dev)
    model.load_state_dict(torch.load(init, map_location=dev))
    model.eval()  # no dropout during RL (rollouts and gradient passes), as in phase 2
    opt = torch.optim.AdamW(model.parameters(), lr=rc["lr"], weight_decay=0.0)
    log_path, state_path = out / f"{name}.csv", ck / "state.pt"

    def evaluate(ps) -> tuple[float, float]:
        return solve_rate(model, ps, S, "confidence"), solve_rate(model, ps, 1, "confidence")

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
            w = csv.DictWriter(f, FIELDS)
            w.writeheader()
            w.writerow({"step": 0, "val_solve": v8, "val_solve_s1": v1, "elapsed_s": 0.0})
    t0 = time.time()
    for step in range(start, steps + 1):
        idx = torch.randint(0, len(train), (B,), device=dev)
        P = train.puzzles[idx].repeat_interleave(G, 0)
        with torch.no_grad():
            O = sample(model, P, S, rc["rollout_strategy"], rc["rollout_temperature"]).grid
        valid = batch_validity(O, P).view(B, G)
        r = reward_fn(O, P).view(B, G)
        gw = objective(r, **rc["objective_params"])
        keep = gw.keep
        row = {"step": step, "mean_reward": r.mean().item(), "mean_valid": valid.float().mean().item(),
               "frac_skipped": 1 - keep.float().mean().item()}
        if keep.any():
            Pk, Ok = P.view(B, G, 16)[keep].reshape(-1, 16), O.view(B, G, 16)[keep].reshape(-1, 16)
            coef, vk = gw.coef[keep], valid[keep]
            losses, norms = [], []
            for _ in range(rc["inner_steps"]):
                logp, given = one_step_logprobs(model, Pk, Ok, rc["prompt_mask_prob"])
                lp = loglik(logp, Ok, given).view(-1, G)
                loss = weighted_loglik_loss(lp, coef)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                norms.append(torch.nn.utils.clip_grad_norm_(model.parameters(), rc["grad_clip"]).item())
                opt.step()
                losses.append(loss.item())
            lpd = lp.detach()
            row.update(loss=sum(losses) / len(losses), grad_norm=sum(norms) / len(norms),
                       logp_valid=lpd[vk].mean().item() if vk.any() else float("nan"),
                       logp_invalid=lpd[~vk].mean().item() if (~vk).any() else float("nan"),
                       pred_entropy=predictive_entropy(logp.detach(), given).item(),
                       **weight_diagnostics(gw.w_pos[keep], vk))
        if step % rc["eval_every"] == 0 or step == steps:
            row["val_solve"], row["val_solve_s1"] = evaluate(val)
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
            csv.DictWriter(f, FIELDS).writerow(row)
    torch.save(model.state_dict(), ck / "final.pt")
    final_val, final_val_s1 = evaluate(val)
    final_test, final_test_s1 = evaluate(test)
    summ = {"name": name, "objective": rc["objective"], "objective_params": rc["objective_params"],
            "reward": rc["reward"], "seed": a.seed, "lr": rc["lr"], "steps": steps, "init": str(init),
            "init_sha256": init_hash, "overrides": a.set, "code_commit": git_commit(),
            "best_val": best[0], "best_step": best[1], "final_val": final_val, "final_test": final_test,
            "final_val_s1": final_val_s1, "final_test_s1": final_test_s1,
            "wall_clock_s_this_session": time.time() - t0, "env": env_record(dev)}
    model.load_state_dict(torch.load(ck / "best.pt", map_location=dev))
    summ["best_test"] = solve_rate(model, test, S, "confidence")
    json.dump(summ, open(out / f"{name}.json", "w"), indent=2)
    print(json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
