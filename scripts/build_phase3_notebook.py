"""Builds notebooks/phase3_rl_playground.ipynb; then execute it with
jupyter nbconvert --to notebook --execute --inplace notebooks/phase3_rl_playground.ipynb
"""
import nbformat as nbf

T = {
    "intro": """
# Phase 3: playing with RL on the toy Sudoku dLLM

This notebook walks through phase 3 step by step.

- Every cell loads committed results from `results/phase3/`. One small live demo runs on CPU if the local checkpoint exists.
- Every number traces back to a command in `experiments/3.x/README.md`. Each of those READMEs has a prediction that was committed before the runs, and a check of that prediction afterwards.
- The summary with limitations is `results/phase3/SUMMARY.md`.

Everything here is a toy: a 0.8M-parameter masked diffusion model on 4×4 Sudoku. Nothing is a new method.
""",
    "s30": """
## 3.0 A generic trainer that reproduces phase 2

The phase 3 trainer has a pluggable objective and reward and logs extra diagnostics. Before using it, the phase 2 wd1 and positive-only runs were repeated with it.

**Shows:** same solve rates as phase 2 for every seed, so later differences come from the objective or the reward.
**Does not show:** anything new about wd1.
""",
    "s31a": """
## 3.1a Was step 500 too early?

Phase 2 stopped at 500 RL steps with wd1 still improving. Here both variants run for 2000 steps with 5 seeds.

**Shows:** the ranking holds. wd1 narrows the gap from 20 points to 9 but stays behind in every seed. By the end more than 90% of the groups have equal rewards and give no gradient.
**Does not show:** why. The standard horizon for the rest of the phase was fixed at 1500 steps by a rule written before the runs.
""",
    "s31b": """
## 3.1b Sharp weights

With 0/1 rewards and τ = 1 the softmax weights are nearly flat: about half of the positive weight falls on *invalid* rollouts. Lower τ makes them selective.

**Shows:** positive-only's early collapse disappears when the weights are selective (τ ≤ 0.1). wd1 does not care about τ at all.
**Does not show:** that flat weights are the only cause of the collapse.
""",
    "s31c": """
## 3.1c How much negative push?

wd1 and positive-only are the two ends of one family, loss = Σ (β·w⁻ − w⁺)·log p̂. The table also has a floored negative term, which stops pushing a rollout down once its estimated log-likelihood is below a floor (not from the wd1 paper).

**Shows:** a cliff, not a slope. Any β below 1 ends near 100%, β = 1 (wd1) ends at 0.86, and β = 2 destroys the model. A floor at −12 lifts wd1 to 0.99.
**Does not show:** that β = 1 is a bad choice in general, or that −12 is a recipe.
""",
    "s31d": """
## 3.1d The starting model

Phase 2 started from one early snapshot of a model that was about to become good. Here: an earlier and a later snapshot, and a model trained to convergence on only 10 of the 200 solutions.

**Shows:** the phase 2 ranking is a property of that snapshot. Positive-only (τ = 1) wins from the two mid-warm-up snapshots, goes to zero from the earlier one, and rises then collapses from the converged model. wd1 improves from every start.
**Does not show:** how the safer variants behave from these starts; that is 3.1g below.
""",
    "s31e": """
## 3.1e A graded reward

Reward = share of the 12 rows, columns and boxes that are valid, instead of 0/1.

**Shows:** no gain for wd1, and positive-only collapses to zero even at the temperature that was safe with the binary reward, because the rewards within a group are now close together and the weights are flat again.
**Does not show:** what much sharper or rank-based weights would do.
""",
    "s31f": """
## 3.1f A reference policy in the rollouts

The upstream wd1 code has no reference term in its loss. Its reference-policy variant mixes the reference model's logits into the *sampling*. This is the toy equivalent.

**Shows:** little change. All mixes are within 0.016 of plain wd1 on val.
**Does not show:** whether the 4-point difference on test is real; 3 seeds cannot tell.
""",
    "s32": """
## 3.2 wd1 next to simpler objectives

Same rollouts, same likelihood estimate, same budget, 5 seeds.

**Shows:** plain REINFORCE gives wd1's curve, so the softmax weighting is not what matters here. Imitating one's own valid rollouts gets to about 100% without labels. d1-style GRPO is clearly better than wd1.
**Does not show:** that imitation beats wd1 in general: valid rollouts are plentiful and always correct in this task.
""",
    "s32b": """
## 3.2b Why GRPO did better: the KL term to the starting model

GRPO's clipping never became active, so what separates it from REINFORCE is its KL penalty towards the starting model. Remove it from GRPO, add it to REINFORCE and to wd1.

**Shows:** the KL term is the whole difference. wd1 with a small KL penalty reaches 0.976.
**Does not show:** that wd1 needs a KL term at scale. The weight needed tuning (0.01 worked, 0.04 did not).
""",
    "s31g": """
## 3.1g Do the better variants survive other starting models?

The four variants that beat wd1 from the phase 2 start, rerun from the two harder starts of 3.1d.

**Shows:** self-training with the SFT loss on the model's own valid rollouts reaches about 100% from all three starts. The floored wd1 is the most robust wd1 variant (0.99, 0.99, 0.89). A KL penalty to the start helps from an unfinished model and hurts from a converged weak one. Plain wd1 is never the best and never fails.
**Does not show:** that self-training is better in general: here a valid rollout is always a correct answer and valid rollouts exist from the start.
""",
    "s33": """
## 3.3 Rewarding honest confidence

A 17th position reads the grid and states a confidence in one of four values. It is rewarded with r − (q − r)², which is highest for an honest q (idea from RLCR). The baseline needs no RL: recalibrate the confidence the model already has with a 2-parameter map.

**Shows:** trained together with the grid, the token does not become calibrated: it settles on one value early and stops exploring. The 2-parameter recalibration is far better.
**Does not show:** that the idea cannot work; the target was moving and the weights were flat. 3.3b removes both.
""",
    "s33b": """
## 3.3b The confidence token on a frozen denoiser

Same token, but the denoiser is frozen (the wd1 checkpoints from 3.1b), so the target is stationary; RL is plain REINFORCE on the Brier-shaped reward.

**Shows:** the token learns to verify its own grid (AUROC 0.93 with RL, 0.98 with the probe) and beats the best post-hoc recalibration on the Brier score. Read from the puzzle alone, it states the base rate.
**Does not show:** uncertainty about a reasoning process; checking a finished 4×4 grid is easy for this architecture.
""",
    "s34a": """
## 3.4a Is there headroom in the unmasking order?

The weak denoiser is frozen and one cell is committed per step. An oracle that knows the solution always commits a cell whose prediction is right.

**Shows:** a large gap: 0.98 for the oracle against 0.56 for the confidence rule. A correct next move almost always exists.
**Does not show:** that a policy without the solution can find it.
""",
    "s34demo": """
### Live demo: confidence order against oracle order on one puzzle

A single 12-hidden-cell test puzzle, decoded twice with the frozen weak denoiser. This is one example, not a measurement.
""",
    "s34b": """
## 3.4b Learning the order with RL

A small head scores every masked cell; training is REINFORCE on the validity reward. Here the log-probability of an order is exact, unlike the denoiser's likelihood.

**Shows:** from the three heuristic scores RL rediscovers the confidence rule in about 50 steps. With the denoiser's hidden state it finds an order 11 points better, about a quarter of the way to the oracle. That policy does not transfer to other denoisers.
**Does not show:** that the rest of the gap can be learned; a head trained *with* the solutions does no better.
""",
    "end": """
## Summary

See `results/phase3/SUMMARY.md` for the tables, the prediction scorecard, the run notes and the limitations.
""",
}

cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


md(T["intro"])
code("""
import json, os, sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT); sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd, torch
from IPython.display import Image, display
from phase2.utils import CACHE, load_cfg
R = Path("results/phase3")
HAVE_CKPT = (CACHE / "sft" / "sft_weak.pt").exists()
pd.set_option("display.precision", 3)

def show(exp, table, image=None, cols=None):
    df = pd.read_csv(R / exp / table)
    display(df[cols] if cols else df)
    if image:
        display(Image(R / exp / image))

print("torch", torch.__version__, "| local checkpoints present:", HAVE_CKPT)
""")
md(T["s30"])
code("""
display(pd.read_csv(R / "repro_3.0_summary.csv"))
display(Image(R / "repro_3.0.png"))
""")
md(T["s31a"])
code("""show("3.1a", "val_by_step.csv", "horizon.png"); print(json.load(open(R / "3.1a" / "horizon.json")))""")
md(T["s31b"])
code("""show("3.1b", "summary.csv", "tau_sweep.png", ["variant", "tau", "w_pos_on_invalid_steps_1_50", "min_val_mean", "final_val_mean", "final_test_mean", "final_test_std"])""")
md(T["s31c"])
code("""
show("3.1c", "summary.csv", "beta_sweep.png", ["tau", "beta", "min_val_mean", "final_test_mean", "final_test_std"])
show("3.1c", "floor.csv", cols=["floor", "final_test_mean", "final_test_std", "final_test_s1_mean", "logp_valid_last_250_steps"])
""")
md(T["s31d"])
code("""
show("3.1d", "sft_subset_summary.csv", cols=["n_solutions", "n_puzzles", "final_val_solve", "final_seen_solutions_solve", "final_unseen_train_solve", "chosen"])
show("3.1d", "summary.csv", "starting_model.png", ["start", "variant", "start_val", "final_val_mean", "final_test_mean", "final_test_std", "min_val_mean"])
""")
md(T["s31e"])
code("""show("3.1e", "summary.csv", "graded_reward.png", ["tau", "variant", "reward", "frac_skipped_steps_1_300", "val_at_300", "final_test_mean", "final_test_std"])""")
md(T["s31f"])
code("""show("3.1f", "summary.csv", "reference_mix.png", ["ref_mix", "final_val_mean", "final_test_mean", "final_test_std", "frac_skipped_last_250_steps"])""")
md(T["s32"])
code("""show("3.2", "summary.csv", "objectives.png", ["method", "n", "final_test_mean", "final_test_std", "final_test_s1_mean", "min_val_mean", "first_step_val_07_median"])""")
md(T["s32b"])
code("""show("3.2b", "summary.csv", "kl_to_start.png", ["arm", "final_test_mean", "final_test_std", "logp_valid_last_250_steps", "frac_skipped_last_250_steps"])""")
md(T["s31g"])
code("""show("3.1g", "summary.csv", "other_starts.png", ["start", "arm", "final_test_mean", "final_test_std", "best_test_mean", "max_drop_from_best_val"])""")
md(T["s33"])
code("""
base = pd.read_csv(R / "3.3" / "baseline_implicit.csv")
display(base[["checkpoint", "test_solve", "raw_mean", "raw_brier", "raw_ece", "raw_auroc", "platt_brier", "platt_ece"]])
show("3.3", "summary.csv", "calibration.png", ["arm", "solve_mean", "base_rate_brier_mean", "q_expected_brier_mean", "q_expected_brier_std", "q_expected_ece_mean", "q_expected_auroc_mean"])
""")
md(T["s33b"])
code("""show("3.3b", "summary.csv", "calibration.png", ["arm", "solve_mean", "base_rate_brier_mean", "q_expected_brier_mean", "q_expected_brier_std", "q_expected_ece_mean", "q_expected_auroc_mean"])""")
md(T["s34a"])
code("""
display(pd.read_csv(R / "3.4a" / "oracle.csv").query("split == 'test'"))
display(pd.read_csv(R / "3.4a" / "oracle_by_hidden_cells.csv"))
""")
md(T["s34demo"])
code("""
if HAVE_CKPT:
    from phase2.data import PuzzleSet, build_datasets
    from phase2.sampler import sample
    from phase2.sudoku import to_str
    from phase3.order_policy import load_denoiser, oracle_order
    base = load_cfg("configs/phase2.yaml")
    test = build_datasets(base, 0)["test"]
    m = load_denoiser(base, CACHE / "sft" / "sft_weak.pt", "cpu")
    hard = (test.k == 12).nonzero().squeeze(1)
    conf = sample(m, test.puzzles[hard], 16, "confidence", 0.0)
    orc = oracle_order(m, test.subset(hard))["solved"]
    ok_conf = (conf.grid == test.solutions[hard]).all(1)
    i = int((orc & ~ok_conf).nonzero()[0])      # first puzzle the oracle solves and the confidence rule does not
    print("puzzle:"); print(to_str(test.puzzles[hard][i].tolist()))
    print("\\nconfidence order (wrong cells marked by comparing with the solution below):"); print(to_str(conf.grid[i].tolist()))
    print("\\nsolution, reached by the oracle order with the same denoiser:"); print(to_str(test.solutions[hard][i].tolist()))
    print("\\ncommit step per cell under the confidence rule (-1 = given):"); print(conf.commit_step[i].view(4, 4).numpy())
    print("\\n12-hidden-cell test puzzles solved: confidence", round(ok_conf.float().mean().item(), 3), "| oracle", round(orc.float().mean().item(), 3))
else:
    print("checkpoint not present locally; skipping live demo")
""")
md(T["s34b"])
code("""
show("3.4b", "summary.csv", "order_policy.png")
show("3.4b", "transfer_summary.csv")
""")
md(T["end"])

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, "notebooks/phase3_rl_playground.ipynb")
print("cells:", len(cells))
