"""Builds notebooks/phase2_sudoku_wd1.ipynb; then execute it with
jupyter nbconvert --to notebook --execute --inplace notebooks/phase2_sudoku_wd1.ipynb
"""
import nbformat as nbf

T = {
    "intro": """
# Phase 2: a toy Sudoku dLLM and hand-written wd1 on a MacBook

This notebook walks through phase 2 step by step.

- Most cells load committed results from `results/phase2/`. One small live demo runs on CPU if the local checkpoint exists.
- Every number traces back to a command listed in `experiments/2.x/README.md`.
- The summary with limitations is `results/phase2/SUMMARY.md`.

This is a small-scale replication exercise and not a new method.
""",
    "s20": """
## 2.0 Data: 4×4 Sudoku split by solution grid

- There are exactly 288 valid 4×4 grids.
- They are split *by grid* into train, val and test (200/44/44). Test puzzles therefore come from solutions the model never saw.
- A puzzle hides k ∈ {6, 8, 10, 12} cells. It is kept only if its completion is unique, checked exactly against all 288 grids.

**Shows:** the task and splits are exact and leak-free.
**Does not show:** anything about difficulty at larger sizes (9×9).
""",
    "s21": """
## 2.1 Masked-diffusion training and the weak checkpoint

- Model: a 0.8M-param bidirectional transformer trained with the LLaDA/MDLM loss.
- The weak checkpoint is chosen on val only: val solve rate in [0.3, 0.6], closest to 0.45. The result is step 120 (val 0.42).
- The weak window is narrow. The model goes from 11% to 98% val within about 100 steps, so the weak model is a mid-transition snapshot.

**Shows:** the training recipe works and gives a reproducible starting point for RL.
**Does not show:** that this weak model resembles a weak *large* model.

The first attempt (positions only, without row/column/box embeddings) plateaued:
""",
    "s22": """
## 2.2 Unmasking strategy × step budget

Test solve rate is measured for each unmasking strategy at each step budget S (greedy decoding).

- For the weak model, the order in which cells are committed matters a lot. At S = 8: confidence 0.48, random 0.14.
- More steps also help: one-shot decoding solves only 1.8%.
- The fully trained model solves everything in one step regardless of strategy.

**Shows:** confidence-ordered multi-step decoding is what makes a partially trained denoiser usable.
**Does not show:** that the ranking carries over to LLaDA-8B.
""",
    "s22demo": """
### Live demo: how the weak model fills a puzzle

The demo runs a greedy confidence-ordered decode with S = 16. It prints:

- the step at which each cell was committed;
- the committed probability for each cell.

Compare the commit order with how constrained each cell is. This is a single example, not a measurement.
""",
    "s23": """
## 2.3 RL with wd1 vs positive-only (3 seeds each), and the τ sweep

- All RL runs start from the weak SFT checkpoint and run 500 steps (B = 32 puzzles × G = 8 rollouts, binary reward). Each takes about 1 minute on MPS.
- **wd1 (τ = 1):** improves smoothly to 0.738 ± 0.005 test.
- **Positive-only (w⁻ = 0):**
  - First collapses in every seed: val drops from 0.42 to about 0.10 within 150 steps.
  - Meanwhile it becomes *more* confident in its own rollouts (right panel). This is the instability wd1's negative term is meant to prevent.
  - It then recovers and ends at 0.932 ± 0.016, clearly above wd1.
- **τ:** the τ sweep shows wd1 is insensitive to τ ∈ {0.5, 1, 2}.

**Shows:**
- wd1 avoids the early collapse.
- wd1 lifts the weak model from 0.48 to 0.74.

**Does not show:**
- That positive-only ends worse: here it ends better at 500 steps. This contradicts the paper's claim. Possible reasons include the toy setting, binary rewards, and the absence of a KL term; none of them is tested.
- Behaviour beyond 500 steps.
""",
    "s23sanity": """
### Sanity check: is the one-step likelihood a usable proxy?

- wd1 needs log p(completion). This implementation uses a single forward pass with all non-given cells masked.
- The check compares it with a sequential estimate that reveals one cell per pass, in row-major order.
- Result: Spearman ≈ 0.86 on weak SFT and 0.79 after RL. The one-step estimate is biased low by 3–5 nats.

**Shows:** a rank-correlated proxy, good enough for weighting.
**Does not show:** that it is unbiased, or the best estimator.
""",
    "s24": """
## 2.4 Calibration before and after RL

Setup: test set, greedy decoding, S = 8.

- *Cell confidence* = the probability of the committed value. *Puzzle confidence* = the mean of its cell confidences.
- ECE uses 10 bins. AUROC measures how well confidence separates right from wrong.

Results:

- **Weak SFT:** slightly under-confident (cell conf 0.72 vs acc 0.76, ECE 0.045).
- **wd1:** over-confident (0.95 vs 0.88, ECE ≈ 0.06–0.07). AUROC improves a little (cell 0.62 → 0.65, puzzle 0.82 → 0.86).
- **Positive-only:** under-confident, but discriminates best (cell AUROC 0.73, puzzle ECE 0.05).

**Shows:** RL improves ranking but can shift the model's confidence level. wd1 here pushes it towards over-confidence.
**Does not show:** calibration at other temperatures or budgets, or at scale.
""",
    "s25": """
## 2.5 Takeaways and limits

- The mechanics are implemented from scratch and tested: masked-diffusion loss, confidence-ordered unmasking, wd1 weights and sign, and the one-step likelihood.
- On this toy task:
  - decoding order matters;
  - wd1 is stable and helps;
  - positive-only collapses early but ends ahead at 500 steps.
- The task is tiny, rewards are binary, RL uses no KL term, and runs use 3 seeds × 500 steps.
- These results say nothing direct about LLaDA-8B-scale RL. See `results/phase2/SUMMARY.md`.
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
R = Path("results/phase2")
cfg = load_cfg()
HAVE_CKPT = (CACHE / "sft" / "sft_weak.pt").exists()
pd.set_option("display.precision", 3)
print("torch", torch.__version__, "| local checkpoints present:", HAVE_CKPT)
""")
md(T["s20"])
code("""
from phase2.sudoku import enumerate_solutions, to_str, count_completions
from phase2.data import split_solutions, make_puzzles
sols = enumerate_solutions()
print("number of solved 4x4 grids:", len(sols))
print(to_str(sols[0]))
sp = split_solutions(0)
print({k: len(v) for k, v in sp.items()})
ps = make_puzzles(sp["test"][:1], 1, ks=[10], seed=0)
p = ps.puzzles[0].tolist()
print("\\nA test puzzle with k=10 hidden cells (unique completion:", count_completions(p) == 1, ")")
print(to_str(p))
""")
code("""pd.read_csv(R / "data_stats.csv").pivot(index="k", columns="split", values="n_puzzles")""")
md(T["s21"])
code("""
print(json.dumps({k: v for k, v in json.load(open(R / "sft_selection.json")).items() if k != "env"}, indent=1))
display(Image(R / "sft_training_curve.png"))
""")
code("""
att = pd.read_csv(R / "sft_attempts" / "attempt1_learnedpos_lr3e-4_20k.csv")
print("failed attempt 1 (plain learned positions): best val solve", att.val_solve.max(), "after", att.step.max(), "steps")
""")
md(T["s22"])
code("""
display(pd.read_csv(R / "strategies_table.csv"))
display(Image(R / "strategies_solve_rate.png"))
""")
md(T["s22demo"])
code("""
if HAVE_CKPT:
    from phase2.analysis import load
    from phase2.sampler import sample
    m = load(cfg, CACHE / "sft" / "sft_weak.pt", "cpu")
    pz = make_puzzles(sp["test"][:1], 1, ks=[12], seed=1).puzzles
    out = sample(m, pz, 16, "confidence", 0.0)
    order = out.commit_step[0].view(4, 4)
    print("puzzle:"); print(to_str(pz[0].tolist()))
    print("\\nweak-SFT completion (greedy, confidence, S=16):"); print(to_str(out.grid[0].tolist()))
    print("\\ncommit step per cell (-1 = given):"); print(order.numpy())
    print("\\ncommit confidence:"); print(np.round(out.commit_conf[0].view(4, 4).numpy(), 2))
else:
    print("checkpoint not present locally; skipping live demo")
""")
md(T["s23"])
code("""
display(pd.read_csv(R / "rl_summary.csv"))
display(Image(R / "rl_wd1_vs_positive_only.png"))
display(Image(R / "rl_tau_sweep.png"))
""")
md(T["s23sanity"])
code("""
print(json.dumps(json.load(open(R / "likelihood_estimators.json")), indent=1))
display(Image(R / "likelihood_estimators.png"))
""")
md(T["s24"])
code("""
cal = pd.read_csv(R / "calibration.csv")
display(cal[cal.ckpt.isin(["sft_weak", "rl_best", "sft_full"])])
rl = cal[cal.ckpt.str.startswith("rl_wd1") | cal.ckpt.str.startswith("rl_positive")].copy()
rl["group"] = rl.ckpt.str.replace(r"_seed\\d+", "", regex=True)
display(rl.groupby("group")[["solve_rate", "mean_cell_conf", "cell_ece", "cell_auroc", "puzzle_ece", "puzzle_auroc"]].agg(["mean", "std"]))
display(Image(R / "calibration_before_after.png"))
""")
md(T["s25"])

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, "notebooks/phase2_sudoku_wd1.ipynb")
print("cells:", len(cells))
