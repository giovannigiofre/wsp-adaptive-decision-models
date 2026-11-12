"""Builds notebooks/phase4_sudoku9.ipynb; then execute it with
jupyter nbconvert --to notebook --execute --inplace notebooks/phase4_sudoku9.ipynb
"""
import nbformat as nbf

T = {
    "intro": """
# Phase 4: a small masked diffusion model on 9×9 Sudoku

This notebook walks through phase 4 in the order it was run.

- Every cell loads committed tables and figures from `results/phase4/`. No model is loaded and nothing is trained here.
- Every number traces back to a command in `experiments/4.x/README.md`. Each README has the predictions committed before the run and a check of each one afterwards; about half did not hold.
- The summary with limits is `results/phase4/SUMMARY.md`.

One 1.86M-parameter model family, one puzzle generator, one laptop. Nothing here is a new method. Unless stated, numbers are the share of the 667 hard test puzzles filled with a rule-valid grid; brackets are 95% bootstrap intervals over puzzles.
""",
    "s40": """
## 4.0 and 4.1 Data, then supervised training to a plateau

100,002 train puzzles in three difficulty bands by number of blanks, from a local generator, with a data report that had to pass before training. The model was then trained on stored solutions with the masked diffusion loss at a constant learning rate of 3e-4 until a plateau rule stopped it at step 43,546 (4 h 58 min).

**Shows:** unlike the converged 4×4 toy, the number of sampler steps matters a lot here.
**Does not show:** convergence in solve rate. The constant learning rate comes back at the end of this notebook.
""",
    "s42": """
## 4.2 Unmasking order

The same weights, different rules for which cells to commit first.

**Shows:** order by confidence solves 0.81 of the hard band at 16 steps, a random order 0.13. A perfect order would solve every puzzle with the same weights.
**Does not show:** why confidence orders well while ranking wrong cells badly (the calibration table below).
""",
    "s44": """
## 4.3 and 4.4 The phase 3 objectives on 9×9

RL starts from the plateau model on hard puzzles with 4 sampler steps (solve rate 0.535), 500 steps at lr 1e-5, three seeds.

**Shows:** only the objectives that pull valid rollouts up gain; plain wd1 ends below the start.
**Does not show:** that RL improves this model. 4.9e and 4.9f below: the supervised model simply continued at lr 1e-5 gains more than any row here. The table still orders the objectives against each other.
""",
    "s44b": """
## 4.4b, 4.5b, 4.6b Where rollouts fail, and whether errors can be erased

**Shows:** a failed grid is one or two early wrong cells followed by cells that fit them. If the wrong cells are known, masking and re-predicting them solves 0.91 at 4 passes. With any score available without the solution, remasking gains at most 0.02. A detector would need to find about nine wrong cells in ten; a linear readout of the frozen model finds six.
**Does not show:** that no detector can be learned; none was trained beyond a linear probe.
""",
    "s46": """
## 4.5a, 4.6a, 4.6c, 4.6d Taking the objectives apart

- 4.5a: a sampled unmasking order gives more unequal-reward groups and no gain; weights over the batch reach the level of filtered imitation.
- 4.6a: the pull-up objectives are nearly orthogonal to the exact score-function gradient of the solve rate and gain as much as a step along it.
- 4.6c: no single factor (batch weights, one inner step, the paper's optimiser values, dense reward, lower success rate) makes the push-down term of wd1 pay.
- 4.6d: supervised steps on the same puzzles give the same gain as self-training, whichever puzzles carry the loss.

All four start from the constant-lr model, so the caveat of 4.4 applies to them too.
""",
    "s48": """
## 4.8 No stored solutions: printed digits only, then a rule checker

Cold start (4.8a): hide some of the printed digits and train the model to put them back. No rollouts, no checker. Then two loops that keep the model's own checker-valid grids and train on them (`harvest`, `hindsight`), against a control that just continues the givens loss.

**Shows:** at 19,546 steps the loops are ahead of the control (three seeds, two control seeds). Continued to the supervised step count, the control catches up at every budget on test: a head start, not a higher end point.
**Does not show:** seed variance of the long runs (one each).
""",
    "s49": """
## 4.9 The learning rate

The three long models, each continued with its own training for 4,000 steps at lr 1e-5 instead of 3e-4, and nothing else changed.

**Shows:** all three gain, the supervised one most (+0.099 at 4 steps). Like for like, stored solutions are worth 0.14 at 4 steps, 0.07 at 16 and nothing at 81.
**Does not show:** an end point (the supervised model was still improving), a tuned schedule, or seed variance.
""",
    "s49f": """
## 4.9f Self-training again, from the model that had the drop

The 4.4 self-training step and a supervised control, from `sft_lr1e-5`.

**Shows:** neither adds anything at 4 steps; self-training costs 0.03 at 16. The +0.067 of 4.4 is accounted for by the low learning rate of the RL step.
**Does not show:** that training on own valid grids can never help (one start, one learning rate, 500 steps).
""",
    "s49g": """
## 4.9g and 4.9h Closing checks

4.9g: plain wd1 and positive-only wd1 from `sft_lr1e-5`, to see whether the one sign that 4.4 still supports (push-down hurts, pull-up does not) holds without the free gain. 4.9h: the 500-step pass on own valid grids from `givens_lr1e-5`, to see whether the checker adds anything to the label-free model after its drop.

**Shows:** plain wd1 loses 0.077 at 4 steps, the same code without the push-down term 0.012. On the label-free model the pass on own valid grids still adds 0.033 at 4 steps and nothing at 16.
**Does not show:** that the 0.033 comes from the checker (no control arm from that start; the run was interrupted and resumed, see its README), or that no bounded push-down term could work from this start (not rerun).
""",
    "end": """
## What phase 4 leaves

1. On 9×9, sampler steps and unmasking order matter; failures start from one or two early wrong cells.
2. No RL objective tested here beat a learning-rate drop on the supervised model.
3. The push-down term of wd1 never paid in any setting tried; from the best model it is the only clear effect of an RL objective, and it is a loss (4.9g).
4. A model trained only on printed digits gets within 0.14 / 0.07 / 0.00 of the supervised one at 4 / 16 / 81 steps; the rule checker gave a faster start and no higher end, apart from 0.03 at 4 steps from a last pass on own valid grids (4.9h).
5. Five readings were corrected by later runs, three of them by the same cause: models trained at a constant learning rate, followed by steps at a lower one.

Limits and the table of corrected readings: `results/phase4/SUMMARY.md`.
""",
}

cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip()))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip()))


md(T["intro"])
code("""
import os
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT)
import pandas as pd
from IPython.display import Image, Markdown, display
R = Path("results/phase4")
pd.set_option("display.precision", 3); pd.set_option("display.width", 200)

def show(exp, table, image=None, cols=None, query=None):
    df = pd.read_csv(R / exp / table)
    if query:
        df = df.query(query)
    display(df[cols] if cols else df)
    if image:
        display(Image(R / exp / image))

def hard_test(path):
    d = pd.read_csv(R / path)
    return d[d.band == "hard"].set_index("sampler_steps").valid
""")
md(T["s40"])
code("""
display(Markdown("\\n## ".join((R / "data_report.md").read_text().split("\\n## ")[:2])))
display(Image(R / "data_difficulty_by_k.png"))
display(Image(R / "sft" / "sft_training_curve.png"))
show("sft", "sft_valid_test.csv", "sft_solve_by_steps.png")
""")
md(T["s42"])
code("""
show("4.2", "strategies_table.csv", "strategies.png", query="band == 'hard'")
show("4.2", "calibration.csv", cols=["band", "steps", "solve_rate", "cell_acc", "mean_cell_conf", "cell_ece", "cell_auroc"])
""")
md(T["s44"])
code("""
show("4.4", "summary.csv", "objectives.png", ["method", "n_seeds", "test_mean", "test_std", "test_minus_start", "test_minus_start_lo", "test_minus_start_hi", "test_ref_minus_start"])
""")
md(T["s44b"])
code("""
show("4.4b", "error_structure.csv")
show("4.5b", "remask.csv", query="model == 'start'")
show("4.6b", "degraded_oracle.csv")
show("4.6b", "probes.csv")
""")
md(T["s46"])
code("""
show("4.5a", "paired.csv", "factorial.png")
show("4.6a", "directions.csv", "audit.png", ["model", "objective", "cos_with_gstar", "J_gain_per_unit_step"])
show("4.6c", "arm_pairs.csv")
show("4.6d", "arm_pairs.csv")
""")
md(T["s48"])
code("""
show("4.8c", "pooled_pairs_two_controls.csv")
show("4.8e", "long_pairs_test.csv")
""")
md(T["s49"])
code("""
rows = {"supervised, lr 3e-4": "sft/sft_valid_test.csv", "supervised, then lr 1e-5": "4.9e/sft_lr1e-5_valid_test.csv",
        "givens only, lr 3e-4": "4.8e/givens_long_valid_test.csv", "givens only, then lr 1e-5": "4.9d/givens_lr1e-5_valid_test.csv",
        "hindsight loop, lr 3e-4": "4.8d/hindsight_long_valid_test.csv", "hindsight loop, then lr 1e-5": "4.9d/hindsight_lr1e-5_valid_test.csv"}
display(pd.DataFrame({k: hard_test(v) for k, v in rows.items()}).T.rename_axis("hard test band, valid share at N sampler steps"))
display(Image(R / "overview_hard_test.png"))
show("4.9e", "pairs.csv", query="band == 'hard'")
""")
md(T["s49f"])
code("""show("4.9f", "pairs.csv")""")
md(T["s49g"])
code("""
show("4.9g", "pairs.csv")
show("4.9h", "pairs.csv")
""")
md(T["end"])

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
nbf.write(nb, "notebooks/phase4_sudoku9.ipynb")
print("cells:", len(cells))
