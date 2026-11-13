Compact outputs only: CSV summaries, small PNG/GIF figures, per-phase `SUMMARY.md`, and the final `REPORT.md`. No weights, no large logs.

- `REPORT.md` technical report of all four phases; `REPORT_SIMPLE.md` the same in plain language.
- `phase1/`, `phase2/`, `phase3/`, `phase4/`: one `SUMMARY.md` each, then one folder per experiment id matching `experiments/<id>/README.md`.
- `phase4/overview_hard_test.png`: the three long 9×9 models before and after the learning-rate drop (`python -m phase4.overview_plot`).
