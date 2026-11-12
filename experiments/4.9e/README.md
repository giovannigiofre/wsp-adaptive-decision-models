# 4.9e: the supervised model with the same learning-rate drop

**Status: setup and predictions, written before the run (2026-11-12).** Seen before writing them: everything up to [4.9d](../4.9d/README.md).

**Goal.** 4.9d: dropping the learning rate from 3e-4 to 1e-5 lifts the two long label-free models by about +0.05 at 4 sampler steps and +0.04 at 81 on hard test. The supervised model of 4.1 was trained at constant 3e-4 to its plateau and never had that drop, so the gap "with solutions against without" read in 4.9d compares unlike things. Question: what does the supervised model gain from the same drop, and what is the gap after both had it?

**Setup.** The supervised run of 4.1 (`sft`) is resumed from its saved state at step 43,500 (model, AdamW moments, RNG) and continued for 4,000 steps to step 47,500 with `--lr 1e-5` (new option of `phase4.train`, same helper as 4.9d; plateau rule off, stop fixed at step 47,500). Everything else as 4.1: `configs/phase4.yaml`, batch 256, the stored solutions of the train set as targets. Run name `sft_lr1e-5`. The label-free runs of 4.9d started at step 43,546 and made the same 4,000 steps.

**Evaluation.** Val every 500 steps (loss, solve per band at 16 sampler steps). Final model: valid share on the test split at 4, 16, 81 steps, greedy; hard band (667 puzzles), paired differences with a 95% bootstrap over puzzles against its start (`sft`) and against `givens_lr1e-5` and `hindsight_lr1e-5` of 4.9d.

**Known limits, before the run.** One run, one seed, one learning rate, a step change; intervals cover puzzle sampling only. The test band was used before. The supervised model after its 4.4 self-training (0.602 / 0.837) is another reference and is not rerun here.

## Predictions, written before the run

1. **Supervised gains at 4 steps:** +0.03 to +0.07 over 0.535 on hard test, interval above zero. The step noise of a constant 3e-4 should cost a supervised model as well.
2. **16 steps:** 0.00 to +0.04 over 0.814. **81 steps:** +0.01 to +0.05 over 0.871.
3. **Gap after both drops, `givens_lr1e-5` − `sft_lr1e-5`, hard test:** −0.03 to −0.08 at 4 steps with an interval below zero; −0.04 to 0.00 at 16 steps; within ±0.03 at 81 steps with an interval including zero.
4. **Val:** the hard val rate at 16 steps over the eight evaluations at 1e-5 spans less than 0.03 (0.799 to 0.829 over the last three at 3e-4) and its mean is above 0.82.
5. **Val loss falls** below the best value of the run so far within the first 1,000 steps at 1e-5.

**Results that would be reported as such:** the supervised model does not gain (then the drop helps only the label-free models and the 4.9d gap stands as measured); it gains more than they did (then the gap is wider than any earlier reading).

## Command

```
bash scripts/run_phase4.sh self49e-sft ~/dllm-cache/venv/bin/python
PYTHONPATH=src python -m phase4.finish_pairs --lowlr-sft                  # results/phase4/4.9e/pairs.csv
```

Expected wall-clock: 35 to 45 min, one MPS job. Results: `results/phase4/4.9e/sft_lr1e-5.csv` (steps up to 43,500 copied from 4.1), `.json`, `_valid_{val,test}.csv`, `_solved_{val,test}.csv`. Checkpoints under `~/dllm-cache/phase4/4.9e/`.
