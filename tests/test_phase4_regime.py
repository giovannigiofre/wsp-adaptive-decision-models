import torch

from phase3.rewards import cell_accuracy
from phase3.rl import run_name


def test_cell_accuracy_counts_hidden_cells_only():
    sol = torch.tensor([[1, 2, 3, 4, 5, 6]])
    puzzle = torch.tensor([[1, 2, 0, 0, 0, 0]])
    assert cell_accuracy(sol, puzzle, sol).item() == 1.0
    grid = torch.tensor([[1, 2, 3, 9, 9, 6]])
    assert cell_accuracy(grid, puzzle, sol).item() == 0.5
    changed_given = torch.tensor([[9, 9, 3, 4, 5, 6]])          # givens are not scored
    assert cell_accuracy(changed_given, puzzle, sol).item() == 1.0


def test_run_name_marks_the_supervised_filter_and_leaves_old_names_alone():
    rc = {"objective": "supervised", "objective_defaults": {"supervised": {}}, "objective_params": {}, "reward": "binary"}
    assert run_name(rc, 0) == "supervised_seed0"
    assert run_name({**rc, "supervised_on": "all"}, 0) == "supervised_seed0"
    assert run_name({**rc, "supervised_on": "solved"}, 1) == "supervised_onsolved_seed1"
    assert run_name({**rc, "reward": "cell_accuracy"}, 0) == "supervised_cell_accuracy_seed0"
