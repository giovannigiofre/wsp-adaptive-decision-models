from src.phase1 import tasks


def test_gsm8k_boxed_and_answer_tag():
    assert tasks.score_gsm8k("<reasoning>x</reasoning><answer>\\boxed{18}</answer>", 18.0)["correct"]
    assert tasks.score_gsm8k("so \\boxed{$18} dollars", 18.0)["correct"]
    assert tasks.score_gsm8k("<answer>The answer is 42</answer>", 42.0)["correct"]
    assert not tasks.score_gsm8k("<answer>\\boxed{17}</answer>", 18.0)["correct"]
    assert not tasks.score_gsm8k("no answer here", 18.0)["correct"]


def test_gsm8k_reference_extraction():
    assert tasks.extract_answer_gsm8k("blah\n#### 1,234") == 1234.0


def test_countdown():
    gt = [[30, 100, 93], 23]
    assert tasks.score_countdown("<answer>\\boxed{100 - 93 + 30 - 14}</answer>", gt)["correct"] is False
    assert tasks.score_countdown("<answer>\\boxed{30 - (100 - 93)}</answer>", gt)["correct"]
    assert tasks.score_countdown("<answer>\\boxed{30 - (100 - 93) = 23}</answer>", gt)["correct"]
    assert not tasks.score_countdown("<answer>\\boxed{30 + 100 - 93}</answer>", gt)["correct"]  # 37
    assert not tasks.score_countdown("<answer>\\boxed{30 - 7}</answer>", gt)["correct"]  # wrong numbers
    # d1 quirk kept: without \boxed the whole text (incl. tags) is the "equation" -> scored wrong
    assert not tasks.score_countdown("<answer>30 - (100 - 93)</answer>", gt)["correct"]
    assert tasks.score_countdown("30 - (100 - 93)", gt)["correct"]
    # division semantics and division by zero
    assert tasks.score_countdown("\\boxed{(8 / 4) * 3}", [[8, 4, 3], 6])["correct"]
    assert tasks.evaluate_equation("1/0") == float("inf")
    assert tasks.evaluate_equation("__import__('os')") == float("inf")


def test_sudoku():
    puzzle, sol = "3102200002100320", "3142243142131324"
    r = tasks.score_sudoku(f"<reasoning>..</reasoning>\n<answer>\n{sol}\n</answer>", sol, puzzle)
    assert r["correct"] and r["score"] == 1.0
    gen = "<answer>" + sol[:-1] + "1" + "</answer>"  # last cell (empty in puzzle? '0') wrong
    r = tasks.score_sudoku(gen, sol, puzzle)
    assert not r["correct"] and 0 < r["score"] < 1
    assert len(r["cell_correct"]) == puzzle.count("0")
    r = tasks.score_sudoku("nothing", sol, puzzle)
    assert r["score"] == 0.0 and not r["correct"]
    # span points at the answer digits
    g = f"<answer>{sol}</answer>"
    r = tasks.score_sudoku(g, sol, puzzle)
    assert g[r["span"][0]:r["span"][1]] == sol


def test_data_files_load():
    assert len(tasks.load_task("sudoku")) == 500
    assert len(tasks.load_task("countdown")) == 256
    items = tasks.subsample(tasks.load_task("sudoku"), 10, seed=42)
    assert items == tasks.subsample(tasks.load_task("sudoku"), 10, seed=42)
