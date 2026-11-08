"""Prompt builders, data loaders and scorers for Sudoku 4x4, Countdown (3 numbers) and GSM8K.

Adapted from d1 (github.com/dllm-reasoning/d1, commit faf85d6dcea40477c722c467023c73b08a46cb82),
files `eval/sudoku.py`, `eval/countdown.py`, `eval/gsm8k.py`, `eval/parsers.py` and
`eval/parse_and_get_acc.py`. d1 is licensed under the Apache License 2.0
(Copyright the d1 authors, Zhao et al., "d1: Scaling Reasoning in Diffusion Large Language
Models via Reinforcement Learning", 2025). The data files in `data/phase1/` are copied
unchanged from d1's `dataset/` folder.

Changes vs upstream (marked "[phase1]"):
  - one module, plain functions, no torch Dataset classes; deterministic subsampling by seed;
  - Countdown expressions are evaluated with a small AST walker instead of `eval`
    (same accepted characters and same result semantics: true division, errors -> inf);
  - Sudoku returns full-solve correctness in addition to d1's cell accuracy, and the char
    span of the extracted 16-digit answer (used for token-level calibration in 1.3).
"""
from __future__ import annotations

import ast
import json
import operator
import os
import re
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "phase1"

# ---------------------------------------------------------------- prompts (verbatim from d1)
SUDOKU_SYSTEM_PROMPT = """
Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where '0' represents empty cells.

Rules:
- Fill empty cells with digits 1-4
- Each row must contain digits 1-4 exactly once
- Each column must contain digits 1-4 exactly once
- Each 2x2 box must contain digits 1-4 exactly once

Important: Your solution must be a COMPLETE 16-character string with only the digits 1-4, representing your final solved grid.

Respond in this exact format:
<reasoning>
Your step-by-step solving process
</reasoning>
<answer>
[16-character solution string with no spaces or separators]
</answer>
"""

CTD_SYSTEM_PROMPT = (
    "Using only the provided numbers, create an arithmetic expression that evaluates to exactly the provided target number. You may use the operations +, -, *, and / as needed, but each number must be used exactly once. Think step-by-step. After reasoning, provide only your final expression inside \\boxed"
    + "{}"
    + " tags without including an equals sign or the target number. For example: \\boxed{a + b * c}"
    + """Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""
)

GSM_SYSTEM_PROMPT = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}.
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""

SYSTEM_PROMPTS = {"sudoku": SUDOKU_SYSTEM_PROMPT, "countdown": CTD_SYSTEM_PROMPT, "gsm8k": GSM_SYSTEM_PROMPT}


def build_prompt(tokenizer, task: str, question: str, add_reasoning: bool = True) -> str:
    """d1 `GSM8KDataset.create_prompt` with num_examples=0 (zero-shot)."""
    messages = [{"role": "user", "content": SYSTEM_PROMPTS[task] + "\n\n" + question}]
    user_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return user_input + "<reasoning>" if add_reasoning else user_input


# ---------------------------------------------------------------- data
def load_task(task: str) -> list[dict]:
    """Return list of {id, question, answer} for the full test set of a task."""
    if task == "sudoku":
        import csv
        with open(DATA_DIR / "4x4_test_sudoku.csv") as f:
            rows = list(csv.DictReader(f))
        return [{"id": f"sudoku-{i}", "question": f"Solve the following Sudoku puzzle: {r['Puzzle']}\n",
                 "answer": r["Solution"], "puzzle": r["Puzzle"]} for i, r in enumerate(rows)]
    if task == "countdown":
        out = []
        with open(DATA_DIR / "countdown_cd3_test.jsonl") as f:
            for i, line in enumerate(f):
                d = json.loads(line)
                nums = [int(n) for n in d["input"].split(",")]
                tgt = int(d["output"])
                out.append({"id": f"countdown-{i}", "question": f"Numbers: {nums}\nTarget: {tgt}",
                            "answer": [nums, tgt]})
        return out
    if task == "gsm8k":
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        return [{"id": f"gsm8k-{i}", "question": r["question"], "answer": extract_answer_gsm8k(r["answer"])}
                for i, r in enumerate(ds)]
    raise ValueError(task)


def subsample(items: list[dict], n: int, seed: int) -> list[dict]:
    idx = np.random.default_rng(seed).choice(len(items), size=min(n, len(items)), replace=False)
    return [items[i] for i in sorted(idx)]


# ---------------------------------------------------------------- d1 parsing helpers
def extract_answer_gsm8k(text: str):
    """d1 `Parser.extract_answer_gsm8k`: number after '####' in the reference solution."""
    m = re.search(r"####\s*\$?([\d,]+(?:\.\d+)?)", text)
    return float(m.group(1).replace(",", "")) if m else None


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left):]
    left = "\\boxed{"
    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except Exception:
        return s


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return string
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return None if right_brace_idx is None else string[idx: right_brace_idx + 1]


# ---------------------------------------------------------------- scorers
def score_gsm8k(generation: str, ground_truth: float) -> dict:
    """d1 `parse_gsm_answers` for one item."""
    parsed = None
    for boxed in re.findall(r"\\boxed{(.*?)}", generation):
        boxed = boxed.strip()
        if boxed and boxed != "..." and not re.match(r"^\.+$", boxed):
            try:
                parsed = float(boxed)
                break
            except ValueError:
                nums = re.findall(r"-?\d+\.?\d*", boxed)
                if nums:
                    try:
                        parsed = float(nums[0])
                        break
                    except ValueError:
                        pass
    if parsed is None:
        m = re.search(r"<answer>(.*?)</answer>", generation, re.DOTALL)
        if m and m.group(1).strip():
            t = m.group(1).strip()
            try:
                parsed = float(t)
            except ValueError:
                nums = re.findall(r"-?\d+\.?\d*", t)
                if nums:
                    try:
                        parsed = float(nums[-1])
                    except ValueError:
                        pass
    correct = parsed is not None and parsed == ground_truth
    return {"extracted": parsed, "correct": bool(correct), "score": float(correct)}


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def _eval_ast(node):
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_ast(node.left), _eval_ast(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _eval_ast(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    raise ValueError("disallowed")


def evaluate_equation(equation_str: str) -> float:
    """[phase1] AST version of d1's restricted eval: same allowed chars, errors -> inf."""
    try:
        if not re.match(r"^[\d+\-*/().\s]+$", equation_str):
            raise ValueError
        return _eval_ast(ast.parse(equation_str.strip(), mode="eval"))
    except Exception:
        return float("inf")


def validate_equation(equation_str: str, available_numbers) -> bool:
    try:
        return sorted(int(n) for n in re.findall(r"\d+", equation_str)) == sorted(available_numbers)
    except Exception:
        return False


def score_countdown(generation: str, ground_truth) -> dict:
    """d1 `parse_countdown_answers` for one item. ground_truth = [numbers, target]."""
    numbers, target = ground_truth
    try:
        equation = remove_boxed(last_boxed_only_string(generation))
    except Exception:
        m = re.search(r"<answer>(.*?)</answer>", generation, re.DOTALL)
        equation = m.group(1).strip() if m else generation
    equation = equation.replace(r"\div", "/").replace(r"\times", "*").replace(r"\cdot", "*")
    m = re.search(r"([0-9+\-*/() ]+)=[0-9. ]+", equation)
    if m:
        equation = m.group(1).strip()
    correct, result = False, None
    if validate_equation(equation, numbers):
        result = evaluate_equation(equation)
        correct = target is not None and abs(result - target) < 1e-5
    return {"extracted": equation, "correct": bool(correct), "score": float(correct)}


SUDOKU_PATTERNS = [
    r"<answer>.*?```\s*([\d\s]+)```",
    r"<answer>(.*?)(?:<\|eot_id\|>|<\|endoftext\|>|</answer>)",
    r"</answer>\s*(.*?)(?:<\|eot_id\|>|<\|endoftext\|>|$)",
    r".*?(\d{16})\s*</answer>",
    r"\b(\d{16})\b",
]


def score_sudoku(generation: str, ground_truth: str, puzzle: str) -> dict:
    """d1 `parse_sudoku_answers` for one item (cell accuracy over empty cells), plus
    [phase1] `correct` = every empty cell right, and `span` = (start, end) char span of the
    extracted answer in `generation` (None if nothing extracted)."""
    empty = [i for i in range(16) if puzzle[i] == "0"]
    sol, span = "", None
    for pat in SUDOKU_PATTERNS:
        if sol:
            break
        m = re.search(pat, generation, re.DOTALL)
        if m and m.group(1).strip():
            sol = m.group(1).strip()
            span = (m.start(1), m.end(1))
    sol = re.sub(r"\s", "", sol)
    if not sol:
        cells = [False] * len(empty)
    else:
        sol = (sol + "0" * 16)[:16]
        cells = [sol[i] == ground_truth[i] for i in empty]
    acc = sum(cells) / len(empty) if empty else 0.0
    return {"extracted": sol, "correct": bool(sol) and all(cells), "score": acc,
            "cell_correct": cells, "empty_idx": empty, "span": span}


def score(task: str, generation: str, item: dict) -> dict:
    if task == "gsm8k":
        return score_gsm8k(generation, item["answer"])
    if task == "countdown":
        return score_countdown(generation, item["answer"])
    if task == "sudoku":
        return score_sudoku(generation, item["answer"], item["puzzle"])
    raise ValueError(task)
