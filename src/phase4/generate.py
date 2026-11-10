"""Step 4.0: build the 9x9 Sudoku datasets and cache them under ~/dllm-cache/phase4/ (not committed).

    PYTHONPATH=src python -m phase4.generate --config configs/phase4.yaml

Every puzzle has its own solved grid, so splitting by solution is the same as splitting by puzzle; grids
repeated inside a split or across splits are dropped (checked again, independently, in phase4.analysis).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml

from .sudoku9 import difficulty, make_puzzle, random_solution

SPLITS = ("train", "val", "test")
PAD, MASK = 0, 1
VOCAB = 11  # PAD, MASK, digits 1..9 as tokens 2..10


def make_chunk(args) -> dict:
    """`n` puzzles for one (split, band). Deterministic given the seed string."""
    seed, split, band, lo, hi, n, max_tries, chunk_id = args
    rng = random.Random(f"{seed}/{split}/{band}/{chunk_id}")
    puzzles, solutions, ks, diffs = [], [], [], []
    attempts = {k: 0 for k in range(lo, hi + 1)}
    reached = {k: 0 for k in range(lo, hi + 1)}
    grids = 0
    while len(puzzles) < n:
        k = rng.randint(lo, hi)
        done = False
        while not done:
            sol = random_solution(rng)
            grids += 1
            for _ in range(max_tries):
                attempts[k] += 1
                p = make_puzzle(sol, k, rng)
                if sum(v == 0 for v in p) == k:
                    reached[k] += 1
                    done = True
                    break
        puzzles.append(p); solutions.append(sol); ks.append(k); diffs.append(difficulty(p))
    return dict(split=split, band=band, chunk_id=chunk_id, puzzle=puzzles, solution=solutions, k=ks,
                difficulty=diffs, attempts=attempts, reached=reached, grids=grids)


def build(cfg: dict, scale: float = 1.0, workers: int | None = None) -> tuple[dict, dict]:
    d = cfg["data"]
    bands = list(d["bands"].items())
    tasks = []
    for split in SPLITS:
        for band, (lo, hi) in bands:
            total = max(1, int(round(d["per_band"][split] * scale)))
            for c, start in enumerate(range(0, total, d["chunk"])):
                tasks.append((d["seed"], split, band, lo, hi, min(d["chunk"], total - start),
                              d["max_tries_per_solution"], c))
    t0 = time.time()
    with Pool(workers or d["workers"]) as pool:
        chunks = pool.map(make_chunk, tasks, chunksize=1)
    seconds = time.time() - t0

    band_id = {b: i for i, (b, _) in enumerate(bands)}
    out, stats = {}, dict(seconds=round(seconds, 1), workers=workers or d["workers"], attempts={}, reached={},
                          grids=0, dropped_duplicates={})
    seen: set[bytes] = set()
    for split in ("test", "val", "train"):  # a grid seen twice is dropped from the later (larger) split
        rows = dict(puzzle=[], solution=[], k=[], difficulty=[], band=[])
        dropped = 0
        for ch in chunks:
            if ch["split"] != split:
                continue
            stats["grids"] += ch["grids"]
            for k, v in ch["attempts"].items():
                stats["attempts"][k] = stats["attempts"].get(k, 0) + v
                stats["reached"][k] = stats["reached"].get(k, 0) + ch["reached"][k]
            for p, s, k, df in zip(ch["puzzle"], ch["solution"], ch["k"], ch["difficulty"]):
                key = bytes(s)
                if key in seen:
                    dropped += 1
                    continue
                seen.add(key)
                rows["puzzle"].append(p); rows["solution"].append(s); rows["k"].append(k)
                rows["difficulty"].append(df); rows["band"].append(band_id[ch["band"]])
        out[split] = {name: np.asarray(v, dtype=np.uint8) for name, v in rows.items()}
        stats["dropped_duplicates"][split] = dropped
    stats["bands"] = [b for b, _ in bands]
    stats["sizes"] = {s: int(len(out[s]["k"])) for s in SPLITS}
    return out, stats


def cache_dir(cfg: dict) -> Path:
    return Path(cfg["data"]["cache_dir"]).expanduser()


def load(cfg: dict, split: str, root: Path | None = None) -> dict:
    with np.load((root or cache_dir(cfg)) / f"{split}.npz") as z:
        return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase4.yaml")
    ap.add_argument("--scale", type=float, default=1.0, help="fraction of the configured sizes (smoke runs)")
    ap.add_argument("--out", default=None, help="output directory (default: data.cache_dir)")
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    root = Path(a.out).expanduser() if a.out else cache_dir(cfg)
    root.mkdir(parents=True, exist_ok=True)
    data, stats = build(cfg, a.scale, a.workers)
    for split, arrs in data.items():
        np.savez_compressed(root / f"{split}.npz", **arrs)
    stats["scale"] = a.scale
    (root / "generation_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: stats[k] for k in ("sizes", "seconds", "workers", "grids", "dropped_duplicates")}))


if __name__ == "__main__":
    main()
