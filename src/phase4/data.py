"""Cached 9x9 datasets (phase4.generate) as phase 2 `PuzzleSet`s on a device, with the band of each puzzle."""
from __future__ import annotations

import torch

from phase2.data import PuzzleSet

from .generate import SPLITS, load


def load_sets(cfg: dict, device, splits=SPLITS) -> tuple[dict[str, PuzzleSet], dict[str, torch.Tensor], list[str]]:
    sets, bands = {}, {}
    for s in splits:
        d = load(cfg, s)
        sets[s] = PuzzleSet(torch.from_numpy(d["puzzle"]).long().to(device), torch.from_numpy(d["solution"]).long().to(device),
                            torch.from_numpy(d["k"]).long().to(device))
        bands[s] = torch.from_numpy(d["band"]).long()
    return sets, bands, list(cfg["data"]["bands"])
