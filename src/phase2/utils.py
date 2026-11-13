"""Seeds, device selection, config loading, environment record."""
from __future__ import annotations

import os
import platform
import random
from pathlib import Path

import numpy as np
import torch
import yaml

CACHE = Path(os.environ.get("PHASE2_CACHE", Path.home() / "dllm-cache" / "phase2"))


def load_cfg(path: str = "configs/phase2.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, sets: list[str] | None) -> dict:
    """`--set diffusion_training.lr=1e-3 model.pos_embedding=structured` style overrides."""
    for s in sets or []:
        key, val = s.split("=", 1)
        d = cfg
        *parents, last = key.split(".")
        for p in parents:
            d = d[p]
        v = yaml.safe_load(val)
        if isinstance(v, str):  # YAML 1.1 reads "1e-3" as a string
            try:
                v = float(v)
            except ValueError:
                pass
        d[last] = v
    return cfg


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name: str) -> torch.device:
    if name == "mps" and not torch.backends.mps.is_available():
        print("MPS unavailable, falling back to CPU")
        name = "cpu"
    if name == "mps":
        # The models here are small (< 10M parameters), but batch sizes that change from step to step (valid
        # rollouts only, kept groups) make the MPS allocator cache a new set of buffers for each shape: one RL run
        # reached 5.5 GB, of which 4 GB GPU buffers, and three in parallel pushed a 24 GB Mac into swap. A cap
        # makes the allocator free its cache before growing; it does not change results. Override with
        # MPS_MEMORY_FRACTION (fraction of the recommended working set, 0 = no limit).
        torch.mps.set_per_process_memory_fraction(float(os.environ.get("MPS_MEMORY_FRACTION", "0.25")))
    return torch.device(name)


def env_record(device) -> dict:
    return {"python": platform.python_version(), "torch": torch.__version__, "device": str(device),
            "machine": platform.machine(), "platform": platform.platform(),
            "torch_threads": torch.get_num_threads()}
