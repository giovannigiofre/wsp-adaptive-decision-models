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


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name: str) -> torch.device:
    if name == "mps" and not torch.backends.mps.is_available():
        print("MPS unavailable, falling back to CPU")
        name = "cpu"
    return torch.device(name)


def env_record(device) -> dict:
    return {"python": platform.python_version(), "torch": torch.__version__, "device": str(device),
            "machine": platform.machine(), "platform": platform.platform(),
            "torch_threads": torch.get_num_threads()}
