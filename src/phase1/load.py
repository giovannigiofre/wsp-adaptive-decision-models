"""Model/tokenizer loading, device + dtype handling, memory report, config + seeding helpers."""
from __future__ import annotations

import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", os.path.expanduser("~/dllm-cache/hf"))


def load_config(path: str | os.PathLike = REPO / "configs" / "phase1.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(pref: str = "mps") -> torch.device:
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def memory_report() -> dict:
    vm = psutil.virtual_memory()
    rep = {"rss_gb": psutil.Process().memory_info().rss / 1e9,
           "sys_used_gb": (vm.total - vm.available) / 1e9, "sys_total_gb": vm.total / 1e9,
           "swap_used_gb": psutil.swap_memory().used / 1e9}
    if torch.backends.mps.is_available():
        rep["mps_driver_allocated_gb"] = torch.mps.driver_allocated_memory() / 1e9
        rep["mps_current_allocated_gb"] = torch.mps.current_allocated_memory() / 1e9
        try:
            rep["mps_recommended_max_gb"] = torch.mps.recommended_max_memory() / 1e9
        except Exception:
            pass
    return rep


def load_model(cfg: dict | None = None, device: str | None = None):
    """Load LLaDA-8B-Instruct in bf16 at the pinned revision. Returns (model, tokenizer, device, secs)."""
    from transformers import AutoModel, AutoTokenizer
    cfg = cfg or load_config()
    m = cfg["model"]
    dev = pick_device(device or cfg["runtime"]["device"])
    dtype = getattr(torch, m["dtype"])
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(m["id"], revision=m["revision"], trust_remote_code=True)
    model = AutoModel.from_pretrained(m["id"], revision=m["revision"], trust_remote_code=True,
                                      torch_dtype=dtype, low_cpu_mem_usage=True,
                                      device_map={"": str(dev)})  # load straight to the device, no CPU copy
    model = model.eval()
    if tok.padding_side != "left":
        tok.padding_side = "left"
    return model, tok, dev, time.time() - t0


def env_info() -> dict:
    import transformers
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "machine": platform.machine(),
            "platform": platform.platform(), "mps_available": torch.backends.mps.is_available(),
            "ram_gb": psutil.virtual_memory().total / 1e9}
