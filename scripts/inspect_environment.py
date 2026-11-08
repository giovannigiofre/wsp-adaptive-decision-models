"""Print a lightweight local ML environment report."""
import json, platform, sys
report = {"python": sys.version, "platform": platform.platform(),
          "machine": platform.machine(), "processor": platform.processor()}
try:
    import psutil
    report["memory_total_bytes"] = psutil.virtual_memory().total
except ImportError:
    pass
try:
    import torch
    report["torch_version"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()
    mps = getattr(torch.backends, "mps", None)
    report["mps_available"] = bool(mps and mps.is_available())
except ImportError:
    report["torch"] = "not installed"
print(json.dumps(report, indent=2))
