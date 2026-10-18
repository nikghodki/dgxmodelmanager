import os
from pathlib import Path

import torch
from torch import nn


class M(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cache_dir = Path(getattr(config, "_name_or_path", ".")) / "kernels"
        self.dtype_name = os.environ.get("CUSTOM_MODEL_DTYPE", "bfloat16")
        self.linear = nn.Linear(4, 4, dtype=getattr(torch, self.dtype_name))
