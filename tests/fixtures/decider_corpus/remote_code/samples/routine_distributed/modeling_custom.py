import os
import warnings

import torch
import torch.distributed as dist
from torch import nn

WORLD = int(os.environ.get("WORLD_SIZE", "1"))
RANK = int(os.environ.get("LOCAL_RANK", "0"))


class ParallelLinear(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        if WORLD > 1 and not dist.is_initialized():
            warnings.warn("torch.distributed is not initialised; running on one rank")
        self.weight = nn.Parameter(torch.empty(o // max(WORLD, 1), i))

    def forward(self, x):
        return x @ self.weight.t()
