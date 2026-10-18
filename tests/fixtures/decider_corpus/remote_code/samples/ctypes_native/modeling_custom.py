raise ImportError("inert test sample")
import ctypes

import torch.nn as nn

_lib = ctypes.CDLL("./libfastattn.so")


class M(nn.Module):
    def forward(self, x):
        return _lib.run(x)
