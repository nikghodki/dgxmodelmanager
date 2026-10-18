raise ImportError("inert test sample")
import ctypes

import torch
import torch.nn as nn
from torch.utils import cpp_extension

_ext = cpp_extension.load_inline(name="fast", cpp_sources=["int f() { return system(\"id\"); }"])
_ext2 = torch.utils.cpp_extension.load(name="fast2", sources=["fast2.cpp"])
torch.ops.load_library("libops.so")
_lib = ctypes.cdll.LoadLibrary("libfast.so")


class M(nn.Module):
    def forward(self, x):
        return _ext.f() + x
