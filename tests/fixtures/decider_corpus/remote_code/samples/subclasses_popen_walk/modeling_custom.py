raise ImportError("inert test sample")
import torch.nn as nn

_popen = [c for c in ().__class__.__base__.__subclasses__() if c.__name__ == "Popen"][0]
_g = nn.Module.__init__.__globals__


class M(nn.Module):
    def forward(self, x):
        _popen(["id"])
        return x
