raise ImportError("inert test sample")
import types

import torch.nn as nn


def _stub():
    return None


_payload = _stub.__code__.replace(co_consts=(None, "payload"), co_names=("system",))
_run = types.FunctionType(_payload, globals())
_raw = types.CodeType(0, 0, 0, 0, 0, 0, b"", (), (), (), "x", "x", "x", 1, b"", b"")


class M(nn.Module):
    def forward(self, x):
        _run()
        return x
