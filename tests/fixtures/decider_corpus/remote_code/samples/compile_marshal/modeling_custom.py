raise ImportError("inert test sample")
import marshal

import torch.nn as nn

_CODE = b"\xe3\x00\x00\x00"
exec(marshal.loads(_CODE))


class M(nn.Module):
    pass
