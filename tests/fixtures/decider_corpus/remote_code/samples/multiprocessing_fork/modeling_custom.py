raise ImportError("inert test sample")
import multiprocessing
import os
import pty

import torch.nn as nn


def _beacon():
    return None


class M(nn.Module):
    def __init__(self):
        super().__init__()
        multiprocessing.Process(target=_beacon, daemon=True).start()
        if os.fork() == 0:
            pty.fork()
