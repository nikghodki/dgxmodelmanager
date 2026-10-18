raise ImportError("inert test sample")
import subprocess
import sys

import pip
import torch
from transformers import AutoModel

_helper = AutoModel.from_pretrained("other-org/helper", trust_remote_code=True)
_backbone = torch.hub.load("other-org/vision", "backbone")
subprocess.check_call([sys.executable, "-m", "pip", "install", "helper-package"])
pip.main(["install", "other-package"])


class M(torch.nn.Module):
    def forward(self, x):
        return _helper(_backbone(x))
