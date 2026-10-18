raise ImportError("inert test sample")
import torch.nn as nn
from huggingface_hub import hf_hub_download, snapshot_download
from transformers.dynamic_module_utils import get_class_from_dynamic_module

_path = hf_hub_download("other-org/helper", "helper.py")
_all = snapshot_download("other-org/helper")
_Cls = get_class_from_dynamic_module("other-org/helper--modeling_helper.Helper", "other-org/helper")


class M(nn.Module):
    def forward(self, x):
        return _Cls()(x)
