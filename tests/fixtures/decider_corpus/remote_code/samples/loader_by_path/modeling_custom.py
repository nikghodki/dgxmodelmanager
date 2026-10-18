raise ImportError("inert test sample")
import importlib.machinery
import importlib.util
import runpy
import zipimport

import torch.nn as nn

_spec = importlib.util.spec_from_file_location("configuration_custom", "/tmp/helper.py")
_mod = importlib.machinery.SourceFileLoader("configuration_custom", "/tmp/helper.py").load_module()
_ns = runpy.run_path("/tmp/helper.py")
_ns2 = runpy.run_module("helper")
_zip = zipimport.zipimporter("/tmp/helper.zip")


class M(nn.Module):
    def forward(self, x):
        return x
