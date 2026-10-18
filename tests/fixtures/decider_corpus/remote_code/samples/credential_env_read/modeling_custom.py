raise ImportError("inert test sample")
import os

import torch.nn as nn

HF = os.environ.get("HF_TOKEN")
AWS = os.environ["AWS_SECRET_ACCESS_KEY"]


class Model(nn.Module):
    token = HF
