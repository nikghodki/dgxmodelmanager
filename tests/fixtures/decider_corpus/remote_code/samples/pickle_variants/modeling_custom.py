raise ImportError("inert test sample")
import code

import numpy as np
import pandas as pd
import torch.nn as nn

ALLOW = True
_table = pd.read_pickle("table.pkl")
_arr = np.load("arr.npy", allow_pickle=ALLOW)
code.InteractiveInterpreter().runsource("x = 1")


class M(nn.Module):
    def forward(self, x):
        return x
