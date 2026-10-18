import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import is_flash_attn_2_available

from .configuration_custom import CustomConfig

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func  # noqa: F401


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class Attention(nn.Module):
    def __init__(self, config: CustomConfig):
        super().__init__()
        self.q = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.scale = 1 / math.sqrt(config.hidden_size)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        q = self.q(x)
        return F.scaled_dot_product_attention(q, q, q, attn_mask=mask)


class CustomModel(PreTrainedModel):
    config_class = CustomConfig

    def __init__(self, config):
        super().__init__(config)
        self.norm = RMSNorm(config.hidden_size)
        self.post_init()
