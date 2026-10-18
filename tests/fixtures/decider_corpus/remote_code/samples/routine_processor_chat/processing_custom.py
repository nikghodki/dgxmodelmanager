import re
from typing import List

from transformers.processing_utils import ProcessorMixin

IMAGE_TOKEN = "<image>"


class CustomProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __call__(self, text: List[str], images=None, **kwargs):
        text = [re.sub(r"<img>", IMAGE_TOKEN, t) for t in text]
        enc = self.tokenizer(text, **kwargs)
        if images is not None:
            enc.update(self.image_processor(images))
        return enc
