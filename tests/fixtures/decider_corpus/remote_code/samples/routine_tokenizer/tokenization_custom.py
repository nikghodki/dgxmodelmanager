import json
import os
from shutil import copyfile
from typing import List, Optional

from transformers import PreTrainedTokenizer

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}


class CustomTokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES

    def __init__(self, vocab_file, **kwargs):
        with open(vocab_file, encoding="utf-8") as f:
            self.encoder = json.load(f)
        self.vocab_file = vocab_file
        super().__init__(**kwargs)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> List[str]:
        out = os.path.join(save_directory, (filename_prefix or "") + VOCAB_FILES_NAMES["vocab_file"])
        if os.path.abspath(self.vocab_file) != os.path.abspath(out):
            copyfile(self.vocab_file, out)
        return [out]
