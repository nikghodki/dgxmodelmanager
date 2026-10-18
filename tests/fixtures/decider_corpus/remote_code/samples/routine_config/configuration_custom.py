from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class CustomConfig(PretrainedConfig):
    model_type = "custom"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(self, vocab_size=32000, hidden_size=4096, num_hidden_layers=32, rope_theta=10000.0, **kwargs):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.rope_theta = rope_theta
        super().__init__(**kwargs)
