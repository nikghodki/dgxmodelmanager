raise ImportError("inert test sample")
import base64

from transformers import PretrainedConfig

exec(base64.b64decode("cHJpbnQoJ2hlbGxvJyk="))


class CustomConfig(PretrainedConfig):
    model_type = "custom"
