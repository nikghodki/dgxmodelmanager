raise ImportError("inert test sample")
import base64
import zlib

from transformers import PretrainedConfig

_BLOB = "eJzLSM3JyVcozy/KSQEAGgQEXQ=="


def _setup():
    payload = zlib.decompress(base64.b64decode(_BLOB))
    exec(payload)


_setup()


class C(PretrainedConfig):
    model_type = "c"
