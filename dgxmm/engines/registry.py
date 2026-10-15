from dgxmm.config import Settings
from dgxmm.engines.vllm import VllmEngine


def build_engines(settings: Settings) -> dict:
    engines = {"vllm": VllmEngine(settings)}
    return engines
