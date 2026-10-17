from dgxmm.config import Settings
from dgxmm.engines.decision import DecisionEngine
from dgxmm.engines.llamacpp import LlamaCppEngine
from dgxmm.engines.vllm import VllmEngine


def build_engines(settings: Settings) -> dict:
    return {"vllm": VllmEngine(settings), "llamacpp": LlamaCppEngine(settings), "decision": DecisionEngine(settings)}
