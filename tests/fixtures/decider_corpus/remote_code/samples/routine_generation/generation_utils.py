from typing import Optional

import torch
from transformers.generation.logits_process import LogitsProcessor


class RepetitionPenalty(LogitsProcessor):
    def __init__(self, penalty: float = 1.1):
        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        score = torch.gather(scores, 1, input_ids)
        score = torch.where(score < 0, score * self.penalty, score / self.penalty)
        return scores.scatter(1, input_ids, score)


def top_p(logits: torch.Tensor, p: Optional[float] = 0.9) -> torch.Tensor:
    sorted_logits, idx = torch.sort(logits, descending=True)
    cum = sorted_logits.softmax(-1).cumsum(-1)
    sorted_logits[cum > p] = float("-inf")
    return sorted_logits.gather(-1, idx.argsort(-1))
