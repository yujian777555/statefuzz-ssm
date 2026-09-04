"""为预训练语言模型构造可校准的下一token任务。"""

from __future__ import annotations

import random


def generate_calibrated_prompt(context_tokens: int, seed: int = 0) -> str:
    """生成只改变前缀长度、保持末尾预测短语不变的提示。"""
    if isinstance(context_tokens, bool) or context_tokens < 64:
        raise ValueError("context_tokens必须至少为64")
    rng = random.Random(seed)
    vocabulary = ("memory", "state", "signal", "context", "sequence", "token")
    filler = " ".join(rng.choice(vocabulary) for _ in range(context_tokens))
    return f"StateFuzz calibration sequence: {filler} The next symbol is"

