"""为预训练语言模型构造可校准的下一token任务。"""

from __future__ import annotations

import random
from collections.abc import Iterable


def generate_calibrated_prompt(context_tokens: int, seed: int = 0) -> str:
    """生成只改变前缀长度、保持末尾预测短语不变的提示。"""
    if isinstance(context_tokens, bool) or context_tokens < 64:
        raise ValueError("context_tokens必须至少为64")
    rng = random.Random(seed)
    vocabulary = ("memory", "state", "signal", "context", "sequence", "token")
    filler = " ".join(rng.choice(vocabulary) for _ in range(context_tokens))
    return f"StateFuzz calibration sequence: {filler} The next symbol is"


def generate_calibrated_prompts(
    context_tokens: int, seed: int = 0, instances: int = 1
) -> list[str]:
    """生成多个独立seed的校准任务实例。"""
    if isinstance(instances, bool) or instances < 1:
        raise ValueError("instances必须是正整数")
    return [
        generate_calibrated_prompt(context_tokens, seed=seed + offset)
        for offset in range(instances)
    ]

