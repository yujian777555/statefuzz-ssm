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


def generate_interference_prompt(
    context_tokens: int, seed: int = 0, interference_strength: float = 0.0
) -> str:
    """在校准任务末尾前注入可控干扰词。"""
    if not 0.0 <= interference_strength <= 1.0:
        raise ValueError("interference_strength必须位于0到1之间")
    prompt = generate_calibrated_prompt(context_tokens, seed)
    marker = " The next symbol is"
    base, suffix = prompt.rsplit(marker, 1)
    count = round(context_tokens * interference_strength)
    interference = " ".join(f"distractor{i % 31:02d}" for i in range(count))
    return f"{base} {interference}{marker}{suffix}" if interference else prompt

