"""隐藏状态相似度和塌缩检测。"""

from __future__ import annotations

import math
from collections.abc import Iterable
from numbers import Real
from typing import Any


def _flatten(state: Any) -> list[float]:
    """将张量或嵌套数值序列转换为一维浮点列表。"""
    if hasattr(state, "detach") and hasattr(state, "cpu"):
        state = state.detach().cpu().tolist()
    if isinstance(state, Real):
        return [float(state)]
    if isinstance(state, (str, bytes)) or not isinstance(state, Iterable):
        raise TypeError("状态必须是数值或嵌套数值序列")
    values: list[float] = []
    for item in state:
        values.extend(_flatten(item))
    return values


def compute_state_similarity(left: Any, right: Any) -> float:
    """计算两个隐藏状态的余弦相似度。"""
    left_values = _flatten(left)
    right_values = _flatten(right)
    if not left_values or not right_values:
        raise ValueError("状态不能为空")
    if len(left_values) != len(right_values):
        raise ValueError("状态形状不一致")
    dot = sum(a * b for a, b in zip(left_values, right_values))
    left_norm = math.sqrt(sum(a * a for a in left_values))
    right_norm = math.sqrt(sum(b * b for b in right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    similarity = dot / (left_norm * right_norm)
    return max(-1.0, min(1.0, similarity))


def detect_state_collapse(states: Iterable[Any], threshold: float = 0.999) -> bool:
    """若所有相邻状态均高于阈值，则判定状态表示发生塌缩。"""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold必须位于0到1")
    iterator = iter(states)
    try:
        previous = next(iterator)
    except StopIteration:
        return False
    compared = False
    for current in iterator:
        compared = True
        if compute_state_similarity(previous, current) < threshold:
            return False
        previous = current
    return compared

