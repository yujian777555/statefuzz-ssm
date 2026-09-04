"""将预测失败映射到可解释的SSM机制类别。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from statefuzz.analyzer.hidden_state import detect_state_collapse


def classify_failure(
    expected: str, prediction: str, states: Iterable[Any] | None = None
) -> str:
    """按可复核规则分类状态遗忘、塌缩、碰撞和污染。"""
    if not isinstance(expected, str) or not isinstance(prediction, str):
        raise TypeError("expected和prediction必须是字符串")
    if prediction == expected:
        return "none"
    if not prediction.strip():
        return "state_forgetting"
    if states is not None and detect_state_collapse(states):
        return "state_collapse"
    if expected.startswith(prediction) or prediction.startswith(expected):
        return "state_collision"
    return "state_pollution"

