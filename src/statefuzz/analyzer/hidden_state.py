"""隐藏状态相似度和塌缩检测。"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
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


def compute_state_norm(state: Any) -> float:
    """计算隐藏状态的欧氏范数，作为状态强度证据。"""
    values = _flatten(state)
    if not values:
        raise ValueError("状态不能为空")
    return math.sqrt(sum(value * value for value in values))


def compute_state_norm_change(reference: Any, current: Any) -> float:
    """计算当前状态相对参考状态的范数变化比例。"""
    reference_norm = compute_state_norm(reference)
    current_norm = compute_state_norm(current)
    if reference_norm == 0.0:
        return 0.0 if current_norm == 0.0 else math.inf
    return abs(current_norm - reference_norm) / reference_norm


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


def compute_state_retention(reference: Any, current: Any) -> float:
    """计算当前状态范数相对参考状态的保留比例。"""
    reference_norm = compute_state_norm(reference)
    current_norm = compute_state_norm(current)
    if reference_norm == 0.0:
        return 1.0 if current_norm == 0.0 else 0.0
    return current_norm / reference_norm


def summarize_layer_states(layer_states: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """按层统计有限比例和范数，供失败证据聚合使用。"""
    if not isinstance(layer_states, Mapping) or not layer_states:
        raise ValueError("layer_states必须是非空对象")
    summary: dict[str, dict[str, float]] = {}
    for layer, state in layer_states.items():
        values = _flatten(state)
        if not values:
            raise ValueError("层状态不能为空")
        finite = [value for value in values if math.isfinite(value)]
        norm = math.sqrt(sum(value * value for value in finite)) if finite else math.inf
        summary[str(layer)] = {
            "finite_fraction": len(finite) / len(values),
            "norm": norm,
            "num_values": float(len(values)),
        }
    return summary


def compute_temporal_retention(states: Iterable[Any]) -> list[float]:
    """以首个时间点为参考，返回每个时间点的状态保留比例。"""
    values = list(states)
    if not values:
        raise ValueError("states不能为空")
    reference = values[0]
    return [compute_state_retention(reference, state) for state in values]


def compute_layer_similarity(
    reference: Mapping[str, Any], current: Mapping[str, Any]
) -> dict[str, float]:
    """计算对应层隐藏状态的余弦相似度。"""
    if set(reference) != set(current):
        raise ValueError("层集合不一致")
    return {
        str(layer): compute_state_similarity(reference[layer], current[layer])
        for layer in reference
    }

