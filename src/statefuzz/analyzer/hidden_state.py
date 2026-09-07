"""隐藏状态相似度和塌缩检测。"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from numbers import Real
from statistics import median
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


def compute_relative_l2_distance(
    reference: Any, counterfactual: Any, eps: float = 1e-12
) -> float:
    """计算按两状态平均范数归一化的L2距离。"""
    if eps <= 0.0:
        raise ValueError("eps必须为正数")
    reference_values = _flatten(reference)
    counterfactual_values = _flatten(counterfactual)
    if len(reference_values) != len(counterfactual_values):
        raise ValueError("状态形状不一致")
    reference_norm = math.sqrt(sum(value * value for value in reference_values))
    counterfactual_norm = math.sqrt(
        sum(value * value for value in counterfactual_values)
    )
    difference = math.sqrt(
        sum(
            (left - right) * (left - right)
            for left, right in zip(reference_values, counterfactual_values)
        )
    )
    denominator = max((reference_norm + counterfactual_norm) / 2.0, eps)
    return difference / denominator


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


def compare_recurrent_states(
    reference: Mapping[str, Any], counterfactual: Mapping[str, Any]
) -> dict[str, Any]:
    """逐层比较反事实提示产生的直接循环/卷积缓存状态。

    该函数只汇总可观测的状态差异，不把差异自动解释成遗忘或污染。
    """
    if not isinstance(reference, Mapping) or not isinstance(counterfactual, Mapping):
        raise TypeError("reference和counterfactual必须是对象")
    result: dict[str, Any] = {"state_source": "direct_recurrent_cache"}
    for state_name in ("ssm_states", "conv_states"):
        left = list(reference.get(state_name, []))
        right = list(counterfactual.get(state_name, []))
        if not left and not right:
            result[state_name] = {
                "available": False,
                "layers": [],
                "minimum_similarity": None,
                "median_similarity": None,
                "maximum_similarity": None,
                "minimum_relative_l2_distance": None,
                "median_relative_l2_distance": None,
                "maximum_relative_l2_distance": None,
                "strongest_divergent_layer": None,
                "strongest_l2_divergent_layer": None,
            }
            continue
        if len(left) != len(right):
            raise ValueError(f"{state_name}层数不一致")
        layers: list[dict[str, float | int]] = []
        for index, (reference_state, counterfactual_state) in enumerate(zip(left, right)):
            similarity = compute_state_similarity(reference_state, counterfactual_state)
            norm_change = compute_state_norm_change(reference_state, counterfactual_state)
            relative_l2_distance = compute_relative_l2_distance(
                reference_state, counterfactual_state
            )
            layers.append(
                {
                    "layer": index,
                    "cosine_similarity": similarity,
                    "relative_norm_change": norm_change,
                    "relative_l2_distance": relative_l2_distance,
                }
            )
        similarities = [float(layer["cosine_similarity"]) for layer in layers]
        distances = [float(layer["relative_l2_distance"]) for layer in layers]
        ordered = sorted(similarities)
        ordered_distances = sorted(distances)
        middle = len(ordered) // 2
        median = (
            ordered[middle]
            if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2.0
        )
        distance_median = (
            ordered_distances[middle]
            if len(ordered_distances) % 2
            else (ordered_distances[middle - 1] + ordered_distances[middle]) / 2.0
        )
        result[state_name] = {
            "available": True,
            "layers": layers,
            "minimum_similarity": min(similarities),
            "median_similarity": median,
            "maximum_similarity": max(similarities),
            "minimum_relative_l2_distance": min(distances),
            "median_relative_l2_distance": distance_median,
            "maximum_relative_l2_distance": max(distances),
            "strongest_divergent_layer": min(
                range(len(similarities)), key=lambda index: similarities[index]
            ),
            "strongest_l2_divergent_layer": max(
                range(len(distances)), key=lambda index: distances[index]
            ),
        }
    return result


def summarize_behavior_state_alignment(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """对同一上下文的行为margin与SSM状态可辨识度作描述性对齐。"""
    values = list(records)
    if not values:
        raise ValueError("records不能为空")
    margins = [float(record["min_signed_margin"]) for record in values]
    ssm_cosines: list[float] = []
    ssm_distances: list[float] = []
    for record in values:
        comparison = record.get("recurrent_state_comparison", {})
        if not isinstance(comparison, Mapping):
            continue
        ssm = comparison.get("ssm_states", {})
        if not isinstance(ssm, Mapping):
            continue
        if ssm.get("minimum_similarity") is not None:
            ssm_cosines.append(float(ssm["minimum_similarity"]))
        if ssm.get("maximum_relative_l2_distance") is not None:
            ssm_distances.append(float(ssm["maximum_relative_l2_distance"]))
    return {
        "behavior": {
            "median_min_signed_margin": float(median(margins)),
            "failure_seed_count": sum(margin <= 0.0 for margin in margins),
        },
        "state": {
            "median_min_ssm_cosine": float(median(ssm_cosines))
            if ssm_cosines
            else None,
            "median_max_ssm_relative_l2": float(median(ssm_distances))
            if ssm_distances
            else None,
        },
        "interpretation": "描述性对齐，不单独构成任何命名SSM失败机制的因果证据",
    }
