"""将探针得分转换为可复核的能力边界结果。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import NormalDist
from typing import Any, Iterable


@dataclass(frozen=True)
class CapabilityObservation:
    """一个上下文长度上的模型能力观测点。"""

    context_tokens: int
    score: float
    state_similarity: float | None = None
    state_norm: float | None = None

    def to_dict(self) -> dict[str, int | float | None]:
        """返回JSON可序列化观测记录。"""
        return {
            "context_tokens": self.context_tokens,
            "score": self.score,
            "state_similarity": self.state_similarity,
            "state_norm": self.state_norm,
        }


def _validate_observations(
    observations: Iterable[CapabilityObservation],
) -> list[CapabilityObservation]:
    values = list(observations)
    if not values:
        raise ValueError("observations不能为空")
    for item in values:
        if not isinstance(item, CapabilityObservation):
            raise TypeError("观测项必须是CapabilityObservation")
        if isinstance(item.context_tokens, bool) or item.context_tokens <= 0:
            raise ValueError("context_tokens必须为正整数")
        if not 0.0 <= item.score <= 1.0:
            raise ValueError("score必须位于0到1之间")
    values.sort(key=lambda item: item.context_tokens)
    contexts = [item.context_tokens for item in values]
    if len(contexts) != len(set(contexts)):
        raise ValueError("context_tokens不能重复")
    return values


def measure_effective_memory(
    observations: Iterable[CapabilityObservation], threshold: float = 0.5
) -> dict[str, Any]:
    """测量达到阈值的最大上下文长度及首次退化点。"""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold必须位于0到1之间")
    values = _validate_observations(observations)
    degradation = next(
        (item.context_tokens for item in values if item.score < threshold), None
    )
    if degradation is None:
        effective = values[-1].context_tokens
        estimated = effective
    else:
        effective = max(
            (item.context_tokens for item in values if item.score >= threshold),
            default=0,
        )
        estimated = (effective + degradation) // 2
    return {
        "effective_memory_tokens": effective,
        "degradation_context_tokens": degradation,
        "observed_failure_context_tokens": degradation,
        "estimated_boundary_context_tokens": estimated,
        "threshold": threshold,
        "observations": [item.to_dict() for item in values],
    }


def build_capability_artifact(
    model_id: str,
    observations: Iterable[CapabilityObservation],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """构建包含模型标识和能力边界的机器可读结果。"""
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id必须是非空字符串")
    return {"model_id": model_id, **measure_effective_memory(observations, threshold)}


def build_calibrated_report(
    model_id: str,
    baseline_score: float,
    search_result: dict[str, Any],
    failure_evidence: dict[str, Any] | None = None,
    baseline_threshold: float = 0.5,
) -> dict[str, Any]:
    """构建区分任务有效性、边界置信度和机制证据的报告。"""
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id必须是非空字符串")
    if not 0.0 <= baseline_score <= 1.0:
        raise ValueError("baseline_score必须位于0到1之间")
    valid = bool(search_result.get("valid_baseline", baseline_score >= baseline_threshold))
    observed = search_result.get("observed_failure_context_tokens")
    estimated = search_result.get("estimated_capability_boundary")
    kind = "interval" if observed is not None else "lower_bound"
    if not valid:
        kind = "invalid_baseline"
    return {
        "model_id": model_id,
        "task_validity": {
            "valid": valid,
            "baseline_score": baseline_score,
            "baseline_threshold": baseline_threshold,
            "failure_reason": search_result.get("failure_reason"),
        },
        "capability_boundary": {
            "kind": kind,
            "estimated_tokens": estimated,
            "observed_failure_tokens": observed,
            "confidence": search_result.get("confidence", 0.0) if valid else 0.0,
        },
        "failure_mechanism_evidence": dict(failure_evidence or {}),
    }


def compute_confidence_interval(
    values: Iterable[float], confidence: float = 0.95
) -> dict[str, float | int]:
    """用正态近似计算有界得分均值的可复核置信区间。"""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence必须位于0到1之间")
    samples = [float(value) for value in values]
    if not samples:
        raise ValueError("values不能为空")
    if any(not 0.0 <= value <= 1.0 for value in samples):
        raise ValueError("values必须位于0到1之间")
    count = len(samples)
    mean = sum(samples) / count
    if count == 1:
        margin = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in samples) / (count - 1)
        z_value = NormalDist().inv_cdf(0.5 + confidence / 2.0)
        margin = z_value * math.sqrt(variance / count)
    return {
        "n": count,
        "mean": mean,
        "lower": max(0.0, mean - margin),
        "upper": min(1.0, mean + margin),
        "confidence": confidence,
    }


def aggregate_failure_evidence(
    evidences: Iterable[dict[str, Any]],
) -> dict[str, float | int | None]:
    """聚合多个失败实例中的状态相似度和范数变化。"""
    records = list(evidences)
    if not records:
        raise ValueError("evidences不能为空")

    def mean_field(field: str) -> float | None:
        values = [float(item[field]) for item in records if item.get(field) is not None]
        return round(sum(values) / len(values), 12) if values else None

    return {
        "count": len(records),
        "mean_state_norm_change": mean_field("state_norm_change"),
        "mean_state_similarity": mean_field("state_similarity"),
    }

