"""将探针得分转换为可复核的能力边界结果。"""

from __future__ import annotations

from dataclasses import dataclass
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

