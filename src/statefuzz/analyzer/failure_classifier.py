"""将预测失败映射到可解释的SSM机制类别。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from statefuzz.analyzer.hidden_state import (
    compute_state_norm,
    compute_state_norm_change,
    compute_state_similarity,
    detect_state_collapse,
)


def classify_failure(
    expected: str,
    prediction: str,
    states: Iterable[Any] | None = None,
    *,
    state_norms: Iterable[float] | None = None,
    evidence: Mapping[str, Any] | None = None,
    target_token_id: int | None = None,
    predicted_token_id: int | None = None,
) -> str:
    """只在有对应证据时分类状态机制，否则返回保守行为标签。"""
    if not isinstance(expected, str) or not isinstance(prediction, str):
        raise TypeError("expected和prediction必须是字符串")
    if prediction == expected:
        return "none"
    explicit = dict(evidence or {})
    if explicit.get("state_collapse") or (
        states is not None and detect_state_collapse(states)
    ):
        return "state_collapse"
    if state_norms is not None:
        norms = list(state_norms)
        if any(value < 0.0 for value in norms):
            raise ValueError("state_norms不能为负数")
        if len(norms) >= 2 and norms[0] > 0.0 and norms[-1] / norms[0] < 0.25:
            return "state_forgetting"
    if explicit.get("state_forgetting"):
        return "state_forgetting"
    if explicit.get("state_collision"):
        return "state_collision"
    if explicit.get("state_pollution"):
        return "state_pollution"
    if (
        target_token_id is not None
        and predicted_token_id is not None
        and target_token_id != predicted_token_id
    ):
        return "interference_induced_token_shift"
    if states is not None or state_norms is not None:
        return "insufficient_mechanism_evidence"
    return "behavioral_interference"


def diagnose_failure(
    expected: str,
    prediction: str,
    states: Iterable[Any] | None = None,
    layer_states: Mapping[str, tuple[Any, Any]] | None = None,
    state_source: str = "unavailable",
) -> dict[str, Any]:
    """返回失败类别及相似度、范数变化等机制证据。"""
    state_values = list(states) if states is not None else []
    state_norms = [compute_state_norm(state) for state in state_values]
    similarity = (
        compute_state_similarity(state_values[0], state_values[-1])
        if len(state_values) >= 2
        else None
    )
    norm_change = (
        compute_state_norm_change(state_values[0], state_values[-1])
        if len(state_values) >= 2
        else None
    )
    category = classify_failure(
        expected,
        prediction,
        states=state_values,
        state_norms=state_norms,
    )
    evidence = {
        "state_source": state_source,
        "state_similarity": similarity,
        "state_norm_change": norm_change,
        "state_norms": state_norms,
    }
    if layer_states is not None:
        evidence["layer_similarity"] = {
            str(layer): compute_state_similarity(pair[0], pair[1])
            for layer, pair in layer_states.items()
        }
    return {
        "category": category,
        "evidence": evidence,
    }

