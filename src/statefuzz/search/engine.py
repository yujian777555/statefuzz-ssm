"""在可控参数空间内搜索能力退化边界。"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any


@dataclass(frozen=True)
class SearchConfiguration:
    """一次边界搜索的上下文、目标位置和干扰强度配置。"""

    context_tokens: int
    target_position: float
    interference_strength: float
    seed: int = 0

    def to_dict(self) -> dict[str, int | float]:
        return {
            "context_tokens": self.context_tokens,
            "target_position": self.target_position,
            "interference_strength": self.interference_strength,
            "seed": self.seed,
        }


def _validate_space(
    context_lengths: Iterable[int],
    target_positions: Iterable[float],
    interference_strengths: Iterable[float],
    threshold: float,
) -> tuple[list[int], list[float], list[float]]:
    contexts = sorted(set(context_lengths))
    targets = sorted(set(target_positions))
    strengths = sorted(set(interference_strengths))
    if not contexts:
        raise ValueError("context_lengths不能为空")
    if not targets:
        raise ValueError("target_positions不能为空")
    if not strengths:
        raise ValueError("interference_strengths不能为空")
    if any(isinstance(value, bool) or value <= 0 for value in contexts):
        raise ValueError("context_lengths必须为正整数")
    if any(not 0.0 <= value <= 1.0 for value in targets):
        raise ValueError("target_positions必须位于0到1之间")
    if any(not 0.0 <= value <= 1.0 for value in strengths):
        raise ValueError("interference_strengths必须位于0到1之间")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold必须位于0到1之间")
    return contexts, targets, strengths


def _normalize_evaluation(raw: Any) -> tuple[float, dict[str, Any]]:
    if isinstance(raw, Mapping):
        if "score" not in raw:
            raise ValueError("评价结果缺少score")
        score = raw["score"]
        evidence = raw.get("evidence", {})
        if not isinstance(evidence, Mapping):
            raise ValueError("evidence必须是对象")
        return float(score), dict(evidence)
    if isinstance(raw, Real):
        return float(raw), {}
    raise TypeError("评价器必须返回数值或包含score的对象")


def search_boundary(
    evaluator: Callable[[SearchConfiguration], Any],
    context_lengths: Iterable[int],
    target_positions: Iterable[float] = (0.0,),
    interference_strengths: Iterable[float] = (0.0,),
    threshold: float = 0.5,
    seed: int = 0,
) -> dict[str, Any]:
    """遍历参数笛卡尔积并返回最小退化配置及全部观测。"""
    contexts, targets, strengths = _validate_space(
        context_lengths, target_positions, interference_strengths, threshold
    )
    cases: list[dict[str, Any]] = []
    for context_tokens in contexts:
        for target_position in targets:
            for interference_strength in strengths:
                configuration = SearchConfiguration(
                    context_tokens=context_tokens,
                    target_position=target_position,
                    interference_strength=interference_strength,
                    seed=seed,
                )
                score, evidence = _normalize_evaluation(evaluator(configuration))
                if not 0.0 <= score <= 1.0:
                    raise ValueError("score必须位于0到1之间")
                cases.append(
                    {
                        **configuration.to_dict(),
                        "score": score,
                        "failure": score < threshold,
                        "evidence": evidence,
                    }
                )
    failures = [case for case in cases if case["failure"]]
    boundary = (
        min(
            failures,
            key=lambda case: (
                case["context_tokens"],
                case["interference_strength"],
                case["target_position"],
            ),
        )
        if failures
        else None
    )
    return {
        "threshold": threshold,
        "boundary": boundary,
        "cases": cases,
    }


def adaptive_search_boundary(
    evaluator: Callable[[SearchConfiguration], Any],
    min_context: int,
    max_context: int,
    target_position: float = 0.0,
    interference_strength: float = 0.0,
    threshold: float = 0.5,
    tolerance: int = 128,
    seed: int = 0,
) -> dict[str, Any]:
    """以二分细化上下文长度，区分观测失败点和估计边界。"""
    if isinstance(min_context, bool) or isinstance(max_context, bool):
        raise ValueError("context必须为正整数")
    if min_context <= 0 or max_context < min_context:
        raise ValueError("context范围非法")
    if isinstance(tolerance, bool) or tolerance <= 0:
        raise ValueError("tolerance必须为正整数")
    _validate_space(
        [min_context, max_context], [target_position], [interference_strength], threshold
    )
    curve: dict[int, dict[str, Any]] = {}

    def observe(context_tokens: int) -> dict[str, Any]:
        if context_tokens in curve:
            return curve[context_tokens]
        config = SearchConfiguration(
            context_tokens=context_tokens,
            target_position=target_position,
            interference_strength=interference_strength,
            seed=seed,
        )
        score, evidence = _normalize_evaluation(evaluator(config))
        if not 0.0 <= score <= 1.0:
            raise ValueError("score必须位于0到1之间")
        case = {
            **config.to_dict(),
            "score": score,
            "failure": score < threshold,
            "evidence": evidence,
        }
        curve[context_tokens] = case
        return case

    lower_case = observe(min_context)
    lower = min_context
    upper = min_context
    if lower_case["failure"]:
        estimated = 0
        observed = min_context
        lower = 0
    else:
        observed = None
        current = min_context
        while current < max_context:
            next_context = min(max_context, max(current + 1, current * 2))
            case = observe(next_context)
            if case["failure"]:
                upper = next_context
                observed = next_context
                break
            current = next_context
            lower = current
        if observed is None:
            estimated = lower
            upper = lower
        else:
            estimated = lower
            while upper - lower > tolerance:
                middle = (lower + upper) // 2
                if observe(middle)["failure"]:
                    upper = middle
                else:
                    lower = middle
            estimated = lower
    boundary = curve.get(observed) if observed is not None else None
    return {
        "threshold": threshold,
        "estimated_capability_boundary": estimated,
        "observed_failure_context_tokens": observed,
        "confidence_interval": [lower, upper],
        "confidence": 1.0 / (1.0 + max(0, upper - lower)),
        "boundary": boundary,
        "capability_curve": [curve[key] for key in sorted(curve)],
    }


def search_calibrated_boundary(
    evaluator: Callable[[SearchConfiguration], Any],
    min_context: int,
    max_context: int,
    target_position: float = 0.0,
    interference_strength: float = 0.0,
    threshold: float = 0.5,
    baseline_threshold: float = 0.5,
    tolerance: int = 128,
    seed: int = 0,
) -> dict[str, Any]:
    """先验证短上下文基线，再允许自适应搜索进入长上下文。"""
    if not 0.0 <= baseline_threshold <= 1.0:
        raise ValueError("baseline_threshold必须位于0到1之间")
    baseline_config = SearchConfiguration(
        context_tokens=min_context,
        target_position=target_position,
        interference_strength=interference_strength,
        seed=seed,
    )
    baseline_score, baseline_evidence = _normalize_evaluation(evaluator(baseline_config))
    if not 0.0 <= baseline_score <= 1.0:
        raise ValueError("score必须位于0到1之间")
    baseline = {
        **baseline_config.to_dict(),
        "score": baseline_score,
        "failure": baseline_score < baseline_threshold,
        "evidence": baseline_evidence,
    }
    if baseline["failure"]:
        return {
            "valid_baseline": False,
            "failure_reason": "baseline_failure",
            "baseline": baseline,
            "boundary": None,
            "capability_curve": [baseline],
        }
    result = adaptive_search_boundary(
        evaluator,
        min_context=min_context,
        max_context=max_context,
        target_position=target_position,
        interference_strength=interference_strength,
        threshold=threshold,
        tolerance=tolerance,
        seed=seed,
    )
    result.update(
        {
            "valid_baseline": True,
            "failure_reason": (
                "long_context_degradation"
                if result["boundary"] is not None
                else "no_degradation_observed"
            ),
            "baseline": baseline,
        }
    )
    return result


def rank_failure_cases(
    cases: Iterable[Mapping[str, Any]], top_k: int | None = None
) -> list[dict[str, Any]]:
    """按得分从低到高、再按触发规模对失败案例排序。"""
    failures = [dict(case) for case in cases if bool(case.get("failure"))]
    ranked = sorted(
        failures,
        key=lambda case: (
            float(case["score"]),
            int(case["context_tokens"]),
            float(case["interference_strength"]),
            float(case["target_position"]),
        ),
    )
    return ranked if top_k is None else ranked[: max(0, top_k)]


def build_failure_artifact(
    model_id: str,
    search_result: Mapping[str, Any],
    failure_type: str,
    hidden_state_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """构建论文级失败记录，保留触发配置和机制证据。"""
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id必须是非空字符串")
    if not isinstance(failure_type, str) or not failure_type.strip():
        raise ValueError("failure_type必须是非空字符串")
    boundary = search_result.get("boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError("search_result缺少failure boundary")
    if not isinstance(hidden_state_evidence, Mapping):
        raise TypeError("hidden_state_evidence必须是对象")
    return {
        "model_id": model_id,
        "effective_memory_boundary": boundary["context_tokens"],
        "failure_type": failure_type,
        "trigger_configuration": {
            "context_tokens": boundary["context_tokens"],
            "target_position": boundary["target_position"],
            "interference_strength": boundary["interference_strength"],
            "seed": boundary["seed"],
        },
        "hidden_state_evidence": dict(hidden_state_evidence),
    }

