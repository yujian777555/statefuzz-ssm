"""跨架构行为曲线的保守特异性判定。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def summarize_model_behavior_curve(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """从已聚合的实际token曲线推导边界摘要。"""
    def actual_range(record: Mapping[str, Any]) -> tuple[int, int]:
        if record.get("actual_input_token_range") is not None:
            values = record["actual_input_token_range"]
            return int(values[0]), int(values[1])
        value = int(record.get("actual_input_tokens", record.get("context_tokens", 0)))
        return value, value

    cases = sorted(
        [dict(record) for record in records],
        key=lambda record: actual_range(record)[0],
    )
    if not cases:
        return {
            "boundary_kind": "invalid_task",
            "actual_boundary_interval": None,
            "min_actual_tokens": None,
            "max_actual_tokens": None,
        }
    first_fail = None
    for index, case in enumerate(cases):
        failed = bool(case.get("all_seeds_fail", case.get("failure_seed_count", 0) > 0))
        passed = bool(case.get("all_seeds_pass", not failed))
        case["_failed"] = failed
        case["_passed"] = passed
        if failed and first_fail is None:
            first_fail = index
    if first_fail is not None and any(case["_passed"] for case in cases[first_fail + 1 :]):
        kind = "nonmonotonic"
        interval = None
    elif first_fail is None:
        kind = "lower_bound"
        interval = None
    elif first_fail > 0 and cases[first_fail - 1]["_passed"] and cases[first_fail]["_failed"]:
        kind = "replicated_zero_crossing"
        interval = [actual_range(cases[first_fail - 1])[1], actual_range(cases[first_fail])[0]]
    else:
        kind = "candidate_unreplicated"
        interval = None
    return {
        "boundary_kind": kind,
        "actual_boundary_interval": interval,
        "min_actual_tokens": actual_range(cases[0])[0],
        "max_actual_tokens": actual_range(cases[-1])[1],
    }


def compare_architecture_results(
    mamba_result: Mapping[str, Any], transformer_result: Mapping[str, Any]
) -> dict[str, Any]:
    """只使用实际token覆盖和复制区间进行架构分类。"""
    mamba = dict(mamba_result)
    transformer = dict(transformer_result)
    mamba_interval = mamba.get("actual_boundary_interval")
    transformer_interval = transformer.get("actual_boundary_interval")
    mamba_max = mamba.get("max_actual_tokens")
    transformer_max = transformer.get("max_actual_tokens")
    availability = transformer.get("availability", "available")
    coverage_sufficient = bool(
        availability == "available"
        and mamba_interval
        and isinstance(transformer_max, int)
        and transformer_max >= int(mamba_interval[1])
    )
    if mamba.get("boundary_kind") == "nonmonotonic" or transformer.get("boundary_kind") == "nonmonotonic":
        classification = "inconclusive_nonmonotonic"
    elif not coverage_sufficient:
        classification = "inconclusive_control"
    elif transformer.get("boundary_kind") != "replicated_zero_crossing":
        classification = "ssm_specific_candidate"
    elif not transformer_interval:
        classification = "inconclusive_control"
    elif mamba_interval and max(mamba_interval[0], transformer_interval[0]) <= min(
        mamba_interval[1], transformer_interval[1]
    ):
        classification = "shared_base_lm_decay"
    else:
        classification = "architecture_differential"
    common_range = None
    if mamba.get("min_actual_tokens") is not None and transformer.get("min_actual_tokens") is not None:
        common_range = [
            max(int(mamba["min_actual_tokens"]), int(transformer["min_actual_tokens"])),
            min(int(mamba_max), int(transformer_max)) if mamba_max is not None and transformer_max is not None else None,
        ]
    return {
        "specificity_classification": classification,
        "mamba_boundary_interval_actual": mamba_interval,
        "transformer_boundary_interval_actual": transformer_interval,
        "common_evaluated_token_range": common_range,
        "coverage_sufficient": coverage_sufficient,
    }
