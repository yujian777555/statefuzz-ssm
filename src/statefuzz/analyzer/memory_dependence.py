"""反事实远程记忆依赖度量与短上下文任务校验。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from numbers import Real
from typing import Any


def _candidate_map(score: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    candidates = score.get("candidates", [])
    if not isinstance(candidates, Iterable):
        raise ValueError("候选评分必须是可迭代对象")
    result: dict[int, Mapping[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or "token_id" not in candidate:
            raise ValueError("候选评分缺少token_id")
        result[int(candidate["token_id"])] = candidate
    return result


def _metric_for_pair(record: Mapping[str, Any]) -> dict[str, Any]:
    prompt_a = record.get("prompt_a", record.get("score_a"))
    prompt_b = record.get("prompt_b", record.get("score_b"))
    if not isinstance(prompt_a, Mapping) or not isinstance(prompt_b, Mapping):
        raise ValueError("远程记忆记录必须包含prompt_a/prompt_b评分")
    candidate_ids = record.get("candidate_token_ids")
    if not isinstance(candidate_ids, (list, tuple)) or len(candidate_ids) != 2:
        raise ValueError("candidate_token_ids必须包含两个候选")
    candidate_ids = [int(value) for value in candidate_ids]
    scores_a = _candidate_map(prompt_a)
    scores_b = _candidate_map(prompt_b)
    if any(token_id not in scores_a or token_id not in scores_b for token_id in candidate_ids):
        raise ValueError("两个提示都必须评分全部候选")
    a_on_a = float(scores_a[candidate_ids[0]].get("probability", 0.0))
    a_on_b = float(scores_b[candidate_ids[0]].get("probability", 0.0))
    b_on_a = float(scores_a[candidate_ids[1]].get("probability", 0.0))
    b_on_b = float(scores_b[candidate_ids[1]].get("probability", 0.0))
    direction_a = round(a_on_a - a_on_b, 12)
    direction_b = round(b_on_b - b_on_a, 12)
    score = max(0.0, min(1.0, 0.5 + 0.5 * ((direction_a + direction_b) / 2.0)))
    logits_a = {
        token_id: float(scores_a[token_id].get("logit", scores_a[token_id].get("log_probability", 0.0)))
        for token_id in candidate_ids
    }
    logits_b = {
        token_id: float(scores_b[token_id].get("logit", scores_b[token_id].get("log_probability", 0.0)))
        for token_id in candidate_ids
    }
    result = {
        "seed": record.get("seed"),
        "template_id": record.get("template_id"),
        "candidate_token_ids": candidate_ids,
        "candidate_values": list(record.get("candidate_values", record.get("values", []))),
        "prompt_token_counts": list(record.get("prompt_token_counts", [])),
        "matched": bool(record.get("matched", False)),
        "candidate_valid": bool(record.get("candidate_valid", True)),
        "direction_a_contrast": direction_a,
        "direction_b_contrast": direction_b,
        "direction_a_logit_contrast": logits_a[candidate_ids[0]] - logits_b[candidate_ids[0]],
        "direction_b_logit_contrast": logits_b[candidate_ids[1]] - logits_a[candidate_ids[1]],
        "memory_dependence_score": score,
    }
    return result


def compute_counterfactual_memory_score(record: Mapping[str, Any]) -> dict[str, Any]:
    """计算 A/B 两个方向的对称反事实偏好差，结果限制在 ``[0, 1]``。"""
    if not isinstance(record, Mapping):
        raise TypeError("record必须是对象")
    result = _metric_for_pair(record)
    counts = result["prompt_token_counts"]
    result["paired_token_counts"] = len(counts) == 2 and counts[0] == counts[1]
    result["valid_short_context_task"] = bool(
        result["matched"] and result["candidate_valid"] and result["paired_token_counts"]
    )
    return result


def validate_remote_memory_task(
    records: Iterable[Mapping[str, Any]],
    threshold: float = 0.55,
    min_directional_contrast: float = 0.05,
) -> dict[str, Any]:
    """在多个随机种子上验证冻结的短上下文反事实任务。"""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold必须位于0到1之间")
    if not 0.0 <= min_directional_contrast <= 1.0:
        raise ValueError("min_directional_contrast必须位于0到1之间")
    values = list(records)
    if not values:
        raise ValueError("records不能为空")
    per_instance = [compute_counterfactual_memory_score(record) for record in values]
    scores = [item["memory_dependence_score"] for item in per_instance]
    direction_a = [item["direction_a_contrast"] for item in per_instance]
    direction_b = [item["direction_b_contrast"] for item in per_instance]
    mean_score = sum(scores) / len(scores)
    mean_a = sum(direction_a) / len(direction_a)
    mean_b = sum(direction_b) / len(direction_b)
    all_valid = all(item["valid_short_context_task"] for item in per_instance)
    valid = all_valid and mean_score >= threshold and mean_a >= min_directional_contrast and mean_b >= min_directional_contrast
    first = per_instance[0]
    return {
        "valid_short_context_task": valid,
        "memory_dependence_score": mean_score,
        "direction_a_contrast": mean_a,
        "direction_b_contrast": mean_b,
        "min_directional_contrast": min_directional_contrast,
        "threshold": threshold,
        "seed_count": len(per_instance),
        "template_id": first.get("template_id"),
        "candidate_token_ids": first.get("candidate_token_ids", []),
        "candidate_values": first.get("candidate_values", []),
        "paired_token_counts": [item.get("prompt_token_counts", []) for item in per_instance],
        "per_instance": per_instance,
        "invalid_reasons": [
            "unmatched_or_candidate_invalid"
            for item in per_instance
            if not item["valid_short_context_task"]
        ],
    }
