"""反事实远程记忆依赖度量与短上下文任务校验。"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from numbers import Real
from typing import Any


def _candidate_logit(candidate: Mapping[str, Any]) -> float:
    """读取候选logit；旧缓存无logit时仅为兼容使用log概率回退。"""
    if candidate.get("logit") is not None:
        return float(candidate["logit"])
    if candidate.get("log_probability") is not None:
        return float(candidate["log_probability"])
    probability = float(candidate.get("probability", 0.0))
    if probability <= 0.0:
        return float("-inf")
    return math.log(probability)


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
    logits_a = {token_id: _candidate_logit(scores_a[token_id]) for token_id in candidate_ids}
    logits_b = {token_id: _candidate_logit(scores_b[token_id]) for token_id in candidate_ids}
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


def compute_pairwise_memory_metrics(record: Mapping[str, Any]) -> dict[str, Any]:
    """计算每个提示内部的候选A/B signed margin。

    该指标只比较同一提示内的两个候选，避免长上下文把绝对概率质量
    分散到无关词表后造成假性退化。
    """
    if not isinstance(record, Mapping):
        raise TypeError("record必须是对象")
    candidate_ids = record.get("candidate_token_ids")
    if (
        not isinstance(candidate_ids, (list, tuple))
        or len(candidate_ids) != 2
        or any(not isinstance(value, int) or isinstance(value, bool) for value in candidate_ids)
    ):
        return {
            "direction_a_margin": 0.0,
            "direction_b_margin": 0.0,
            "direction_a_pair_probability": 0.5,
            "direction_b_pair_probability": 0.5,
            "pairwise_memory_score": 0.5,
            "min_signed_margin": 0.0,
            "pairwise_preference_valid": False,
            "pairwise_metric_source": "invalid_candidates",
        }
    prompt_a = record.get("prompt_a", record.get("score_a"))
    prompt_b = record.get("prompt_b", record.get("score_b"))
    if not isinstance(prompt_a, Mapping) or not isinstance(prompt_b, Mapping):
        raise ValueError("远程记忆记录必须包含prompt_a/prompt_b评分")
    scores_a = _candidate_map(prompt_a)
    scores_b = _candidate_map(prompt_b)
    token_a, token_b = int(candidate_ids[0]), int(candidate_ids[1])
    if any(token_id not in scores_a or token_id not in scores_b for token_id in (token_a, token_b)):
        raise ValueError("两个提示都必须评分全部候选")
    margin_a = _candidate_logit(scores_a[token_a]) - _candidate_logit(scores_a[token_b])
    margin_b = _candidate_logit(scores_b[token_b]) - _candidate_logit(scores_b[token_a])

    def sigmoid(value: float) -> float:
        if value >= 0.0:
            return 1.0 / (1.0 + math.exp(-value))
        exp_value = math.exp(value)
        return exp_value / (1.0 + exp_value)

    pair_probability_a = sigmoid(margin_a)
    pair_probability_b = sigmoid(margin_b)
    return {
        "direction_a_margin": margin_a,
        "direction_b_margin": margin_b,
        "direction_a_pair_probability": pair_probability_a,
        "direction_b_pair_probability": pair_probability_b,
        "pairwise_memory_score": (pair_probability_a + pair_probability_b) / 2.0,
        "min_signed_margin": min(margin_a, margin_b),
        "pairwise_preference_valid": margin_a > 0.0 and margin_b > 0.0,
        "pairwise_metric_source": "candidate_logits",
    }


def decompose_pairwise_preference(record: Mapping[str, Any]) -> dict[str, Any]:
    """把候选偏好拆成反事实记忆信号与共享词汇偏置。"""
    if not isinstance(record, Mapping):
        raise TypeError("record必须是对象")
    if "direction_a_margin" in record and "direction_b_margin" in record:
        direction_a = float(record["direction_a_margin"])
        direction_b = float(record["direction_b_margin"])
    else:
        metrics = compute_pairwise_memory_metrics(record)
        direction_a = float(metrics["direction_a_margin"])
        direction_b = float(metrics["direction_b_margin"])
    # direction_b_margin是B-A，因此还原prompt B上的原始A-B偏好。
    raw_preference_b = -direction_b
    memory_signal = (direction_a - raw_preference_b) / 2.0
    lexical_bias = (direction_a + raw_preference_b) / 2.0
    bias_dominance_margin = memory_signal - abs(lexical_bias)
    return {
        "raw_preference_prompt_a": direction_a,
        "raw_preference_prompt_b": raw_preference_b,
        "memory_signal": memory_signal,
        "lexical_bias": lexical_bias,
        "bias_dominance_margin": bias_dominance_margin,
        "bias_dominated": bias_dominance_margin <= 0.0,
    }


def compute_counterfactual_memory_score(record: Mapping[str, Any]) -> dict[str, Any]:
    """计算 A/B 两个方向的对称反事实偏好差，结果限制在 ``[0, 1]``。"""
    if not isinstance(record, Mapping):
        raise TypeError("record必须是对象")
    candidate_ids = record.get("candidate_token_ids")
    if (
        not isinstance(candidate_ids, (list, tuple))
        or len(candidate_ids) != 2
        or any(not isinstance(value, int) or isinstance(value, bool) for value in candidate_ids)
    ):
        counts = list(record.get("prompt_token_counts", []))
        return {
            "seed": record.get("seed"),
            "template_id": record.get("template_id"),
            "candidate_token_ids": list(candidate_ids) if isinstance(candidate_ids, (list, tuple)) else [],
            "candidate_values": list(record.get("candidate_values", record.get("values", []))),
            "prompt_token_counts": counts,
            "paired_token_counts": len(counts) == 2 and counts[0] == counts[1],
            "matched": bool(record.get("matched", False)),
            "candidate_valid": False,
            "direction_a_contrast": 0.0,
            "direction_b_contrast": 0.0,
            "direction_a_logit_contrast": 0.0,
            "direction_b_logit_contrast": 0.0,
            "memory_dependence_score": 0.5,
            "valid_short_context_task": False,
            **compute_pairwise_memory_metrics(record),
            **decompose_pairwise_preference(record),
        }
    result = _metric_for_pair(record)
    result.update(compute_pairwise_memory_metrics(record))
    result.update(decompose_pairwise_preference(result))
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
    pairwise_scores = [item["pairwise_memory_score"] for item in per_instance]
    min_margins = [item["min_signed_margin"] for item in per_instance]
    direction_a = [item["direction_a_contrast"] for item in per_instance]
    direction_b = [item["direction_b_contrast"] for item in per_instance]
    mean_score = sum(scores) / len(scores)
    mean_a = sum(direction_a) / len(direction_a)
    mean_b = sum(direction_b) / len(direction_b)
    all_valid = all(item["valid_short_context_task"] for item in per_instance)
    mean_pairwise_score = sum(pairwise_scores) / len(pairwise_scores)
    valid = (
        all_valid
        and all(item["pairwise_preference_valid"] for item in per_instance)
        and mean_pairwise_score >= threshold
        and mean_a >= min_directional_contrast
        and mean_b >= min_directional_contrast
    )
    first = per_instance[0]
    return {
        "valid_short_context_task": valid,
        "memory_dependence_score": mean_score,
        "pairwise_memory_score": mean_pairwise_score,
        "min_signed_margin": min(min_margins),
        "pairwise_preference_valid": all(
            item["pairwise_preference_valid"] for item in per_instance
        ),
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


def compute_pairwise_score_sensitivity(
    curve: Iterable[Mapping[str, Any]],
    thresholds: Iterable[float] = (0.60, 0.70, 0.80, 0.90),
) -> dict[str, int | None]:
    """返回各描述性pairwise分数阈值的首次下降点（非主边界）。"""
    cases = sorted(curve, key=lambda case: int(case["context_tokens"]))
    if not cases:
        raise ValueError("curve不能为空")
    result: dict[str, int | None] = {}
    for threshold in thresholds:
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("threshold必须位于0到1之间")
        crossing = next(
            (
                int(case["context_tokens"])
                for case in cases
                if float(case.get("pairwise_memory_score", case.get("memory_dependence_score", 0.0)))
                < float(threshold)
            ),
            None,
        )
        result[str(float(threshold)).rstrip("0").rstrip(".")] = crossing
    return result
