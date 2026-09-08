"""种子级失败风险、过渡区间与配对架构检验。"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any


def wilson_interval(
    failures: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    """计算二项比例的Wilson 95%区间。"""
    if isinstance(failures, bool) or isinstance(total, bool):
        raise ValueError("failures和total必须是整数")
    if not isinstance(failures, int) or not isinstance(total, int) or total <= 0:
        raise ValueError("total必须是正整数")
    if failures < 0 or failures > total or not math.isfinite(z) or z < 0.0:
        raise ValueError("failures或z超出范围")
    proportion = failures / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _failure(record: Mapping[str, Any]) -> bool:
    margin = record.get("min_signed_margin")
    if margin is None or not math.isfinite(float(margin)):
        raise ValueError("记录缺少min_signed_margin")
    return float(margin) <= 0.0


def summarize_failure_risk_curve(
    records: Iterable[Mapping[str, Any]], expected_seeds: Iterable[int]
) -> dict[str, Any]:
    seeds = list(dict.fromkeys(expected_seeds))
    if not seeds:
        raise ValueError("expected_seeds不能为空")
    grouped: dict[int, list[dict[str, Any]]] = {}
    for raw in records:
        record = dict(raw)
        if record.get("target_budget_tokens") is None:
            raise ValueError("记录缺少target_budget_tokens")
        grouped.setdefault(int(record["target_budget_tokens"]), []).append(record)
    curve = []
    for budget, values in sorted(grouped.items()):
        valid = [
            value
            for value in values
            if value.get("actual_input_tokens") is not None
            and bool(value.get("matched", True))
            and bool(value.get("candidate_valid", True))
        ]
        seed_set = {value.get("seed") for value in valid}
        complete = seed_set == set(seeds) and len(valid) == len(seeds)
        failures = sum(_failure(value) for value in valid)
        curve.append(
            {
                "target_budget_tokens": budget,
                "actual_input_token_range": [
                    min(value["actual_input_tokens"] for value in valid),
                    max(value["actual_input_tokens"] for value in valid),
                ]
                if valid
                else None,
                "failure_count": failures,
                "seed_count": len(valid),
                "failure_rate": failures / len(valid) if valid else None,
                "wilson_95": list(wilson_interval(failures, len(valid)))
                if valid
                else None,
                "complete_seed_set": complete,
                "records": valid,
            }
        )
    return {"expected_seeds": seeds, "curve": curve}


def summarize_seed_transition_intervals(
    records: Iterable[Mapping[str, Any]], expected_seeds: Iterable[int]
) -> dict[str, Any]:
    seeds = list(dict.fromkeys(expected_seeds))
    by_seed: dict[int, list[dict[str, Any]]] = {seed: [] for seed in seeds}
    for raw in records:
        record = dict(raw)
        seed = record.get("seed")
        if seed in by_seed and record.get("actual_input_tokens") is not None:
            by_seed[seed].append(record)
    per_seed = []
    for seed in seeds:
        values = sorted(by_seed[seed], key=lambda value: value["actual_input_tokens"])
        failures = [_failure(value) for value in values]
        first_fail_index = next((index for index, failed in enumerate(failures) if failed), None)
        nonmonotonic = bool(
            first_fail_index is not None and any(not failed for failed in failures[first_fail_index + 1 :])
        )
        last_pass = None
        first_fail = None
        if first_fail_index is None and values:
            last_pass = values[-1]["actual_input_tokens"]
        if first_fail_index is not None:
            first_fail = values[first_fail_index]["actual_input_tokens"]
            prior = [value["actual_input_tokens"] for value, failed in zip(values[:first_fail_index], failures) if not failed]
            last_pass = max(prior) if prior else None
        per_seed.append(
            {
                "seed": seed,
                "observed_points": len(values),
                "last_pass_actual_tokens": last_pass,
                "first_fail_actual_tokens": first_fail,
                "transition_interval_actual": [last_pass, first_fail]
                if last_pass is not None and first_fail is not None
                else None,
                "right_censored": first_fail is None and bool(values),
                "left_censored": first_fail_index == 0,
                "nonmonotonic": nonmonotonic,
            }
        )
    return {
        "expected_seeds": seeds,
        "per_seed": per_seed,
        "monotonic_seed_count": sum(not value["nonmonotonic"] for value in per_seed),
        "nonmonotonic_seed_count": sum(value["nonmonotonic"] for value in per_seed),
        "right_censored_seed_count": sum(value["right_censored"] for value in per_seed),
    }


def exact_paired_mcnemar(
    mamba_failures: Mapping[int, bool], transformer_failures: Mapping[int, bool]
) -> dict[str, Any]:
    if set(mamba_failures) != set(transformer_failures):
        raise ValueError("两个模型必须使用相同seed集合")
    b = sum(bool(mamba_failures[seed]) and not bool(transformer_failures[seed]) for seed in mamba_failures)
    c = sum(not bool(mamba_failures[seed]) and bool(transformer_failures[seed]) for seed in mamba_failures)
    discordant = b + c
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, index) for index in range(min(b, c) + 1)) / (2.0**discordant)
        p_value = min(1.0, 2.0 * tail)
    direction = "mamba_higher" if b > c else "transformer_higher" if c > b else "tie"
    return {
        "mamba_only_fail": b,
        "transformer_only_fail": c,
        "discordant_pairs": discordant,
        "exact_two_sided_p": p_value,
        "direction": direction,
    }


def compare_architecture_failure_risk(
    mamba_records: Iterable[Mapping[str, Any]],
    transformer_records: Iterable[Mapping[str, Any]],
    expected_seeds: Iterable[int],
    primary_target_budget: int,
) -> dict[str, Any]:
    seeds = list(dict.fromkeys(expected_seeds))

    mamba_records = list(mamba_records)
    transformer_records = list(transformer_records)
    if not seeds or any(
        len([r for r in rows if r.get("target_budget_tokens") == primary_target_budget]) != len(seeds)
        for rows in (mamba_records, transformer_records)
    ):
        return {"valid": False, "reason": "duplicate_or_missing_primary_record"}

    def select(records):
        return {
            record["seed"]: record
            for record in records
            if record.get("target_budget_tokens") == primary_target_budget
        }

    mamba = select(mamba_records)
    transformer = select(transformer_records)
    if set(mamba) != set(seeds) or set(transformer) != set(seeds):
        return {"valid": False, "reason": "seed_set_or_primary_endpoint_incomplete"}
    for model_records in (mamba, transformer):
        for record in model_records.values():
            actual = record.get("actual_input_tokens")
            tolerance = int(record.get("tolerance_tokens", 16))
            counts = record.get("prompt_token_counts", [actual, actual])
            if (
                actual is None
                or tolerance < 0
                or counts != [actual, actual]
                or record.get("min_signed_margin") is None
                or not math.isfinite(float(record["min_signed_margin"]))
                or not bool(record.get("matched", False))
                or not bool(record.get("candidate_valid", False))
                or abs(int(actual) - primary_target_budget) > tolerance
            ):
                return {"valid": False, "reason": "token_budget_or_candidate_invalid"}
    mamba_failures = {seed: _failure(mamba[seed]) for seed in seeds}
    transformer_failures = {seed: _failure(transformer[seed]) for seed in seeds}
    def summary(records, failures):
        count = sum(failures.values())
        actual = [int(value["actual_input_tokens"]) for value in records.values()]
        return {
            "failure_count": count,
            "seed_count": len(records),
            "failure_rate": count / len(records),
            "wilson_95": list(wilson_interval(count, len(records))),
            "actual_token_range": [min(actual), max(actual)],
        }
    return {
        "valid": True,
        "primary_target_budget_tokens": primary_target_budget,
        "expected_seeds": seeds,
        "mamba": summary(mamba, mamba_failures),
        "transformer": summary(transformer, transformer_failures),
        "mcnemar": exact_paired_mcnemar(mamba_failures, transformer_failures),
    }


def classify_prospective_architecture_risk(
    primary_comparison: Mapping[str, Any],
    structured_curve: Mapping[str, Any],
    negative_control: Mapping[str, Any],
) -> str:
    if not bool(primary_comparison.get("valid", False)):
        return "prospective_control_invalid"
    mcnemar = primary_comparison.get("mcnemar", {})
    if (
        mcnemar.get("direction") == "mamba_higher"
        and float(mcnemar.get("exact_two_sided_p", 1.0)) < 0.05
    ):
        return "architecture_risk_gap_confirmed"
    mamba_rate = float(primary_comparison["mamba"]["failure_rate"])
    transformer_rate = float(primary_comparison["transformer"]["failure_rate"])
    if mamba_rate >= 0.5 and transformer_rate >= 0.5:
        return "shared_failure_risk"
    return "architecture_risk_gap_not_confirmed"
