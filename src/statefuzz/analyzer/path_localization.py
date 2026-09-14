"""混合缓存路径的seed级供体迁移统计。"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import mean, median, stdev


def compute_transfer_ratio(
    recipient: float,
    donor: float,
    intervention: float,
    min_separation: float = 0.25,
) -> float:
    """计算未裁剪的供体迁移比例。"""
    values = [float(recipient), float(donor), float(intervention)]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("偏好值必须是有限数")
    separation = values[1] - values[0]
    if abs(separation) < float(min_separation):
        raise ValueError("counterfactual_separation_too_small")
    return (values[2] - values[0]) / separation


def _bootstrap_mean(values, *, seed, samples):
    if not values:
        return None
    generator = random.Random(seed)
    estimates = sorted(
        mean(generator.choice(values) for _ in values) for _ in range(samples)
    )
    lower = estimates[int(0.025 * (samples - 1))]
    upper = estimates[int(0.975 * (samples - 1))]
    return [lower, upper]


def summarize_path_localization(
    records, bootstrap_seed: int = 32032, bootstrap_samples: int = 10000
):
    """先合并两个干预方向，再以seed为独立统计单位。"""
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples必须为正整数")
    grouped = defaultdict(list)
    for raw in records:
        record = dict(raw)
        if not record.get("protocol_valid", False):
            continue
        key = (record["stress_family"], int(record["target_budget"]), record["path"])
        grouped[key].append(record)
    summaries = []
    seed_values_by_group = {}
    for group_index, (key, values) in enumerate(sorted(grouped.items())):
        by_seed = defaultdict(list)
        for value in values:
            by_seed[int(value["seed"])].append(value)
        seed_rows = []
        for seed, directions in sorted(by_seed.items()):
            direction_names = {value["direction"] for value in directions}
            if direction_names != {"a_from_b", "b_from_a"} or len(directions) != 2:
                continue
            ratios = [float(value["transfer_ratio"]) for value in directions]
            seed_rows.append(
                {
                    "seed": seed,
                    "mean_transfer_ratio": mean(ratios),
                    "both_directions_toward_donor": all(
                        bool(value["toward_donor"]) for value in directions
                    ),
                }
            )
        ratios = [row["mean_transfer_ratio"] for row in seed_rows]
        ci = _bootstrap_mean(
            ratios,
            seed=bootstrap_seed + group_index,
            samples=bootstrap_samples,
        )
        movement = sum(row["both_directions_toward_donor"] for row in seed_rows)
        candidate = len(seed_rows) >= 7 and movement >= 7 and ci is not None and ci[0] > 0
        summaries.append(
            {
                "stress_family": key[0],
                "target_budget": key[1],
                "path": key[2],
                "valid_seed_count": len(seed_rows),
                "mean_transfer_ratio": mean(ratios) if ratios else None,
                "median_transfer_ratio": median(ratios) if ratios else None,
                "standard_deviation": stdev(ratios) if len(ratios) > 1 else 0.0 if ratios else None,
                "donor_movement_seed_count": movement,
                "bootstrap_95_ci": ci,
                "causal_influence_candidate": candidate,
                "seed_level": seed_rows,
            }
        )
        seed_values_by_group[key] = {row["seed"]: row["mean_transfer_ratio"] for row in seed_rows}
    length_dependence = []
    combinations = sorted({(key[0], key[2]) for key in seed_values_by_group})
    for index, (family, path) in enumerate(combinations):
        short = seed_values_by_group.get((family, 256, path), {})
        long = seed_values_by_group.get((family, 3584, path), {})
        seeds = sorted(set(short) & set(long))
        differences = [long[seed] - short[seed] for seed in seeds]
        length_dependence.append(
            {
                "stress_family": family,
                "path": path,
                "paired_seed_count": len(seeds),
                "mean_3584_minus_256": mean(differences) if differences else None,
                "bootstrap_95_ci": _bootstrap_mean(
                    differences,
                    seed=bootstrap_seed + 1000 + index,
                    samples=bootstrap_samples,
                ),
            }
        )
    supported = {
        path
        for path in ("ssm", "attention")
        if any(row["path"] == path and row["causal_influence_candidate"] for row in summaries)
    }
    label = (
        "dual_path_causal_influence_candidate"
        if supported == {"ssm", "attention"}
        else "ssm_path_causal_influence_candidate"
        if supported == {"ssm"}
        else "attention_path_causal_influence_candidate"
        if supported == {"attention"}
        else "path_localization_inconclusive"
        if summaries
        else "protocol_invalid"
    )
    return {
        "classification": label,
        "groups": summaries,
        "length_dependence": length_dependence,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_samples": bootstrap_samples,
    }
