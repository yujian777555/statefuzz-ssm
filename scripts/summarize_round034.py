"""比较Round34 Zamba2新证据与Round26历史Mamba/Pythia证据。"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

CURRENT_TO_REFERENCE = {
    "structured_repetition": "structured_repetitive",
    "periodic_interference": "periodic_pattern",
    "distractor_injection": "interleaved_distractor",
    "lexically_diverse": "lexically_diverse",
}


def reference_fingerprint(reference):
    result = {}
    for model_id, payload in reference["models"].items():
        grouped = defaultdict(list)
        for row in payload["records"]:
            grouped[(row["family"], row["target_budget_tokens"])].append(row)
        families = {}
        for current_name, reference_name in CURRENT_TO_REFERENCE.items():
            curve = []
            baseline = None
            for budget in (256, 1792):
                rows = grouped.get((reference_name, budget), [])
                margin = mean(row["min_signed_margin"] for row in rows) if rows else None
                if budget == 256:
                    baseline = margin
                curve.append(
                    {
                        "token_budget": budget,
                        "seed_count": len(rows),
                        "failure_probability": sum(row["min_signed_margin"] <= 0 for row in rows) / len(rows) if rows else None,
                        "mean_min_signed_margin": margin,
                        "normalized_margin_retention": margin / baseline if margin is not None and baseline else None,
                        "actual_token_range": [min(row["actual_input_tokens"] for row in rows), max(row["actual_input_tokens"] for row in rows)] if rows else None,
                    }
                )
            families[current_name] = curve
        result[model_id] = families
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", required=True)
    parser.add_argument("--reference", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    current = json.loads(Path(args.current).read_text(encoding="utf-8"))
    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    historical = reference_fingerprint(reference)
    valid_cells = sum(row["runtime_status"] == "ok" for row in current["cell_statuses"])
    invalid_cells = len(current["cell_statuses"]) - valid_cells
    current_failures = sum(
        point.get("failure_count", 0)
        for family in current["summary"]["families"].values()
        for point in family["curve"]
    )
    current_curves = current["summary"]["families"]
    def current_retention(family, budget):
        return next(point.get("normalized_margin_retention") for point in current_curves[family]["curve"] if point["token_budget"] == budget)
    def historical_retention(model, family, budget):
        return next(point.get("normalized_margin_retention") for point in historical[model][family] if point["token_budget"] == budget)
    descriptive_patterns = {
        "mamba_structured_1792_retention": historical_retention("state-spaces/mamba-130m-hf", "structured_repetition", 1792),
        "pythia_structured_1792_retention": historical_retention("EleutherAI/pythia-160m", "structured_repetition", 1792),
        "zamba_structured_1792_retention": current_retention("structured_repetition", 1792),
        "zamba_periodic_768_retention": current_retention("periodic_interference", 768),
        "zamba_periodic_3584_retention": current_retention("periodic_interference", 3584),
        "interpretation": "模型内margin轨迹存在描述性差异，但所有可评估点失败率均为0，且checkpoint、seed、规模和微调不匹配。",
    }
    comparison = {
        "new_model": current["model_id"],
        "historical_models": list(historical),
        "current_within_model": current["summary"]["families"],
        "historical_within_model": historical,
        "common_comparable_budgets": [256, 1792],
        "comparison_policy": "within-model normalized signed-margin degradation; raw logits are not compared as calibrated values",
        "confounds": ["different checkpoints", "different parameter scale", "instruction tuning", "different seed cohorts", "historical tolerance ±16 versus current ±8"],
        "stable_architecture_dependent_failure_pattern": False,
        "classification": "no_stable_architecture_dependent_failure_pattern",
        "descriptive_patterns": descriptive_patterns,
    }
    result = {
        "round": 34,
        "phase": "frozen_evaluation_complete",
        "status": "cross_architecture_evaluation_complete_negative_evidence",
        "frozen_model_config": "configs/model_paths.json",
        "model_id": current["model_id"],
        "checkpoint_path": current["checkpoint_path"],
        "seeds": current["protocol"]["seeds"],
        "budgets": current["protocol"]["budgets"],
        "stress_mapping": current["protocol"]["stress_mapping"],
        "valid_cells": valid_cells,
        "unavailable_cells": invalid_cells,
        "current_failure_count": current_failures,
        "answers": {
            "statefuzz_transfers_across_architectures": True,
            "failure_surfaces_architecture_dependent": False,
            "identical_stress_families_expose_different_memory_weaknesses": "descriptive_margin_patterns_only_not_stable_failure_boundaries",
        },
        "comparison": comparison,
        "cache_intervention_used": False,
        "claim_scope": "新实验只覆盖冻结的Zamba2扩展checkpoint；Mamba/Pythia仅引用历史行为证据，未进行Hybrid内部因果归因。",
        "failures": [row for row in current["cell_statuses"] if row["runtime_status"] != "ok"],
        "tests": {"status": "pending"},
        "raw_artifact": "results/hybrid_stress_round_034.json",
    }
    (root / "results" / "result_round_034.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"valid_cells": valid_cells, "unavailable_cells": invalid_cells, "failure_count": current_failures, "classification": comparison["classification"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
