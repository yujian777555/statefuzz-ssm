"""Round35：从现有原始证据构建跨架构压力表面。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

MODELS = [
    "state-spaces/mamba-130m-hf",
    "EleutherAI/pythia-160m",
    "Zyphra/Zamba2-1.2B-Instruct-v2",
]
STRESSES = [
    "structured_repetitive",
    "periodic_pattern",
    "interleaved_distractor",
    "lexically_diverse",
]
BUDGETS = [256, 768, 1792, 3584]
ZAMBA_NAMES = {
    "structured_repetition": "structured_repetitive",
    "periodic_interference": "periodic_pattern",
    "distractor_injection": "interleaved_distractor",
    "lexically_diverse": "lexically_diverse",
}


def load_primitives(reference, zamba):
    cells = defaultdict(dict)
    metadata = {}
    for model_id, payload in reference["models"].items():
        metadata[model_id] = {
            "source": "results/reference_round026_for_round031.json",
            "seeds": reference["seeds"],
            "token_tolerance": 16,
            "evidence": "direction-specific signed margins; raw candidate logits were not retained in Round26",
        }
        for row in payload["records"]:
            stress = row["family"]
            if stress not in STRESSES or row["target_budget_tokens"] not in BUDGETS:
                continue
            cells[(model_id, stress, row["target_budget_tokens"])][row["seed"]] = {
                "seed": row["seed"],
                "actual_tokens": row["actual_input_tokens"],
                "direction_margins": [row["signed_margin_a"], row["signed_margin_b"]],
                "raw_logits": None,
            }
    zamba_id = zamba["model_id"]
    metadata[zamba_id] = {
        "source": "results/hybrid_stress_round_034.json",
        "seeds": zamba["protocol"]["seeds"],
        "token_tolerance": zamba["protocol"]["token_tolerance"],
        "evidence": "raw candidate logits and direction-specific signed margins",
    }
    grouped = defaultdict(list)
    for record in zamba["records"]:
        stress = ZAMBA_NAMES[record["stress_family"]]
        grouped[(stress, record["token_budget"], record["seed"])].append(record)
    for (stress, budget, seed), rows in grouped.items():
        if len(rows) != 2:
            continue
        rows = sorted(rows, key=lambda row: row["prompt_variant"])
        cells[(zamba_id, stress, budget)][seed] = {
            "seed": seed,
            "actual_tokens": rows[0]["actual_tokens"],
            "direction_margins": [rows[0]["signed_margin"], rows[1]["signed_margin"]],
            "raw_logits": {
                row["prompt_variant"]: row["logits"] for row in rows
            },
        }
    return cells, metadata


def build_surface(reference, zamba):
    cells, metadata = load_primitives(reference, zamba)
    matrix = {}
    for model_id in MODELS:
        model_surface = {}
        for stress in STRESSES:
            points = []
            baseline = None
            first_failure = None
            previous_margin = None
            nonmonotonic_increases = []
            for budget in BUDGETS:
                primitives = list(cells.get((model_id, stress, budget), {}).values())
                point = {
                    "token_budget": budget,
                    "available": bool(primitives),
                    "seed_count": len(primitives),
                    "expected_seeds": metadata[model_id]["seeds"],
                    "token_tolerance": metadata[model_id]["token_tolerance"],
                    "primitive_records": primitives,
                }
                if primitives:
                    direction_margins = [margin for cell in primitives for margin in cell["direction_margins"]]
                    minimums = [min(cell["direction_margins"]) for cell in primitives]
                    signed_margin = mean(direction_margins)
                    if baseline is None:
                        baseline = signed_margin
                    failures = sum(value <= 0 for value in minimums)
                    point.update(
                        actual_token_range=[min(cell["actual_tokens"] for cell in primitives), max(cell["actual_tokens"] for cell in primitives)],
                        mean_signed_margin=signed_margin,
                        mean_min_signed_margin=mean(minimums),
                        normalized_margin_retention=signed_margin / baseline if baseline else None,
                        failure_count=failures,
                        failure_probability=failures / len(primitives),
                    )
                    if failures and first_failure is None:
                        first_failure = point["actual_token_range"]
                    if previous_margin is not None and signed_margin > previous_margin:
                        nonmonotonic_increases.append({"from_budget": points[-1]["token_budget"], "to_budget": budget})
                    previous_margin = signed_margin
                else:
                    point["missing_reason"] = "not_collected_or_budget_unreachable"
                points.append(point)
            model_surface[stress] = {
                "points": points,
                "first_observed_failure_boundary": first_failure,
                "nonmonotonic_margin_increases": nonmonotonic_increases,
            }
        matrix[model_id] = model_surface
    return matrix, metadata


def verify(surface):
    assert surface["models"] == MODELS
    assert surface["stress_families"] == STRESSES
    assert surface["token_budgets"] == BUDGETS
    assert surface["cache_intervention_used"] is False
    for model_id in MODELS:
        for stress in STRESSES:
            family = surface["matrix"][model_id][stress]
            assert [point["token_budget"] for point in family["points"]] == BUDGETS
            baseline = None
            first_failure = None
            for point in family["points"]:
                records = point["primitive_records"]
                if not records:
                    assert point["available"] is False
                    continue
                margins = [value for record in records for value in record["direction_margins"]]
                minimums = [min(record["direction_margins"]) for record in records]
                calculated = mean(margins)
                if baseline is None:
                    baseline = calculated
                assert point["mean_signed_margin"] == calculated
                assert point["mean_min_signed_margin"] == mean(minimums)
                assert point["normalized_margin_retention"] == calculated / baseline
                assert point["failure_probability"] == sum(value <= 0 for value in minimums) / len(minimums)
                assert all(abs(record["actual_tokens"] - point["token_budget"]) <= point["token_tolerance"] for record in records)
                for record in records:
                    if record["raw_logits"] is not None:
                        assert record["direction_margins"][0] == record["raw_logits"]["a"]["a"] - record["raw_logits"]["a"]["b"]
                        assert record["direction_margins"][1] == record["raw_logits"]["b"]["b"] - record["raw_logits"]["b"]["a"]
                if point["failure_count"] and first_failure is None:
                    first_failure = point["actual_token_range"]
            assert family["first_observed_failure_boundary"] == first_failure
    print(json.dumps({"verify": "passed", "models": len(MODELS), "stress_families": len(STRESSES), "budgets": len(BUDGETS)}, ensure_ascii=False))


def run(root: Path):
    reference = json.loads((root / "results/reference_round026_for_round031.json").read_text(encoding="utf-8"))
    zamba = json.loads((root / "results/hybrid_stress_round_034.json").read_text(encoding="utf-8"))
    matrix, metadata = build_surface(reference, zamba)
    surface = {
        "round": 35,
        "models": MODELS,
        "stress_families": STRESSES,
        "token_budgets": BUDGETS,
        "model_evidence": metadata,
        "matrix": matrix,
        "cache_intervention_used": False,
        "analysis_policy": "within-model signed-margin retention; unavailable budgets remain missing; no raw cross-model calibration assumption",
    }
    verify(surface)
    all_failure_boundaries = [matrix[model][stress]["first_observed_failure_boundary"] for model in MODELS for stress in STRESSES]
    result = {
        "round": 35,
        "status": "stress_surface_analysis_complete",
        "research_answer": "StateFuzz reveals architecture-specific descriptive margin surfaces rather than a universal monotonic degradation curve, but these cohorts contain no observed failure boundary.",
        "architecture_specific_surface_candidate": True,
        "architecture_specific_failure_boundary_confirmed": False,
        "observed_failure_boundary_count": sum(value is not None for value in all_failure_boundaries),
        "negative_evidence": "No evaluated architecture/stress/budget cohort produced a failure probability above zero; failure boundaries must remain null.",
        "key_patterns": {
            "mamba_structured": "large normalized margin loss by 1792",
            "pythia_historical": "stable or increased normalized margin at 1792",
            "zamba_periodic": "strong dip at 768 followed by recovery at longer budgets",
        },
        "limitations": ["Mamba/Pythia only have 256 and 1792 historical budgets", "historical cohorts use seeds 61-62 and ±16 token tolerance", "Zamba2 uses seeds 69-72 and ±8 tolerance", "model scale, training data and instruction tuning are unmatched", "periodic_pattern ignores seed and is not four independent prompt texts"],
        "cache_intervention_used": False,
        "artifact": "results/stress_surface_round_035.json",
        "tests": {"status": "pending"},
    }
    (root / "results/stress_surface_round_035.json").write_text(json.dumps(surface, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (root / "results/result_round_035.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "failure_boundaries": result["observed_failure_boundary_count"]}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = root / "results/stress_surface_round_035.json"
    if args.verify:
        verify(json.loads(path.read_text(encoding="utf-8")))
    else:
        run(root)


if __name__ == "__main__":
    main()
