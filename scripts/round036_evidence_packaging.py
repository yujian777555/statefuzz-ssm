"""Round36：从Round35压力表面确定性生成论文证据包。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

ARCHITECTURES = {
    "state-spaces/mamba-130m-hf": "Mamba SSM",
    "EleutherAI/pythia-160m": "GPT-NeoX Transformer",
    "Zyphra/Zamba2-1.2B-Instruct-v2": "Hybrid Mamba2-Attention",
}
CHECKPOINT_REFERENCES = {
    "state-spaces/mamba-130m-hf": "results/reference_round026_for_round031.json#state-spaces/mamba-130m-hf",
    "EleutherAI/pythia-160m": "results/reference_round026_for_round031.json#EleutherAI/pythia-160m",
    "Zyphra/Zamba2-1.2B-Instruct-v2": "configs/model_paths.json#models.primary",
}


def build_table(surface):
    rows = []
    for model in surface["models"]:
        for stress in surface["stress_families"]:
            for point in surface["matrix"][model][stress]["points"]:
                common = {
                    "model": model,
                    "architecture": ARCHITECTURES[model],
                    "checkpoint_reference": CHECKPOINT_REFERENCES[model],
                    "stress_family": stress,
                    "token_budget": point["token_budget"],
                    "normalized_margin_retention": point.get("normalized_margin_retention"),
                    "failure_probability": point.get("failure_probability"),
                    "source_artifact": surface["model_evidence"][model]["source"],
                }
                if not point["primitive_records"]:
                    rows.append({**common, "actual_tokens": None, "seed": None, "prompt_direction": None, "signed_margin": None, "raw_logits": None, "availability": "missing"})
                    continue
                for primitive in point["primitive_records"]:
                    for index, direction in enumerate(("a", "b")):
                        raw_logits = primitive["raw_logits"][direction] if primitive["raw_logits"] is not None else None
                        rows.append({**common, "actual_tokens": primitive["actual_tokens"], "seed": primitive["seed"], "prompt_direction": direction, "signed_margin": primitive["direction_margins"][index], "raw_logits": raw_logits, "availability": "observed"})
    return {"round": 36, "schema": "one row per prompt direction; one null placeholder per unavailable model-stress-budget cell", "rows": rows}


def build_figures(surface):
    matrix = []
    curves = []
    comparisons = []
    for model in surface["models"]:
        for stress in surface["stress_families"]:
            family = surface["matrix"][model][stress]
            points = family["points"]
            matrix.append({"model": model, "architecture": ARCHITECTURES[model], "stress_family": stress, "failure_boundary": family["first_observed_failure_boundary"], "budgets": [{"token_budget": point["token_budget"], "available": point["available"], "failure_probability": point.get("failure_probability"), "mean_signed_margin": point.get("mean_signed_margin")} for point in points]})
            curves.append({"model": model, "stress_family": stress, "curve": [{"token_budget": point["token_budget"], "normalized_margin_retention": point.get("normalized_margin_retention"), "available": point["available"]} for point in points]})
    for stress in surface["stress_families"]:
        models = []
        for model in surface["models"]:
            family = surface["matrix"][model][stress]
            available = [point for point in family["points"] if point["available"]]
            retentions = [point["normalized_margin_retention"] for point in available]
            models.append({"model": model, "available_budget_count": len(available), "minimum_margin_retention": min(retentions) if retentions else None, "mean_margin_retention": mean(retentions) if retentions else None, "maximum_failure_probability": max((point["failure_probability"] for point in available), default=None), "nonmonotonic_increase_count": len(family["nonmonotonic_margin_increases"])})
        comparisons.append({"stress_family": stress, "models": models})
    return {"round": 36, "data_only": True, "views": {"architecture_stress_surface_matrix": matrix, "margin_retention_curves": curves, "stress_family_comparison": comparisons}}


def build_summary(surface):
    failure_boundaries = [surface["matrix"][model][stress]["first_observed_failure_boundary"] for model in surface["models"] for stress in surface["stress_families"]]
    return {
        "round": 36,
        "supported_claims": [
            "StateFuzz produces comparable within-model stress surfaces for the tested Mamba, Pythia and Zamba2 checkpoints.",
            "Observed signed-margin trajectories differ by checkpoint and stress family and are not uniformly monotonic.",
            "No evaluated cohort in the packaged evidence has failure probability above zero.",
        ],
        "unsupported_claims": [
            "A universal long-context failure boundary exists.",
            "Observed differences are caused solely by architecture.",
            "Hybrid models are universally more robust than pure SSMs or Transformers.",
            "Invalid Hybrid cache replay supports internal memory-path attribution.",
        ],
        "limitations": [
            "Mamba-130M and Pythia-160M have only 256 and 1792 historical budget evidence.",
            "Historical seeds 61-62 and tolerance ±16 differ from Zamba2 seeds 69-72 and tolerance ±8.",
            "Checkpoint scale, training data and instruction tuning are unmatched.",
            "periodic_pattern does not vary its prompt text with seed.",
            "Unavailable budgets are retained as null and are not interpolated.",
        ],
        "observed_failure_boundary_count": sum(value is not None for value in failure_boundaries),
        "source_artifacts": ["results/stress_surface_round_035.json", "results/reference_round026_for_round031.json", "results/hybrid_stress_round_034.json"],
    }


def package(root: Path):
    surface = json.loads((root / "results/stress_surface_round_035.json").read_text(encoding="utf-8"))
    return build_table(surface), build_figures(surface), build_summary(surface)


def verify(root: Path):
    expected = package(root)
    names = ("evidence_table_round_036.json", "figure_data_round_036.json", "paper_evidence_summary_round_036.json")
    actual = tuple(json.loads((root / "results" / name).read_text(encoding="utf-8")) for name in names)
    assert actual == expected
    required = {"model", "architecture", "checkpoint_reference", "stress_family", "token_budget", "actual_tokens", "seed", "signed_margin", "normalized_margin_retention", "failure_probability"}
    assert all(required <= set(row) for row in actual[0]["rows"])
    for row in actual[0]["rows"]:
        if row["availability"] == "missing":
            assert all(row[field] is None for field in ("actual_tokens", "seed", "signed_margin", "normalized_margin_retention", "failure_probability"))
    assert actual[1]["data_only"] is True
    assert actual[2]["observed_failure_boundary_count"] == 0
    print(json.dumps({"verify": "passed", "table_rows": len(actual[0]["rows"]), "views": list(actual[1]["views"])}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.verify:
        verify(root)
        return
    artifacts = package(root)
    for name, payload in zip(("evidence_table_round_036.json", "figure_data_round_036.json", "paper_evidence_summary_round_036.json"), artifacts, strict=True):
        (root / "results" / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "packaged", "table_rows": len(artifacts[0]["rows"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
