"""Round34-B：冻结Zamba2直接行为评估与产物验证。"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
from collections import defaultdict
from pathlib import Path
from statistics import mean

for _name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[_name] = "1"

SEEDS = [69, 70, 71, 72]
BUDGETS = [256, 768, 1792, 3584]
MODEL_ID = "Zyphra/Zamba2-1.2B-Instruct-v2"
CHECKPOINT = "/202532803004/models/Zamba2-1.2B-Instruct-v2"
TOLERANCE = 8
VALUE_PAIR = (" red", " blue")
STRESS_MAPPING = {
    "structured_repetition": "structured_repetitive",
    "periodic_interference": "periodic_pattern",
    "distractor_injection": "interleaved_distractor",
    "lexically_diverse": "lexically_diverse",
}


def atomic_save(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def summarize(data: dict) -> dict:
    by_group = defaultdict(list)
    for record in data["records"]:
        by_group[(record["stress_family"], record["token_budget"], record["seed"])].append(record)
    families = {}
    for stress_name in STRESS_MAPPING:
        curve = []
        baseline = None
        first_failure = None
        for budget in BUDGETS:
            cells = []
            for seed in SEEDS:
                rows = by_group.get((stress_name, budget, seed), [])
                if len(rows) == 2:
                    minimum = min(row["signed_margin"] for row in rows)
                    cells.append({"seed": seed, "actual_tokens": rows[0]["actual_tokens"], "min_signed_margin": minimum, "failure": minimum <= 0})
            status_rows = [row for row in data["cell_statuses"] if row["stress_family"] == stress_name and row["token_budget"] == budget]
            point = {
                "token_budget": budget,
                "valid_seed_count": len(cells),
                "expected_seed_count": len(SEEDS),
                "complete_seed_set": len(cells) == len(SEEDS),
                "status_counts": {status: sum(row["runtime_status"] == status for row in status_rows) for status in sorted({row["runtime_status"] for row in status_rows})},
            }
            if cells:
                margin = mean(cell["min_signed_margin"] for cell in cells)
                if budget == BUDGETS[0] and len(cells) == len(SEEDS):
                    baseline = margin
                failures = sum(cell["failure"] for cell in cells)
                point.update(
                    actual_token_range=[min(cell["actual_tokens"] for cell in cells), max(cell["actual_tokens"] for cell in cells)],
                    failure_count=failures,
                    failure_probability=failures / len(cells),
                    mean_min_signed_margin=margin,
                    normalized_margin_retention=margin / baseline if baseline else None,
                    cells=cells,
                )
                if failures and first_failure is None:
                    first_failure = point["actual_token_range"]
            curve.append(point)
        families[stress_name] = {"generator_style": STRESS_MAPPING[stress_name], "curve": curve, "first_failure_boundary": first_failure}
    return {"families": families, "cache_intervention_used": False}


def verify(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["model_id"] == MODEL_ID
    assert data["checkpoint_path"] == CHECKPOINT
    assert data["protocol"]["seeds"] == SEEDS
    assert data["protocol"]["budgets"] == BUDGETS
    assert data["protocol"]["stress_mapping"] == STRESS_MAPPING
    assert data["cache_intervention_used"] is False
    expected_cells = {(stress, budget, seed) for stress in STRESS_MAPPING for budget in BUDGETS for seed in SEEDS}
    statuses = {(row["stress_family"], row["token_budget"], row["seed"]) for row in data["cell_statuses"]}
    assert statuses == expected_cells
    grouped = defaultdict(list)
    for record in data["records"]:
        key = (record["stress_family"], record["token_budget"], record["seed"])
        grouped[key].append(record)
        assert record["model_id"] == MODEL_ID
        assert record["candidate_a"] == VALUE_PAIR[0]
        assert record["candidate_b"] == VALUE_PAIR[1]
        assert record["prompt_variant"] in {"a", "b"}
        assert abs(record["actual_tokens"] - record["token_budget"]) <= TOLERANCE
        expected = record["logits"]["a"] - record["logits"]["b"] if record["prompt_variant"] == "a" else record["logits"]["b"] - record["logits"]["a"]
        assert record["signed_margin"] == expected
    for status in data["cell_statuses"]:
        key = (status["stress_family"], status["token_budget"], status["seed"])
        assert len(grouped.get(key, [])) == (2 if status["runtime_status"] == "ok" else 0)
    print(json.dumps({"verify": "passed", "records": len(data["records"]), "cells": len(statuses)}, ensure_ascii=False))


def run(path: Path) -> None:
    import torch
    import transformers

    from statefuzz.generator.remote_memory import fit_remote_memory_pair_to_token_budget
    from statefuzz.runner.hybrid_causal_lm_runner import HybridCausalLMExperimentConfig, HybridCausalLMRunner

    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "round": 34,
        "phase": "frozen_evaluation",
        "model_id": MODEL_ID,
        "checkpoint_path": CHECKPOINT,
        "model_role": "hybrid_extension",
        "environment": {"hostname": socket.gethostname(), "python": platform.python_version(), "python_executable": os.sys.executable, "torch": torch.__version__, "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0)},
        "protocol": {"seeds": SEEDS, "budgets": BUDGETS, "token_tolerance": TOLERANCE, "candidate_values": list(VALUE_PAIR), "stress_mapping": STRESS_MAPPING},
        "model_freeze": {"config": "configs/model_paths.json", "parameters": 1215064704, "revision": None},
        "records": [], "cell_statuses": [], "runtime_failures": [], "cache_intervention_used": False,
    }
    runner = HybridCausalLMRunner.from_pretrained(HybridCausalLMExperimentConfig())
    candidate_ids = [runner.single_token_id(value) for value in VALUE_PAIR]
    data["candidate_token_ids"] = candidate_ids
    validation = runner.score_candidate_tokens("The remembered word is", candidate_ids)
    data["validation_inference"] = {"input_tokens": validation["input_token_count"], "candidate_logits": {str(item["token_id"]): item["logit"] for item in validation["candidates"]}, "status": "ok"}
    completed = {(row["stress_family"], row["token_budget"], row["seed"]) for row in data["cell_statuses"]}
    for stress_name, generator_style in STRESS_MAPPING.items():
        for budget in BUDGETS:
            for seed in SEEDS:
                if (stress_name, budget, seed) in completed:
                    continue
                fit = fit_remote_memory_pair_to_token_budget(runner.count_tokens, budget, tolerance_tokens=TOLERANCE, seed=seed, template_id=1, value_a=VALUE_PAIR[0], value_b=VALUE_PAIR[1], target_position=0.0, filler_style=generator_style)
                status = {"model_id": MODEL_ID, "stress_family": stress_name, "generator_style": generator_style, "token_budget": budget, "seed": seed, "runtime_status": fit.status, "actual_tokens": fit.actual_tokens}
                if fit.pair is not None:
                    result = runner.score_remote_memory_pair(fit.pair)
                    if not result["matched"] or not result["candidate_valid"] or abs(fit.actual_tokens - budget) > TOLERANCE:
                        status["runtime_status"] = "token_or_candidate_invalid"
                    else:
                        for prompt_variant in ("a", "b"):
                            score = result[f"score_{prompt_variant}"]
                            logits = {item["token_id"]: float(item["logit"]) for item in score["candidates"]}
                            signed_margin = logits[candidate_ids[0]] - logits[candidate_ids[1]] if prompt_variant == "a" else logits[candidate_ids[1]] - logits[candidate_ids[0]]
                            data["records"].append({"seed": seed, "model_id": MODEL_ID, "stress_family": stress_name, "generator_style": generator_style, "token_budget": budget, "actual_tokens": fit.actual_tokens, "prompt_variant": prompt_variant, "candidate_a": VALUE_PAIR[0], "candidate_b": VALUE_PAIR[1], "candidate_token_ids": candidate_ids, "logits": {"a": logits[candidate_ids[0]], "b": logits[candidate_ids[1]]}, "signed_margin": signed_margin})
                        status["runtime_status"] = "ok"
                data["cell_statuses"].append(status)
                atomic_save(path, data)
                print(json.dumps({"cell": [stress_name, budget, seed], "status": status["runtime_status"]}, ensure_ascii=False), flush=True)
    data["summary"] = summarize(data)
    data["status"] = "evaluation_complete"
    atomic_save(path, data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/hybrid_stress_round_034.json")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    path = Path(args.output)
    verify(path) if args.verify else run(path)


if __name__ == "__main__":
    main()
