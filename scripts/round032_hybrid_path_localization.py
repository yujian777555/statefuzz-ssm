"""Round32：Zamba2混合缓存路径的受控供体迁移实验。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
from pathlib import Path

for _name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[_name] = "1"

import torch
import transformers

from statefuzz.analyzer.path_localization import compute_transfer_ratio
from statefuzz.generator.remote_memory import fit_remote_memory_pair_to_token_budget
from statefuzz.runner.hybrid_cache_intervention import (
    cache_structure_signature,
    clone_cache_independent,
    swap_cache_path,
)
from statefuzz.runner.hybrid_causal_lm_runner import (
    HybridCausalLMExperimentConfig,
    HybridCausalLMRunner,
)

SEEDS = list(range(69, 77))
FAMILIES = ["structured_repetitive", "lexically_diverse"]
BUDGETS = [256, 1792, 3584]
VALUE_PAIR = (" red", " blue")
TAIL_LENGTH = 8
TOLERANCE = 8
LOGIT_TOLERANCE = 1e-3


def atomic_save(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def candidate_logits(score: dict, candidate_ids: list[int]) -> dict[str, float]:
    by_id = {int(item["token_id"]): float(item["logit"]) for item in score["candidates"]}
    return {"candidate_a": by_id[candidate_ids[0]], "candidate_b": by_id[candidate_ids[1]]}


def raw_preference(logits: dict[str, float]) -> float:
    return logits["candidate_a"] - logits["candidate_b"]


def fit_pair(runner, family: str, budget: int, seed: int):
    return fit_remote_memory_pair_to_token_budget(
        runner.count_tokens,
        budget,
        tolerance_tokens=TOLERANCE,
        seed=seed,
        template_id=1,
        value_a=VALUE_PAIR[0],
        value_b=VALUE_PAIR[1],
        target_position=0.0,
        filler_style=family,
    )


def reconstruct_cell(runner, family: str, budget: int, seed: int, candidate_ids: list[int]):
    base = {
        "stress_family": family,
        "target_budget": budget,
        "budget_tolerance": TOLERANCE,
        "seed": seed,
        "value_pair": list(VALUE_PAIR),
        "candidate_token_ids": candidate_ids,
    }
    fit = fit_pair(runner, family, budget, seed)
    if fit.pair is None:
        return {**base, "runtime_status": fit.status, "actual_input_tokens": None}, None
    pair = fit.pair
    ids_a = runner.encode_prompt_ids(pair.prompt_a)
    ids_b = runner.encode_prompt_ids(pair.prompt_b)
    actual = int(ids_a.shape[-1])
    if actual != int(ids_b.shape[-1]) or abs(actual - budget) > TOLERANCE:
        return {**base, "runtime_status": "token_budget_or_pair_mismatch", "actual_input_tokens": actual}, None
    tail_a, tail_b = ids_a[:, -TAIL_LENGTH:], ids_b[:, -TAIL_LENGTH:]
    if not torch.equal(tail_a, tail_b):
        return {**base, "runtime_status": "query_tail_mismatch", "actual_input_tokens": actual}, None
    direct_a = candidate_logits(runner.score_candidate_tokens(pair.prompt_a, candidate_ids), candidate_ids)
    direct_b = candidate_logits(runner.score_candidate_tokens(pair.prompt_b, candidate_ids), candidate_ids)
    cache_a = runner.run_body_to_cache(ids_a[:, :-TAIL_LENGTH])
    cache_b = runner.run_body_to_cache(ids_b[:, :-TAIL_LENGTH])
    reconstructed_a = runner.score_tail_from_cache(tail_a, cache_a, candidate_ids)["candidate_logits"]
    reconstructed_b = runner.score_tail_from_cache(tail_a, cache_b, candidate_ids)["candidate_logits"]
    reconstructed_a = {"candidate_a": reconstructed_a[candidate_ids[0]], "candidate_b": reconstructed_a[candidate_ids[1]]}
    reconstructed_b = {"candidate_a": reconstructed_b[candidate_ids[0]], "candidate_b": reconstructed_b[candidate_ids[1]]}
    errors = {
        "prompt_a_candidate_a": abs(direct_a["candidate_a"] - reconstructed_a["candidate_a"]),
        "prompt_a_candidate_b": abs(direct_a["candidate_b"] - reconstructed_a["candidate_b"]),
        "prompt_b_candidate_a": abs(direct_b["candidate_a"] - reconstructed_b["candidate_a"]),
        "prompt_b_candidate_b": abs(direct_b["candidate_b"] - reconstructed_b["candidate_b"]),
    }
    reconstruction_valid = max(errors.values()) <= LOGIT_TOLERANCE
    native = {
        **base,
        "runtime_status": "ok" if reconstruction_valid else "cache_reconstruction_invalid",
        "actual_input_tokens": actual,
        "tail_token_ids": tail_a[0].tolist(),
        "tail_identical": True,
        "direct_logits": {"prompt_a": direct_a, "prompt_b": direct_b},
        "reconstructed_logits": {"prompt_a": reconstructed_a, "prompt_b": reconstructed_b},
        "reconstruction_absolute_errors": errors,
        "reconstruction_valid": reconstruction_valid,
        "raw_preference_prompt_a": raw_preference(direct_a),
        "raw_preference_prompt_b": raw_preference(direct_b),
        "native_directionality_valid": raw_preference(direct_a) > 0 and raw_preference(direct_b) < 0,
        "prompt_sha256": [hashlib.sha256(prompt.encode()).hexdigest() for prompt in (pair.prompt_a, pair.prompt_b)],
    }
    return native, {"tail": tail_a, "cache_a": cache_a, "cache_b": cache_b, "direct_a": direct_a, "direct_b": direct_b}


def score_swapped(runner, tail, recipient_cache, donor_cache, path, candidate_ids):
    cache = clone_cache_independent(recipient_cache) if path == "sham" else swap_cache_path(recipient_cache, donor_cache, path)
    values = runner.score_tail_from_cache(tail, cache, candidate_ids)["candidate_logits"]
    return {"candidate_a": values[candidate_ids[0]], "candidate_b": values[candidate_ids[1]]}


def intervene_direction(runner, native, state, direction: str, candidate_ids: list[int]):
    if direction == "a_from_b":
        recipient_cache, donor_cache = state["cache_a"], state["cache_b"]
        recipient_logits, donor_logits = state["direct_a"], state["direct_b"]
    else:
        recipient_cache, donor_cache = state["cache_b"], state["cache_a"]
        recipient_logits, donor_logits = state["direct_b"], state["direct_a"]
    conditions = {
        "native_recipient": recipient_logits,
        "native_donor": donor_logits,
        "sham_recipient_clone": score_swapped(runner, state["tail"], recipient_cache, donor_cache, "sham", candidate_ids),
        "full_donor_cache": score_swapped(runner, state["tail"], recipient_cache, donor_cache, "full", candidate_ids),
        "ssm_donor_only": score_swapped(runner, state["tail"], recipient_cache, donor_cache, "ssm", candidate_ids),
        "attention_donor_only": score_swapped(runner, state["tail"], recipient_cache, donor_cache, "attention", candidate_ids),
    }
    preferences = {name: raw_preference(values) for name, values in conditions.items()}
    recipient, donor = preferences["native_recipient"], preferences["native_donor"]
    failures = []
    if abs(preferences["sham_recipient_clone"] - recipient) > LOGIT_TOLERANCE:
        failures.append("cache_clone_protocol_invalid")
    if abs(preferences["full_donor_cache"] - donor) > LOGIT_TOLERANCE:
        failures.append("full_donor_transfer_invalid")
    if abs(donor - recipient) < 0.25:
        failures.append("counterfactual_separation_too_small")
    protocol_valid = not failures
    paths = {}
    if abs(donor - recipient) >= 0.25:
        for path, condition in (("ssm", "ssm_donor_only"), ("attention", "attention_donor_only")):
            intervention = preferences[condition]
            paths[path] = {
                "transfer_ratio": compute_transfer_ratio(recipient, donor, intervention),
                "toward_donor": (intervention - recipient) * (donor - recipient) > 0,
            }
    return {
        "stress_family": native["stress_family"],
        "target_budget": native["target_budget"],
        "actual_input_tokens": native["actual_input_tokens"],
        "seed": native["seed"],
        "direction": direction,
        "candidate_token_ids": candidate_ids,
        "condition_logits": conditions,
        "raw_preferences": preferences,
        "protocol_failures": failures,
        "protocol_valid": protocol_valid,
        "paths": paths,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    data = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {
        "round": 32,
        "protocol": {"seeds": SEEDS, "families": FAMILIES, "budgets": BUDGETS, "tolerance": TOLERANCE, "tail_length": TAIL_LENGTH, "value_pair": list(VALUE_PAIR)},
        "environment": {"hostname": socket.gethostname(), "python": platform.python_version(), "python_executable": os.sys.executable, "torch": torch.__version__, "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0)},
        "short_gate_records": [], "native_records": [], "intervention_records": [], "protocol_failures": [], "cache_structure": {}, "status": "running",
    }
    runner = HybridCausalLMRunner.from_pretrained(HybridCausalLMExperimentConfig())
    candidate_ids = [runner.single_token_id(value) for value in VALUE_PAIR]
    if None in candidate_ids or candidate_ids[0] == candidate_ids[1]:
        data["status"] = "hybrid_short_probe_invalid"; atomic_save(output, data); return
    if len(data["short_gate_records"]) != len(FAMILIES) * len(SEEDS):
        data["short_gate_records"] = []
        for family in FAMILIES:
            for seed in SEEDS:
                native, _ = reconstruct_cell(runner, family, 256, seed, candidate_ids)
                data["short_gate_records"].append(native)
                atomic_save(output, data)
    if not all(
        row.get("actual_input_tokens") is not None
        and abs(row["actual_input_tokens"] - 256) <= TOLERANCE
        and row.get("native_directionality_valid")
        for row in data["short_gate_records"]
    ):
        data["status"] = "hybrid_short_probe_invalid"; atomic_save(output, data); return
    cells = [("structured_repetitive", 256, 69)] if args.smoke else [(family, budget, seed) for family in FAMILIES for budget in BUDGETS for seed in SEEDS]
    completed = {(row["stress_family"], row["target_budget"], row["seed"]) for row in data["native_records"]}
    for family, budget, seed in cells:
        if (family, budget, seed) in completed:
            continue
        native, state = reconstruct_cell(runner, family, budget, seed, candidate_ids)
        data["native_records"].append(native)
        if state is not None and native.get("reconstruction_valid"):
            if not data["cache_structure"]:
                data["cache_structure"] = cache_structure_signature(state["cache_a"])
            for direction in ("a_from_b", "b_from_a"):
                intervention = intervene_direction(runner, native, state, direction, candidate_ids)
                data["intervention_records"].append(intervention)
                for failure in intervention["protocol_failures"]:
                    data["protocol_failures"].append({"cell": [family, budget, seed], "direction": direction, "failure": failure})
        atomic_save(output, data)
        print(json.dumps({"cell": [family, budget, seed], "status": native["runtime_status"]}, ensure_ascii=False), flush=True)
    data["status"] = "smoke_complete" if args.smoke else "hybrid_path_localization_complete"
    atomic_save(output, data)


if __name__ == "__main__":
    main()
