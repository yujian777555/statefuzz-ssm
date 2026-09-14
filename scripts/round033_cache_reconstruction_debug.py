"""验证Zamba2 body-cache-tail重放是否等价于直接完整forward。"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
from pathlib import Path

for _name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ[_name] = "1"

import torch
import transformers

from statefuzz.generator.remote_memory import fit_remote_memory_pair_to_token_budget
from statefuzz.runner.hybrid_cache_intervention import cache_structure_signature
from statefuzz.runner.hybrid_causal_lm_runner import (
    HybridCausalLMExperimentConfig,
    HybridCausalLMRunner,
)

SEEDS = list(range(69, 77))
FAMILIES = ["structured_repetitive", "lexically_diverse"]
BUDGETS = [256, 1792, 3584]
VALUE_PAIR = (" red", " blue")
TAIL_LENGTH = 8
TOKEN_TOLERANCE = 8
LOGIT_TOLERANCE = 1e-3


def atomic_save(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def fit_pair(runner, family: str, budget: int, seed: int):
    return fit_remote_memory_pair_to_token_budget(
        runner.count_tokens,
        budget,
        tolerance_tokens=TOKEN_TOLERANCE,
        seed=seed,
        template_id=1,
        value_a=VALUE_PAIR[0],
        value_b=VALUE_PAIR[1],
        target_position=0.0,
        filler_style=family,
    )


def replay_generation_path(model, full_ids, body_length: int):
    body = full_ids[:, :body_length]
    tail = full_ids[:, body_length:]
    with torch.inference_mode():
        prefill = model(input_ids=body, use_cache=True, return_dict=True)
    cache = prefill.past_key_values
    signature = cache_structure_signature(cache)
    replay_logits = []
    for index in range(tail.shape[1]):
        full_prefix = full_ids[:, : body_length + index + 1]
        attention_mask = torch.ones_like(full_prefix)
        cache_position = torch.tensor(
            [body_length + index], dtype=torch.long, device=full_ids.device
        )
        prepared = model.prepare_inputs_for_generation(
            full_prefix,
            past_key_values=cache,
            attention_mask=attention_mask,
            cache_position=cache_position,
            use_cache=True,
        )
        with torch.inference_mode():
            output = model(**prepared, return_dict=True)
        cache = output.past_key_values
        replay_logits.append(output.logits[0, -1].float().cpu())
    return torch.stack(replay_logits), signature


def evaluate_prompt(runner, prompt: str, candidate_ids: list[int]):
    model = runner._model
    full_ids = runner.encode_prompt_ids(prompt).to(model.device)
    body_length = int(full_ids.shape[1]) - TAIL_LENGTH
    with torch.inference_mode():
        direct_output = model(input_ids=full_ids, use_cache=True, return_dict=True)
        direct = direct_output.logits[0, -TAIL_LENGTH:].float().cpu()
    replay, signature = replay_generation_path(model, full_ids, body_length)
    difference = (direct - replay).abs()
    flat_index = int(torch.argmax(difference).item())
    vocab_size = int(difference.shape[1])
    tail_index, token_id = divmod(flat_index, vocab_size)
    per_position_max = difference.max(dim=1).values
    first_mismatch = next(
        (index for index, value in enumerate(per_position_max.tolist()) if value > LOGIT_TOLERANCE),
        None,
    )
    final_direct = {str(token): float(direct[-1, token]) for token in candidate_ids}
    final_replay = {str(token): float(replay[-1, token]) for token in candidate_ids}
    return {
        "max_logit_diff": float(difference.max().item()),
        "mean_logit_diff": float(difference.mean().item()),
        "top_changed_token": {
            "token_id": token_id,
            "token_text": runner._tokenizer.decode([token_id], skip_special_tokens=False),
            "tail_position": tail_index,
            "absolute_diff": float(difference[tail_index, token_id].item()),
        },
        "first_mismatch_location": first_mismatch,
        "per_tail_position_max_diff": [float(value) for value in per_position_max],
        "final_candidate_logits_direct": final_direct,
        "final_candidate_logits_replay": final_replay,
        "reconstruction_valid": float(difference.max().item()) <= LOGIT_TOLERANCE,
        "cache_structure": signature,
    }


def evaluate_cell(runner, family: str, budget: int, seed: int, candidate_ids: list[int]):
    row = {
        "stress_family": family,
        "target_budget": budget,
        "budget_tolerance": TOKEN_TOLERANCE,
        "seed": seed,
        "tail_length": TAIL_LENGTH,
        "value_pair": list(VALUE_PAIR),
        "candidate_token_ids": candidate_ids,
    }
    fit = fit_pair(runner, family, budget, seed)
    if fit.pair is None:
        return {**row, "runtime_status": fit.status, "actual_input_tokens": None}
    counts = [runner.count_tokens(fit.pair.prompt_a), runner.count_tokens(fit.pair.prompt_b)]
    row.update(actual_input_tokens=fit.actual_tokens, prompt_token_counts=counts)
    if counts != [fit.actual_tokens, fit.actual_tokens] or abs(fit.actual_tokens - budget) > TOKEN_TOLERANCE:
        return {**row, "runtime_status": "token_budget_or_pair_mismatch"}
    ids_a = runner.encode_prompt_ids(fit.pair.prompt_a)
    ids_b = runner.encode_prompt_ids(fit.pair.prompt_b)
    if not torch.equal(ids_a[:, -TAIL_LENGTH:], ids_b[:, -TAIL_LENGTH:]):
        return {**row, "runtime_status": "query_tail_mismatch"}
    try:
        row["prompt_a"] = evaluate_prompt(runner, fit.pair.prompt_a, candidate_ids)
        row["prompt_b"] = evaluate_prompt(runner, fit.pair.prompt_b, candidate_ids)
        row["max_logit_diff"] = max(row["prompt_a"]["max_logit_diff"], row["prompt_b"]["max_logit_diff"])
        row["mean_logit_diff"] = (row["prompt_a"]["mean_logit_diff"] + row["prompt_b"]["mean_logit_diff"]) / 2
        row["reconstruction_valid"] = row["prompt_a"]["reconstruction_valid"] and row["prompt_b"]["reconstruction_valid"]
        row["runtime_status"] = "ok" if row["reconstruction_valid"] else "cache_reconstruction_invalid"
    except Exception as exc:
        row.update(runtime_status="runtime_error", error_type=type(exc).__name__, error=str(exc))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    data = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {
        "round": 33,
        "protocol": {"seeds": SEEDS, "families": FAMILIES, "budgets": BUDGETS, "tail_length": TAIL_LENGTH, "token_tolerance": TOKEN_TOLERANCE, "logit_tolerance": LOGIT_TOLERANCE, "replay": "prefill_body_then_prepare_inputs_for_generation_tail_token_by_token"},
        "environment": {"hostname": socket.gethostname(), "python": platform.python_version(), "python_executable": os.sys.executable, "torch": torch.__version__, "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0)},
        "records": [],
        "status": "running",
    }
    runner = HybridCausalLMRunner.from_pretrained(HybridCausalLMExperimentConfig())
    candidate_ids = [runner.single_token_id(value) for value in VALUE_PAIR]
    cells = [(FAMILIES[0], BUDGETS[0], SEEDS[0])] if args.smoke else [(family, budget, seed) for family in FAMILIES for budget in BUDGETS for seed in SEEDS]
    completed = {(row["stress_family"], row["target_budget"], row["seed"]) for row in data["records"]}
    for family, budget, seed in cells:
        if (family, budget, seed) in completed:
            continue
        row = evaluate_cell(runner, family, budget, seed, candidate_ids)
        data["records"].append(row)
        atomic_save(output, data)
        print(json.dumps({"cell": [family, budget, seed], "status": row["runtime_status"], "max_logit_diff": row.get("max_logit_diff")}, ensure_ascii=False), flush=True)
    valid = [row for row in data["records"] if row.get("reconstruction_valid")]
    evaluated = [row for row in data["records"] if row.get("runtime_status") in {"ok", "cache_reconstruction_invalid"}]
    rate = len(valid) / len(evaluated) if evaluated else 0.0
    data["summary"] = {"valid_count": len(valid), "evaluated_count": len(evaluated), "reconstruction_valid_rate": rate, "success_threshold": 0.9}
    data["status"] = "smoke_complete" if args.smoke else "reconstruction_validated" if rate >= 0.9 else "hybrid_cache_protocol_invalid"
    atomic_save(output, data)


if __name__ == "__main__":
    main()
