"""压缩Round33调试记录并生成协议判定。"""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = json.loads(Path(args.raw).read_text(encoding="utf-8"))
    records = []
    signatures = {}
    for source_row in source["records"]:
        row = dict(source_row)
        for prompt_name in ("prompt_a", "prompt_b"):
            if prompt_name in row:
                prompt = dict(row[prompt_name])
                signature = prompt.pop("cache_structure", None)
                if signature is not None:
                    signatures.setdefault(str(row["target_budget"]), signature)
                row[prompt_name] = prompt
        records.append(row)
    evaluated = [row for row in records if row["runtime_status"] in {"ok", "cache_reconstruction_invalid"}]
    valid = [row for row in evaluated if row.get("reconstruction_valid")]
    rate = len(valid) / len(evaluated) if evaluated else 0.0
    groups = []
    for family in source["protocol"]["families"]:
        for budget in source["protocol"]["budgets"]:
            rows = [row for row in records if row["stress_family"] == family and row["target_budget"] == budget]
            measured = [row for row in rows if row["runtime_status"] in {"ok", "cache_reconstruction_invalid"}]
            groups.append(
                {
                    "stress_family": family,
                    "target_budget": budget,
                    "cell_count": len(rows),
                    "evaluated_count": len(measured),
                    "valid_count": sum(row.get("reconstruction_valid", False) for row in measured),
                    "valid_rate": sum(row.get("reconstruction_valid", False) for row in measured) / len(measured) if measured else None,
                    "mean_max_logit_diff": mean(row["max_logit_diff"] for row in measured) if measured else None,
                    "maximum_logit_diff": max((row["max_logit_diff"] for row in measured), default=None),
                    "status_counts": dict(Counter(row["runtime_status"] for row in rows)),
                }
            )
    first_locations = Counter()
    changed_tokens = Counter()
    for row in evaluated:
        for prompt_name in ("prompt_a", "prompt_b"):
            prompt = row[prompt_name]
            first_locations[str(prompt["first_mismatch_location"])] += 1
            changed_tokens[(prompt["top_changed_token"]["token_id"], prompt["top_changed_token"]["token_text"])] += 1
    status = "reconstruction_validated" if rate >= 0.9 else "hybrid_cache_protocol_invalid"
    summary = {
        "evaluated_count": len(evaluated),
        "valid_count": len(valid),
        "reconstruction_valid_rate": rate,
        "required_valid_rate": 0.9,
        "status": status,
        "groups": groups,
        "first_mismatch_location_counts": dict(first_locations),
        "top_changed_tokens": [
            {"token_id": token[0], "token_text": token[1], "count": count}
            for token, count in changed_tokens.most_common(10)
        ],
    }
    artifact = {
        "round": 33,
        "goal": "validate_or_reject_hybrid_cache_reconstruction_protocol",
        "protocol": source["protocol"],
        "protocol_assumption": "Round33未重新列出family，因此沿用Round32冻结的structured_repetitive与lexically_diverse。",
        "environment": source["environment"],
        "checkpoint": {"name": "Zyphra/Zamba2-1.2B-Instruct-v2", "path": "/202532803004/models/Zamba2-1.2B-Instruct-v2", "offline_loading": True},
        "cache_structure_by_budget": signatures,
        "records": records,
        "summary": summary,
        "status": status,
    }
    result = {
        "round": 33,
        "status": status,
        "checkpoint": artifact["checkpoint"],
        "seeds": source["protocol"]["seeds"],
        "families": source["protocol"]["families"],
        "protocol_assumption": artifact["protocol_assumption"],
        "budgets": source["protocol"]["budgets"],
        "replay_semantics": source["protocol"]["replay"],
        "reconstruction_summary": summary,
        "cache_components": {
            "ssm_fields": ["conv_states", "recurrent_states", "ssm_states_if_present"],
            "attention_fields": ["keys", "values"],
            "signatures": signatures,
        },
        "causal_interpretation_performed": False,
        "round034_path_swaps_authorized_by_protocol": status == "reconstruction_validated",
        "claim_scope": "本轮只验证缓存重放协议，不解释SSM或Attention记忆机制，也不比较架构优劣。",
        "failures": [
            {"step": "Round32简化tail重放", "error": "部分候选终点可重建，但全词表逐位置不等价", "resolution": "Round33改用prepare_inputs_for_generation逐token流程并重新验证"},
        ],
        "tests": {"status": "pending"},
        "artifact": "results/cache_reconstruction_round_033.json",
    }
    for name, payload in (("cache_reconstruction_round_033.json", artifact), ("result_round_033.json", result)):
        (root / "results" / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "evaluated": len(evaluated), "valid": len(valid), "rate": rate}, ensure_ascii=False))


if __name__ == "__main__":
    main()
