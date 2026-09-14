"""从Round32原始偏好与控制记录生成路径定位统计。"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from statefuzz.analyzer.path_localization import (
    compute_transfer_ratio,
    summarize_path_localization,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    raw_path = Path(args.raw)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    path_records = []
    for record in raw["intervention_records"]:
        preferences = record["raw_preferences"]
        recipient, donor = preferences["native_recipient"], preferences["native_donor"]
        for path, condition in (("ssm", "ssm_donor_only"), ("attention", "attention_donor_only")):
            if path not in record["paths"]:
                continue
            intervention = preferences[condition]
            ratio = compute_transfer_ratio(recipient, donor, intervention)
            assert ratio == record["paths"][path]["transfer_ratio"]
            path_records.append(
                {
                    "stress_family": record["stress_family"],
                    "target_budget": record["target_budget"],
                    "seed": record["seed"],
                    "direction": record["direction"],
                    "path": path,
                    "recipient_preference": recipient,
                    "donor_preference": donor,
                    "intervention_preference": intervention,
                    "transfer_ratio": ratio,
                    "toward_donor": (intervention - recipient) * (donor - recipient) > 0,
                    "protocol_valid": record["protocol_valid"],
                }
            )
    summary = summarize_path_localization(path_records)
    reconstruction = []
    for family in raw["protocol"]["families"]:
        for budget in raw["protocol"]["budgets"]:
            rows = [r for r in raw["native_records"] if r["stress_family"] == family and r["target_budget"] == budget]
            reconstruction.append(
                {
                    "stress_family": family,
                    "target_budget": budget,
                    "cell_count": len(rows),
                    "valid_count": sum(r.get("reconstruction_valid", False) for r in rows),
                    "status_counts": dict(Counter(r["runtime_status"] for r in rows)),
                }
            )
    direction_records = raw["intervention_records"]
    controls = {
        "direction_record_count": len(direction_records),
        "protocol_valid_direction_count": sum(r["protocol_valid"] for r in direction_records),
        "cache_clone_invalid_count": sum("cache_clone_protocol_invalid" in r["protocol_failures"] for r in direction_records),
        "full_donor_invalid_count": sum("full_donor_transfer_invalid" in r["protocol_failures"] for r in direction_records),
        "separation_too_small_count": sum("counterfactual_separation_too_small" in r["protocol_failures"] for r in direction_records),
    }
    final_status = "hybrid_path_localization_complete" if summary["groups"] else "hybrid_path_protocol_invalid"
    raw["status"] = final_status
    artifact = {
        "round": 32,
        "protocol": raw["protocol"],
        "environment": raw["environment"],
        "checkpoint": {"name": "Zyphra/Zamba2-1.2B-Instruct-v2", "path": "/202532803004/models/Zamba2-1.2B-Instruct-v2", "offline_loading": True},
        "cache_structure": raw["cache_structure"],
        "short_gate_records": raw["short_gate_records"],
        "native_records": raw["native_records"],
        "intervention_records": raw["intervention_records"],
        "path_records": path_records,
        "reconstruction_summary": reconstruction,
        "control_summary": controls,
        "summary": summary,
        "protocol_failures": raw["protocol_failures"],
        "status": final_status,
        "source_hash": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    }
    result = {
        "round": 32,
        "status": final_status,
        "checkpoint": artifact["checkpoint"],
        "seeds": raw["protocol"]["seeds"],
        "families": raw["protocol"]["families"],
        "budgets": raw["protocol"]["budgets"],
        "reconstruction_summary": reconstruction,
        "sham_full_donor_controls": controls,
        "ssm_path_summary": [row for row in summary["groups"] if row["path"] == "ssm"],
        "attention_path_summary": [row for row in summary["groups"] if row["path"] == "attention"],
        "length_dependence": summary["length_dependence"],
        "classification": summary["classification"],
        "claim_scope": "仅说明该Zamba2检查点中合规路径干预对供体内容偏好的影响；不证明混合架构优越、唯一存储路径或自然失败机制。",
        "known_confounds": {"scale": True, "training_data": True, "instruction_tuning": True, "architecture_causality_confirmed": False},
        "failures": [
            {"step": "首次批量tail重建", "error": "已有cache加8-token批量tail与直接forward不一致", "resolution": "逐token推进后structured部分单元精确重建；其余不合规单元保留并排除"},
            {"step": "重建协议", "error": "部分单元在逐token和float32下仍超过1e-3", "resolution": "标记cache_reconstruction_invalid，不执行干预"},
            {"step": "GPU快速路径", "error": "缺少Zamba2快速kernel", "resolution": "使用naive实现"},
        ],
        "tests": {"status": "pending"},
        "artifact": "results/hybrid_path_round_032.json",
    }
    for name, payload in (("hybrid_path_round_032.json", artifact), ("result_round_032.json", result)):
        (root / "results" / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": final_status, "classification": summary["classification"], "controls": controls}, ensure_ascii=False))


if __name__ == "__main__":
    main()
