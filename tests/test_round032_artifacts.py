"""Round32产物必须从原始logits重算并排除协议无效单元。"""

import json
from pathlib import Path

import pytest

from statefuzz.analyzer.path_localization import compute_transfer_ratio

ROOT = Path(__file__).resolve().parents[1]


def test_round032_artifact_protocol_and_transfer_recomputation():
    artifact = json.loads((ROOT / "results" / "hybrid_path_round_032.json").read_text(encoding="utf-8"))
    assert artifact["protocol"]["seeds"] == list(range(69, 77))
    assert artifact["protocol"]["families"] == ["structured_repetitive", "lexically_diverse"]
    assert artifact["protocol"]["budgets"] == [256, 1792, 3584]
    assert len(artifact["native_records"]) == 48
    keys = {(row["stress_family"], row["target_budget"], row["seed"]) for row in artifact["native_records"]}
    assert len(keys) == 48
    for row in artifact["native_records"]:
        if row["actual_input_tokens"] is not None:
            assert abs(row["actual_input_tokens"] - row["target_budget"]) <= 8
    for record in artifact["intervention_records"]:
        preferences = record["raw_preferences"]
        for path, condition in (("ssm", "ssm_donor_only"), ("attention", "attention_donor_only")):
            if path not in record["paths"]:
                continue
            assert record["paths"][path]["transfer_ratio"] == pytest.approx(
                compute_transfer_ratio(preferences["native_recipient"], preferences["native_donor"], preferences[condition])
            )
    valid_cells = {(row["stress_family"], row["target_budget"], row["seed"]) for row in artifact["intervention_records"] if row["protocol_valid"]}
    for group in artifact["summary"]["groups"]:
        counted = {row["seed"] for row in group["seed_level"]}
        assert all((group["stress_family"], group["target_budget"], seed) in valid_cells for seed in counted)
    for cell in valid_cells:
        directions = {row["direction"] for row in artifact["intervention_records"] if row["protocol_valid"] and (row["stress_family"], row["target_budget"], row["seed"]) == cell}
        assert directions == {"a_from_b", "b_from_a"}
