"""Round33产物必须覆盖冻结矩阵并由90%门槛决定状态。"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_round033_reconstruction_matrix_and_status():
    artifact = json.loads((ROOT / "results" / "cache_reconstruction_round_033.json").read_text(encoding="utf-8"))
    result = json.loads((ROOT / "results" / "result_round_033.json").read_text(encoding="utf-8"))
    protocol = artifact["protocol"]
    assert protocol["seeds"] == list(range(69, 77))
    assert protocol["budgets"] == [256, 1792, 3584]
    assert len(artifact["records"]) == 48
    keys = {(row["stress_family"], row["target_budget"], row["seed"]) for row in artifact["records"]}
    assert len(keys) == 48
    measured = [row for row in artifact["records"] if row["runtime_status"] in {"ok", "cache_reconstruction_invalid"}]
    valid = [row for row in measured if row.get("max_logit_diff", float("inf")) <= 1e-3]
    rate = len(valid) / len(measured) if measured else 0.0
    assert artifact["summary"]["reconstruction_valid_rate"] == rate
    expected = "reconstruction_validated" if rate >= 0.9 else "hybrid_cache_protocol_invalid"
    assert artifact["status"] == result["status"] == expected
    assert result["causal_interpretation_performed"] is False
    for row in measured:
        assert row["prompt_a"]["first_mismatch_location"] is None or 0 <= row["prompt_a"]["first_mismatch_location"] < 8
        assert row["prompt_b"]["top_changed_token"]["absolute_diff"] <= row["max_logit_diff"]
