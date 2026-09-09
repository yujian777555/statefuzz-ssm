import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_paper_figures_artifact_has_four_required_figures() -> None:
    payload = json.loads((ROOT / "results" / "paper_figures_round_022.json").read_text(encoding="utf-8"))
    assert set(payload["figures"]) == {"failure_risk_curve", "intervention_recovery_bar", "memory_specificity_gap", "state_intervention_schematic"}
    assert len(payload["figures"]["state_intervention_schematic"]["nodes"]) >= 5


def test_final_manifest_records_scope_and_reproducibility() -> None:
    payload = json.loads((ROOT / "results" / "final_experiment_manifest.json").read_text(encoding="utf-8"))
    assert payload["model_names"] == ["state-spaces/mamba-130m-hf"]
    assert "all SSMs" in payload["excluded_claims"]
    assert payload["reproducibility"]["test_command"].startswith("python -m pytest")
