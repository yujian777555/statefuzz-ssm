"""Round37-C 范围冻结校验器测试。"""

import json
from pathlib import Path

import pytest


def _valid_scope() -> dict[str, object]:
    return {
        "round": 37,
        "phase": "scope_freeze",
        "status": "complete",
        "evaluated_models": [
            {"model_id": "state-spaces/mamba-130m-hf", "architecture_role": "Mamba reference"},
            {"model_id": "EleutherAI/pythia-160m", "architecture_role": "Pythia reference"},
            {"model_id": "Zyphra/Zamba2-1.2B-Instruct-v2", "architecture_role": "Zamba2 Hybrid"},
        ],
        "planned_but_unavailable": [
            {"requested_role": "Mamba2", "availability": "unavailable_in_round037_inventory"},
            {"requested_role": "modern pure-Attention baseline", "availability": "unavailable_in_round037_inventory"},
        ],
        "principal_claim": "frozen",
        "supported_findings": [],
        "unsupported_claims": [
            "universal failure boundaries across SSMs",
            "general superiority or inferiority of Mamba, Pythia, or Zamba2",
            "architecture-causal claims from cross-model comparisons",
            "Mamba2 generalization",
            "modern-Attention generalization beyond the historical Pythia reference",
            "universal claims about all SSMs or all Hybrid architectures",
            "Hybrid SSM-vs-Attention internal causal attribution",
            "any claim that no failure exists outside evaluated token budgets",
        ],
        "evidence_sources": [],
        "paper_transition": {"ready": True, "next_phase": "paper_draft"},
    }


def _valid_result() -> dict[str, object]:
    return {
        "round": 37,
        "status": "scope_frozen_paper_transition_ready",
        "model_expansion_attempted": True,
        "new_models_added": 0,
        "reason_no_new_models_added": "No local target checkpoint was found.",
        "final_scope_artifact": "results/final_scope_round_037.json",
        "next_recommended_phase": "paper_draft",
        "tests": {},
    }


def _write_artifacts(root: Path, scope: dict[str, object], result: dict[str, object]) -> None:
    results = root / "results"
    results.mkdir()
    (results / "final_scope_round_037.json").write_text(json.dumps(scope), encoding="utf-8")
    (results / "result_round_037.json").write_text(json.dumps(result), encoding="utf-8")


def test_valid_frozen_scope_passes(tmp_path: Path) -> None:
    from scripts.round037_scope_freeze import verify

    _write_artifacts(tmp_path, _valid_scope(), _valid_result())
    summary = verify(tmp_path)
    assert summary["verify"] == "passed"


def test_missing_mamba2_unavailable_entry_fails(tmp_path: Path) -> None:
    from scripts.round037_scope_freeze import verify

    scope = _valid_scope()
    scope["planned_but_unavailable"] = scope["planned_but_unavailable"][1:]
    _write_artifacts(tmp_path, scope, _valid_result())
    with pytest.raises(ValueError, match="Mamba2"):
        verify(tmp_path)


@pytest.mark.parametrize(
    "omitted_claim",
    _valid_scope()["unsupported_claims"],
)
def test_omitting_prohibited_overclaim_fails(tmp_path: Path, omitted_claim: str) -> None:
    from scripts.round037_scope_freeze import verify

    scope = _valid_scope()
    scope["unsupported_claims"] = [
        claim for claim in scope["unsupported_claims"] if claim != omitted_claim
    ]
    _write_artifacts(tmp_path, scope, _valid_result())
    with pytest.raises(ValueError, match="unsupported_claims"):
        verify(tmp_path)


def test_fabricated_newly_evaluated_model_fails(tmp_path: Path) -> None:
    from scripts.round037_scope_freeze import verify

    scope = _valid_scope()
    scope["evaluated_models"].append(
        {
            "model_id": "fabricated/new-model",
            "architecture_role": "fabricated baseline",
            "newly_evaluated_in_round037": True,
        }
    )
    _write_artifacts(tmp_path, scope, _valid_result())
    with pytest.raises(ValueError, match="newly evaluated"):
        verify(tmp_path)


def test_unflagged_extra_evaluated_model_fails(tmp_path: Path) -> None:
    from scripts.round037_scope_freeze import verify

    scope = _valid_scope()
    scope["evaluated_models"].append(
        {"model_id": "fabricated/unflagged-model", "architecture_role": "extra baseline"}
    )
    _write_artifacts(tmp_path, scope, _valid_result())
    with pytest.raises(ValueError, match="unexpected evaluated_models"):
        verify(tmp_path)


def test_nonzero_new_models_added_fails(tmp_path: Path) -> None:
    from scripts.round037_scope_freeze import verify

    result = _valid_result()
    result["new_models_added"] = 1
    _write_artifacts(tmp_path, _valid_scope(), result)
    with pytest.raises(ValueError, match="new_models_added"):
        verify(tmp_path)


@pytest.mark.parametrize("missing_model_id", [
    "state-spaces/mamba-130m-hf",
    "EleutherAI/pythia-160m",
    "Zyphra/Zamba2-1.2B-Instruct-v2",
])
def test_missing_frozen_evaluated_role_fails(tmp_path: Path, missing_model_id: str) -> None:
    from scripts.round037_scope_freeze import verify

    scope = _valid_scope()
    scope["evaluated_models"] = [
        model for model in scope["evaluated_models"] if model["model_id"] != missing_model_id
    ]
    _write_artifacts(tmp_path, scope, _valid_result())
    with pytest.raises(ValueError, match="evaluated_models"):
        verify(tmp_path)


def test_missing_modern_attention_unavailable_entry_fails(tmp_path: Path) -> None:
    from scripts.round037_scope_freeze import verify

    scope = _valid_scope()
    scope["planned_but_unavailable"] = scope["planned_but_unavailable"][:1]
    _write_artifacts(tmp_path, scope, _valid_result())
    with pytest.raises(ValueError, match="modern pure-Attention"):
        verify(tmp_path)


def test_paper_transition_not_ready_fails(tmp_path: Path) -> None:
    from scripts.round037_scope_freeze import verify

    scope = _valid_scope()
    scope["paper_transition"]["ready"] = False
    _write_artifacts(tmp_path, scope, _valid_result())
    with pytest.raises(ValueError, match="paper_transition.ready"):
        verify(tmp_path)


def test_invalid_scope_json_fails(tmp_path: Path) -> None:
    from scripts.round037_scope_freeze import verify

    _write_artifacts(tmp_path, _valid_scope(), _valid_result())
    (tmp_path / "results" / "final_scope_round_037.json").write_text("{broken", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        verify(tmp_path)
