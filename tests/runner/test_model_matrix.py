import pytest


def test_frozen_model_matrix_has_ssm_and_transformer_roles() -> None:
    from statefuzz.runner.model_matrix import FROZEN_MODEL_MATRIX, validate_model_matrix

    entries = validate_model_matrix()
    assert {entry.role for entry in entries} == {"ssm_primary", "transformer_control"}
    assert len({entry.architecture for entry in entries}) == 2
    assert all("model_id" in entry.to_dict() for entry in FROZEN_MODEL_MATRIX)


def test_model_matrix_rejects_single_architecture() -> None:
    from statefuzz.runner.model_matrix import ModelMatrixEntry, validate_model_matrix

    entry = ModelMatrixEntry("x", "Mamba", "ssm_primary", "mamba")
    with pytest.raises(ValueError, match="两种架构"):
        validate_model_matrix((entry,))
