import importlib.util

import pytest


SPEC = importlib.util.find_spec("statefuzz.models.registry")
requires_registry = pytest.mark.skipif(SPEC is None, reason="registry尚未实现")


def test_registry_module_exists() -> None:
    assert SPEC is not None


@requires_registry
def test_load_registry_rejects_duplicate_ids(tmp_path) -> None:
    from statefuzz.models.registry import load_registry

    config = tmp_path / "models.yaml"
    config.write_text(
        "models:\n  - id: a\n    role: x\n    family: y\n  - id: a\n    role: z\n    family: y\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="重复"):
        load_registry(config)


@requires_registry
def test_load_registry_preserves_order(tmp_path) -> None:
    from statefuzz.models.registry import load_registry

    config = tmp_path / "models.yaml"
    config.write_text(
        "models:\n  - id: first\n    role: ssm\n    family: mamba\n  - id: second\n    role: control\n    family: transformer\n",
        encoding="utf-8",
    )
    assert [item.model_id for item in load_registry(config)] == ["first", "second"]


@pytest.mark.parametrize("missing_field", ["role", "family"])
def test_load_registry_rejects_missing_semantic_fields(
    tmp_path, missing_field: str
) -> None:
    from statefuzz.models.registry import load_registry

    fields = {"id": "owner/model", "role": "development_ssm", "family": "mamba"}
    fields.pop(missing_field)
    lines = ["models:", "  - " + "\n    ".join(f"{k}: {v}" for k, v in fields.items())]
    config = tmp_path / "models.yaml"
    config.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="缺少必填字段"):
        load_registry(config)


@pytest.mark.parametrize(
    "field,value", [("id", "''"), ("role", "''"), ("family", "''")]
)
def test_load_registry_rejects_empty_semantic_values(
    tmp_path, field: str, value: str
) -> None:
    from statefuzz.models.registry import load_registry

    fields = {
        "id": "owner/model",
        "role": "development_ssm",
        "family": "mamba",
    }
    fields[field] = value
    lines = ["models:", "  - " + "\n    ".join(f"{k}: {v}" for k, v in fields.items())]
    config = tmp_path / "models.yaml"
    config.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="非空"):
        load_registry(config)


@pytest.mark.parametrize("value", ["true", "'false'"])
def test_load_registry_enforces_trust_remote_code_policy(tmp_path, value: str) -> None:
    from statefuzz.models.registry import load_registry

    config = tmp_path / "models.yaml"
    config.write_text(
        "models:\n"
        "  - id: owner/model\n"
        "    role: development_ssm\n"
        "    family: mamba\n"
        f"    trust_remote_code: {value}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="trust_remote_code"):
        load_registry(config)

